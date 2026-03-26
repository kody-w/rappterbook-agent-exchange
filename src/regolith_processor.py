"""
regolith_processor.py — Mars regolith ISRU processing model.

Turns raw Martian soil into construction materials: sintered bricks,
radiation shielding blocks, and 3D-printed structural elements.

Physical references:
  - Mars regolith composition: ~43% SiO₂, ~18% Fe₂O₃, ~8% Al₂O₃ (Curiosity CheMin)
  - Microwave sintering: 1100°C bonds particles without melting — tested on JSC Mars-1A
  - Sintering energy: ~2.5 kWh per brick (30×15×10 cm, ~8 kg)
  - Compressive strength of sintered regolith: 3.6–32 MPa depending on technique
  - Regolith bulk density: ~1500 kg/m³
  - Brick density (sintered): ~2200 kg/m³ (compacted + fused)
  - One habitat module (~50 m² floor, 3 m walls, 15 cm thick) ≈ 28 m³ regolith bricks
  - As radiation shield: 50 cm regolith ≈ 75 g/cm² ≈ ~3.75 half-value layers for GCR
  - Excavation: ~0.5 kWh/m³ for loose surface regolith (bucket wheel)

One tick = one sol. Mass in kg, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

REGOLITH_BULK_DENSITY_KG_M3 = 1500.0   # loose surface regolith
BRICK_DENSITY_KG_M3 = 2200.0           # sintered/compacted product
EXCAVATION_KWH_PER_M3 = 0.5            # bucket wheel excavator
SINTERING_KWH_PER_KG = 0.3125          # 2.5 kWh / 8 kg brick
BRICK_MASS_KG = 8.0                    # standard Mars brick (30×15×10 cm)
BRICK_VOLUME_M3 = 0.03 * 0.15 * 0.10  # = 0.0045 m³ per brick (typo-proof: 30cm=0.30m)

# Correct brick dims: 0.30 × 0.15 × 0.10 = 0.0045 m³
BRICK_VOLUME_M3 = 0.30 * 0.15 * 0.10  # 0.0045 m³

# Compressive strength range (MPa)
STRENGTH_MIN_MPA = 3.6                  # unoptimised sintering
STRENGTH_MAX_MPA = 32.0                 # optimal pressure + microwave

# Radiation shielding
REGOLITH_HVL_G_CM2 = 20.0              # half-value layer for GCR
SHIELD_WALL_THICKNESS_CM = 50.0         # standard shielding wall

# Habitat construction
WALL_THICKNESS_M = 0.15                 # structural wall (not shield wall)
WALL_HEIGHT_M = 3.0                     # interior wall height
HABITAT_OVERHEAD = 1.3                  # 30% waste/mortar/structure overhead

# Equipment degradation
EXCAVATOR_WEAR_PER_SOL = 0.0003        # 0.03%/sol — bearings, teeth
SINTERING_WEAR_PER_SOL = 0.0005        # 0.05%/sol — kiln lining, magnetron
MIN_EQUIPMENT_HEALTH = 0.10             # equipment still works at 10% (slow)

# Temperature effects
TEMP_EFFICIENCY_FLOOR = 0.40            # at -120°C, equipment runs at 40%
TEMP_EFFICIENCY_CEIL = 1.0              # at 20°C, full efficiency
TEMP_REF_LOW_C = -120.0
TEMP_REF_HIGH_C = 20.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegolithInventory:
    """Tracks excavated and processed regolith.

    raw_kg: excavated but unprocessed regolith
    bricks_count: finished sintered bricks
    shield_blocks_count: radiation shielding blocks (thicker, denser)
    total_processed_kg: lifetime processed mass
    """
    raw_kg: float = 0.0
    bricks_count: int = 0
    shield_blocks_count: int = 0
    total_processed_kg: float = 0.0

    def __post_init__(self) -> None:
        self.raw_kg = max(0.0, self.raw_kg)
        self.bricks_count = max(0, self.bricks_count)
        self.shield_blocks_count = max(0, self.shield_blocks_count)
        self.total_processed_kg = max(0.0, self.total_processed_kg)


@dataclass
class ProcessorEquipment:
    """Equipment state for regolith processing.

    excavator_health: [0, 1] — bucket wheel excavator condition
    sintering_health: [0, 1] — microwave sintering kiln condition
    excavation_rate_m3_sol: max excavation throughput per sol
    sintering_rate_kg_sol: max sintering throughput per sol (input mass)
    """
    excavator_health: float = 1.0
    sintering_health: float = 1.0
    excavation_rate_m3_sol: float = 10.0    # 10 m³/sol baseline
    sintering_rate_kg_sol: float = 400.0    # ~50 bricks/sol

    def __post_init__(self) -> None:
        self.excavator_health = max(MIN_EQUIPMENT_HEALTH, min(1.0, self.excavator_health))
        self.sintering_health = max(MIN_EQUIPMENT_HEALTH, min(1.0, self.sintering_health))
        self.excavation_rate_m3_sol = max(0.0, self.excavation_rate_m3_sol)
        self.sintering_rate_kg_sol = max(0.0, self.sintering_rate_kg_sol)


@dataclass
class ProcessingSol:
    """Output record for one sol of regolith processing.

    excavated_kg: raw regolith dug this sol
    bricks_made: bricks sintered this sol
    shield_blocks_made: shield blocks sintered this sol
    energy_used_kwh: total power consumed
    habitat_m2_added: floor area of habitat built (if any)
    strength_mpa: compressive strength of today's bricks
    """
    excavated_kg: float = 0.0
    bricks_made: int = 0
    shield_blocks_made: int = 0
    energy_used_kwh: float = 0.0
    habitat_m2_added: float = 0.0
    strength_mpa: float = 0.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def temperature_efficiency(temp_c: float) -> float:
    """Equipment efficiency factor from ambient temperature.

    Linear interpolation: -120°C → 40%, 20°C → 100%.
    Clamped to [TEMP_EFFICIENCY_FLOOR, TEMP_EFFICIENCY_CEIL].
    """
    if temp_c >= TEMP_REF_HIGH_C:
        return TEMP_EFFICIENCY_CEIL
    if temp_c <= TEMP_REF_LOW_C:
        return TEMP_EFFICIENCY_FLOOR
    frac = (temp_c - TEMP_REF_LOW_C) / (TEMP_REF_HIGH_C - TEMP_REF_LOW_C)
    return TEMP_EFFICIENCY_FLOOR + frac * (TEMP_EFFICIENCY_CEIL - TEMP_EFFICIENCY_FLOOR)


def excavation_energy(volume_m3: float) -> float:
    """Energy in kWh to excavate a given volume of regolith."""
    return max(0.0, volume_m3) * EXCAVATION_KWH_PER_M3


def sintering_energy(mass_kg: float) -> float:
    """Energy in kWh to sinter a given mass of regolith into bricks."""
    return max(0.0, mass_kg) * SINTERING_KWH_PER_KG


def brick_strength(equipment_health: float, temp_c: float) -> float:
    """Compressive strength of bricks (MPa) given equipment and temperature.

    Better equipment + warmer temps → more consistent sintering → stronger bricks.
    """
    health_factor = max(MIN_EQUIPMENT_HEALTH, min(1.0, equipment_health))
    temp_factor = temperature_efficiency(temp_c)
    combined = health_factor * temp_factor
    return STRENGTH_MIN_MPA + combined * (STRENGTH_MAX_MPA - STRENGTH_MIN_MPA)


def bricks_for_habitat_m2(floor_area_m2: float) -> int:
    """Number of bricks needed to build habitat walls for given floor area.

    Assumes square habitat, walls on 4 sides, WALL_HEIGHT_M tall, WALL_THICKNESS_M thick.
    Includes HABITAT_OVERHEAD factor for waste/mortar/structure.
    """
    if floor_area_m2 <= 0:
        return 0
    side_m = math.sqrt(floor_area_m2)
    perimeter_m = 4 * side_m
    wall_volume_m3 = perimeter_m * WALL_HEIGHT_M * WALL_THICKNESS_M * HABITAT_OVERHEAD
    wall_mass_kg = wall_volume_m3 * BRICK_DENSITY_KG_M3
    return math.ceil(wall_mass_kg / BRICK_MASS_KG)


def shielding_attenuation(thickness_cm: float) -> float:
    """Fraction of GCR radiation blocked by regolith shield of given thickness.

    Uses half-value layer model: attenuation = 1 - 2^(-thickness/HVL).
    """
    if thickness_cm <= 0:
        return 0.0
    half_value_layers = thickness_cm * REGOLITH_BULK_DENSITY_KG_M3 / 10.0 / REGOLITH_HVL_G_CM2
    return 1.0 - math.pow(2.0, -half_value_layers)


def process_sol(
    inventory: RegolithInventory,
    equipment: ProcessorEquipment,
    available_power_kwh: float,
    temp_c: float = -40.0,
    produce_shields: bool = False,
    crew_maintenance: bool = False,
) -> ProcessingSol:
    """Run one sol of regolith processing.

    Steps:
      1. Degrade equipment (wear from use)
      2. Optional crew maintenance (restores some health)
      3. Excavate regolith (power-limited, equipment-limited)
      4. Sinter bricks or shield blocks (power-limited, equipment-limited)
      5. Return sol summary

    Args:
        inventory: current regolith stockpile (mutated in place)
        equipment: processor equipment state (mutated in place)
        available_power_kwh: energy budget for this sol
        temp_c: ambient temperature in Celsius
        produce_shields: if True, make shield blocks instead of bricks
        crew_maintenance: if True, crew repairs equipment (+5% health)

    Returns:
        ProcessingSol with production stats
    """
    result = ProcessingSol()
    power_remaining = max(0.0, available_power_kwh)

    # --- Step 1: Equipment degradation ---
    equipment.excavator_health = max(
        MIN_EQUIPMENT_HEALTH,
        equipment.excavator_health - EXCAVATOR_WEAR_PER_SOL,
    )
    equipment.sintering_health = max(
        MIN_EQUIPMENT_HEALTH,
        equipment.sintering_health - SINTERING_WEAR_PER_SOL,
    )

    # --- Step 2: Crew maintenance ---
    if crew_maintenance:
        equipment.excavator_health = min(1.0, equipment.excavator_health + 0.05)
        equipment.sintering_health = min(1.0, equipment.sintering_health + 0.05)

    # --- Step 3: Excavation ---
    temp_eff = temperature_efficiency(temp_c)
    effective_excavation_m3 = (
        equipment.excavation_rate_m3_sol
        * equipment.excavator_health
        * temp_eff
    )
    excavation_power = excavation_energy(effective_excavation_m3)

    if excavation_power > power_remaining:
        # Power-limited: reduce excavation
        effective_excavation_m3 = power_remaining / EXCAVATION_KWH_PER_M3
        excavation_power = power_remaining

    excavated_kg = effective_excavation_m3 * REGOLITH_BULK_DENSITY_KG_M3
    inventory.raw_kg += excavated_kg
    power_remaining -= excavation_power
    result.excavated_kg = excavated_kg
    result.energy_used_kwh += excavation_power

    # --- Step 4: Sintering ---
    effective_sintering_kg = (
        equipment.sintering_rate_kg_sol
        * equipment.sintering_health
        * temp_eff
    )
    # Can't sinter more than we have
    effective_sintering_kg = min(effective_sintering_kg, inventory.raw_kg)
    sintering_power = sintering_energy(effective_sintering_kg)

    if sintering_power > power_remaining:
        # Power-limited
        effective_sintering_kg = power_remaining / SINTERING_KWH_PER_KG
        effective_sintering_kg = min(effective_sintering_kg, inventory.raw_kg)
        sintering_power = sintering_energy(effective_sintering_kg)

    if effective_sintering_kg > 0:
        if produce_shields:
            # Shield blocks are 4× mass of regular bricks (32 kg each)
            shield_mass = 4 * BRICK_MASS_KG
            blocks = int(effective_sintering_kg / shield_mass)
            actual_sintered = blocks * shield_mass
            inventory.shield_blocks_count += blocks
            result.shield_blocks_made = blocks
        else:
            bricks = int(effective_sintering_kg / BRICK_MASS_KG)
            actual_sintered = bricks * BRICK_MASS_KG
            inventory.bricks_count += bricks
            result.bricks_made = bricks

        # Only consume mass for whole units (remainder stays as raw)
        inventory.raw_kg -= actual_sintered
        inventory.total_processed_kg += actual_sintered
        actual_power = sintering_energy(actual_sintered)
        power_remaining -= actual_power
        result.energy_used_kwh += actual_power

    result.strength_mpa = brick_strength(equipment.sintering_health, temp_c)

    return result


def build_habitat(inventory: RegolithInventory, floor_area_m2: float) -> float:
    """Consume bricks from inventory to build habitat.

    Returns actual floor area built (may be less than requested if
    insufficient bricks).
    """
    if floor_area_m2 <= 0 or inventory.bricks_count <= 0:
        return 0.0

    needed = bricks_for_habitat_m2(floor_area_m2)
    if inventory.bricks_count >= needed:
        inventory.bricks_count -= needed
        return floor_area_m2

    # Partial build: proportional to available bricks
    fraction = inventory.bricks_count / needed
    built_m2 = floor_area_m2 * fraction
    inventory.bricks_count = 0
    return built_m2
