"""
Tests for regolith_processor.py — Mars ISRU regolith processing.

Coverage:
  - Physical constants sanity (positive, reasonable)
  - Temperature efficiency curve (boundaries, monotonicity)
  - Energy calculations (non-negative, proportional)
  - Brick strength model (bounded, monotone in health/temp)
  - Habitat brick estimation (positive for positive area, zero for zero)
  - Shielding attenuation (bounded [0,1], monotone in thickness)
  - Equipment degradation (monotone decrease, floor respected)
  - Crew maintenance (health increases)
  - Power-limited processing (respects budget)
  - Inventory conservation (mass in = mass out)
  - Multi-sol simulation (10 sols without crash)
  - Edge cases (zero power, zero inventory, extreme temps)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from regolith_processor import (
    BRICK_MASS_KG,
    BRICK_VOLUME_M3,
    EXCAVATOR_WEAR_PER_SOL,
    MIN_EQUIPMENT_HEALTH,
    REGOLITH_BULK_DENSITY_KG_M3,
    SINTERING_KWH_PER_KG,
    SINTERING_WEAR_PER_SOL,
    STRENGTH_MAX_MPA,
    STRENGTH_MIN_MPA,
    TEMP_EFFICIENCY_CEIL,
    TEMP_EFFICIENCY_FLOOR,
    TEMP_REF_HIGH_C,
    TEMP_REF_LOW_C,
    ProcessingSol,
    ProcessorEquipment,
    RegolithInventory,
    brick_strength,
    bricks_for_habitat_m2,
    build_habitat,
    excavation_energy,
    process_sol,
    shielding_attenuation,
    sintering_energy,
    temperature_efficiency,
)


# ===================================================================
# Constants sanity
# ===================================================================

class TestConstants:
    """Physical constants must be positive and physically reasonable."""

    def test_brick_mass_positive(self) -> None:
        assert BRICK_MASS_KG > 0

    def test_brick_volume_positive(self) -> None:
        assert BRICK_VOLUME_M3 > 0

    def test_brick_density_consistent(self) -> None:
        """Brick mass / volume should be close to stated brick density."""
        density = BRICK_MASS_KG / BRICK_VOLUME_M3
        assert 1000 < density < 3000, f"brick density {density} kg/m³ out of range"

    def test_regolith_density_positive(self) -> None:
        assert REGOLITH_BULK_DENSITY_KG_M3 > 0

    def test_sintering_energy_positive(self) -> None:
        assert SINTERING_KWH_PER_KG > 0

    def test_strength_range_ordered(self) -> None:
        assert 0 < STRENGTH_MIN_MPA < STRENGTH_MAX_MPA

    def test_equipment_wear_rates_positive(self) -> None:
        assert EXCAVATOR_WEAR_PER_SOL > 0
        assert SINTERING_WEAR_PER_SOL > 0

    def test_equipment_health_floor_positive(self) -> None:
        assert 0 < MIN_EQUIPMENT_HEALTH < 1

    def test_temp_efficiency_ordered(self) -> None:
        assert 0 < TEMP_EFFICIENCY_FLOOR < TEMP_EFFICIENCY_CEIL <= 1.0


# ===================================================================
# Temperature efficiency
# ===================================================================

class TestTemperatureEfficiency:

    def test_at_ref_low(self) -> None:
        assert temperature_efficiency(TEMP_REF_LOW_C) == TEMP_EFFICIENCY_FLOOR

    def test_at_ref_high(self) -> None:
        assert temperature_efficiency(TEMP_REF_HIGH_C) == TEMP_EFFICIENCY_CEIL

    def test_below_ref_low_clamps(self) -> None:
        assert temperature_efficiency(-200.0) == TEMP_EFFICIENCY_FLOOR

    def test_above_ref_high_clamps(self) -> None:
        assert temperature_efficiency(50.0) == TEMP_EFFICIENCY_CEIL

    def test_monotone_increasing(self) -> None:
        """Warmer → better efficiency."""
        temps = [-120, -80, -40, 0, 20]
        effs = [temperature_efficiency(t) for t in temps]
        for i in range(len(effs) - 1):
            assert effs[i] <= effs[i + 1], f"not monotone at {temps[i]}→{temps[i+1]}"

    def test_midpoint_interpolation(self) -> None:
        mid_temp = (TEMP_REF_LOW_C + TEMP_REF_HIGH_C) / 2
        mid_eff = (TEMP_EFFICIENCY_FLOOR + TEMP_EFFICIENCY_CEIL) / 2
        assert abs(temperature_efficiency(mid_temp) - mid_eff) < 0.01

    def test_output_bounded(self) -> None:
        for t in [-200, -120, -60, 0, 20, 100]:
            eff = temperature_efficiency(t)
            assert TEMP_EFFICIENCY_FLOOR <= eff <= TEMP_EFFICIENCY_CEIL


# ===================================================================
# Energy calculations
# ===================================================================

class TestEnergy:

    def test_excavation_energy_non_negative(self) -> None:
        assert excavation_energy(0) == 0
        assert excavation_energy(5) > 0

    def test_excavation_energy_proportional(self) -> None:
        assert excavation_energy(10) == 2 * excavation_energy(5)

    def test_excavation_energy_negative_volume(self) -> None:
        assert excavation_energy(-5) == 0

    def test_sintering_energy_non_negative(self) -> None:
        assert sintering_energy(0) == 0
        assert sintering_energy(100) > 0

    def test_sintering_energy_proportional(self) -> None:
        assert sintering_energy(200) == 2 * sintering_energy(100)

    def test_sintering_energy_negative_mass(self) -> None:
        assert sintering_energy(-10) == 0

    def test_one_brick_energy(self) -> None:
        """Energy for one brick should be ~2.5 kWh."""
        e = sintering_energy(BRICK_MASS_KG)
        assert 2.0 < e < 3.0, f"one brick energy {e} kWh not ~2.5"


# ===================================================================
# Brick strength
# ===================================================================

class TestBrickStrength:

    def test_bounded(self) -> None:
        for health in [0.1, 0.5, 1.0]:
            for temp in [-120, -40, 20]:
                s = brick_strength(health, temp)
                assert STRENGTH_MIN_MPA <= s <= STRENGTH_MAX_MPA

    def test_monotone_in_health(self) -> None:
        """Better equipment → stronger bricks."""
        s_low = brick_strength(0.2, 0.0)
        s_high = brick_strength(1.0, 0.0)
        assert s_low <= s_high

    def test_monotone_in_temperature(self) -> None:
        """Warmer → stronger bricks."""
        s_cold = brick_strength(0.8, -100)
        s_warm = brick_strength(0.8, 20)
        assert s_cold <= s_warm

    def test_perfect_conditions(self) -> None:
        """Health=1, temp=20°C → max strength."""
        s = brick_strength(1.0, 20.0)
        assert abs(s - STRENGTH_MAX_MPA) < 0.1

    def test_worst_conditions(self) -> None:
        """Health=0.1, temp=-120°C → near min strength."""
        s = brick_strength(MIN_EQUIPMENT_HEALTH, TEMP_REF_LOW_C)
        expected = STRENGTH_MIN_MPA + MIN_EQUIPMENT_HEALTH * TEMP_EFFICIENCY_FLOOR * (STRENGTH_MAX_MPA - STRENGTH_MIN_MPA)
        assert abs(s - expected) < 0.1


# ===================================================================
# Habitat brick estimation
# ===================================================================

class TestBricksForHabitat:

    def test_zero_area(self) -> None:
        assert bricks_for_habitat_m2(0) == 0

    def test_negative_area(self) -> None:
        assert bricks_for_habitat_m2(-10) == 0

    def test_positive_area_needs_bricks(self) -> None:
        n = bricks_for_habitat_m2(50)
        assert n > 0

    def test_larger_area_needs_more_bricks(self) -> None:
        n1 = bricks_for_habitat_m2(25)
        n2 = bricks_for_habitat_m2(100)
        assert n2 > n1

    def test_returns_integer(self) -> None:
        n = bricks_for_habitat_m2(50)
        assert isinstance(n, int)

    def test_reasonable_count(self) -> None:
        """50 m² habitat should need hundreds to low thousands of bricks."""
        n = bricks_for_habitat_m2(50)
        assert 100 < n < 10000, f"50m² needs {n} bricks — suspicious"


# ===================================================================
# Shielding attenuation
# ===================================================================

class TestShieldingAttenuation:

    def test_zero_thickness(self) -> None:
        assert shielding_attenuation(0) == 0.0

    def test_negative_thickness(self) -> None:
        assert shielding_attenuation(-10) == 0.0

    def test_bounded_zero_one(self) -> None:
        for t in [1, 10, 50, 100, 500]:
            a = shielding_attenuation(t)
            assert 0.0 <= a <= 1.0

    def test_monotone_increasing(self) -> None:
        """Thicker shield → more attenuation."""
        thicknesses = [5, 10, 20, 50, 100]
        attens = [shielding_attenuation(t) for t in thicknesses]
        for i in range(len(attens) - 1):
            assert attens[i] <= attens[i + 1]

    def test_50cm_substantial(self) -> None:
        """50 cm of regolith should block >80% of GCR."""
        a = shielding_attenuation(50)
        assert a > 0.80, f"50cm blocks only {a*100:.1f}%"

    def test_approaches_one(self) -> None:
        """Very thick shield → nearly total blocking."""
        a = shielding_attenuation(500)
        assert a > 0.999


# ===================================================================
# Data structure construction
# ===================================================================

class TestDataStructures:

    def test_inventory_defaults(self) -> None:
        inv = RegolithInventory()
        assert inv.raw_kg == 0
        assert inv.bricks_count == 0
        assert inv.shield_blocks_count == 0
        assert inv.total_processed_kg == 0

    def test_inventory_clamps_negative(self) -> None:
        inv = RegolithInventory(raw_kg=-10, bricks_count=-5)
        assert inv.raw_kg == 0
        assert inv.bricks_count == 0

    def test_equipment_defaults(self) -> None:
        eq = ProcessorEquipment()
        assert eq.excavator_health == 1.0
        assert eq.sintering_health == 1.0

    def test_equipment_clamps(self) -> None:
        eq = ProcessorEquipment(excavator_health=2.0, sintering_health=-1.0)
        assert eq.excavator_health == 1.0
        assert eq.sintering_health == MIN_EQUIPMENT_HEALTH


# ===================================================================
# process_sol — core simulation
# ===================================================================

class TestProcessSol:

    def test_zero_power_no_production(self) -> None:
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=0.0)
        assert result.excavated_kg == 0
        assert result.bricks_made == 0
        assert result.energy_used_kwh == 0

    def test_produces_bricks_with_power(self) -> None:
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=200.0, temp_c=0.0)
        assert result.excavated_kg > 0
        assert result.bricks_made > 0
        assert result.energy_used_kwh > 0

    def test_energy_conservation(self) -> None:
        """Energy used must not exceed available power."""
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=50.0, temp_c=-40.0)
        assert result.energy_used_kwh <= 50.0 + 0.001  # float tolerance

    def test_inventory_mass_conservation(self) -> None:
        """Raw regolith excavated - sintered = net raw change."""
        inv = RegolithInventory(raw_kg=1000.0)
        eq = ProcessorEquipment()
        raw_before = inv.raw_kg
        result = process_sol(inv, eq, available_power_kwh=200.0, temp_c=0.0)
        # raw_after = raw_before + excavated - sintered
        sintered_mass = result.bricks_made * BRICK_MASS_KG
        expected_raw = raw_before + result.excavated_kg - sintered_mass
        assert abs(inv.raw_kg - expected_raw) < 1.0  # integer truncation tolerance

    def test_equipment_degrades(self) -> None:
        eq = ProcessorEquipment(excavator_health=1.0, sintering_health=1.0)
        inv = RegolithInventory()
        process_sol(inv, eq, available_power_kwh=100.0)
        assert eq.excavator_health < 1.0
        assert eq.sintering_health < 1.0

    def test_equipment_floor_respected(self) -> None:
        """Even after many sols, equipment stays above floor."""
        eq = ProcessorEquipment(excavator_health=MIN_EQUIPMENT_HEALTH + 0.001)
        inv = RegolithInventory()
        for _ in range(100):
            process_sol(inv, eq, available_power_kwh=100.0)
        assert eq.excavator_health >= MIN_EQUIPMENT_HEALTH
        assert eq.sintering_health >= MIN_EQUIPMENT_HEALTH

    def test_crew_maintenance_restores_health(self) -> None:
        eq = ProcessorEquipment(excavator_health=0.5, sintering_health=0.5)
        inv = RegolithInventory()
        process_sol(inv, eq, available_power_kwh=100.0, crew_maintenance=True)
        # After degradation + maintenance: should be above 0.5
        # Degradation is tiny (~0.0003), maintenance adds 0.05
        assert eq.excavator_health > 0.5
        assert eq.sintering_health > 0.5

    def test_shield_block_production(self) -> None:
        inv = RegolithInventory(raw_kg=5000.0)
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=200.0, temp_c=0.0, produce_shields=True)
        assert result.shield_blocks_made > 0
        assert result.bricks_made == 0  # shields, not bricks

    def test_cold_temperature_reduces_output(self) -> None:
        """Colder → less production."""
        inv_warm = RegolithInventory()
        eq_warm = ProcessorEquipment()
        r_warm = process_sol(inv_warm, eq_warm, available_power_kwh=200.0, temp_c=20.0)

        inv_cold = RegolithInventory()
        eq_cold = ProcessorEquipment()
        r_cold = process_sol(inv_cold, eq_cold, available_power_kwh=200.0, temp_c=-100.0)

        assert r_warm.excavated_kg > r_cold.excavated_kg

    def test_power_limiting(self) -> None:
        """Very low power → very low production."""
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=1.0, temp_c=0.0)
        assert result.energy_used_kwh <= 1.0 + 0.001

    def test_strength_reported(self) -> None:
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=100.0, temp_c=0.0)
        assert STRENGTH_MIN_MPA <= result.strength_mpa <= STRENGTH_MAX_MPA


# ===================================================================
# build_habitat
# ===================================================================

class TestBuildHabitat:

    def test_zero_area(self) -> None:
        inv = RegolithInventory(bricks_count=1000)
        assert build_habitat(inv, 0) == 0.0

    def test_negative_area(self) -> None:
        inv = RegolithInventory(bricks_count=1000)
        assert build_habitat(inv, -10) == 0.0

    def test_no_bricks(self) -> None:
        inv = RegolithInventory(bricks_count=0)
        assert build_habitat(inv, 50) == 0.0

    def test_sufficient_bricks(self) -> None:
        inv = RegolithInventory(bricks_count=100000)
        built = build_habitat(inv, 50.0)
        assert built == 50.0
        assert inv.bricks_count < 100000  # consumed some

    def test_partial_build(self) -> None:
        """Insufficient bricks → partial build."""
        inv = RegolithInventory(bricks_count=10)
        built = build_habitat(inv, 50.0)
        assert 0 < built < 50.0
        assert inv.bricks_count == 0  # all consumed

    def test_consumes_exact_bricks(self) -> None:
        """If enough bricks, exactly the right number consumed."""
        inv = RegolithInventory(bricks_count=100000)
        needed = bricks_for_habitat_m2(50.0)
        build_habitat(inv, 50.0)
        assert inv.bricks_count == 100000 - needed


# ===================================================================
# Multi-sol simulation smoke test
# ===================================================================

class TestMultiSolSmoke:
    """Run 10+ sols of processing without crash."""

    def test_10_sol_brick_production(self) -> None:
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        total_bricks = 0
        total_energy = 0.0

        for sol in range(10):
            # Alternate temperatures to test robustness
            temp = -60 + (sol * 10)
            result = process_sol(inv, eq, available_power_kwh=150.0, temp_c=temp)
            total_bricks += result.bricks_made
            total_energy += result.energy_used_kwh

        assert total_bricks > 0, "10 sols should produce some bricks"
        assert total_energy > 0
        assert inv.raw_kg >= 0
        assert inv.bricks_count == total_bricks
        assert inv.total_processed_kg > 0
        assert eq.excavator_health < 1.0  # degraded
        assert eq.sintering_health < 1.0

    def test_10_sol_shield_production(self) -> None:
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        total_shields = 0

        for _ in range(10):
            result = process_sol(inv, eq, available_power_kwh=150.0, temp_c=-20.0, produce_shields=True)
            total_shields += result.shield_blocks_made

        assert total_shields > 0

    def test_50_sol_with_maintenance(self) -> None:
        """50 sols with periodic maintenance — no crash, equipment holds."""
        inv = RegolithInventory()
        eq = ProcessorEquipment()

        for sol in range(50):
            maintain = (sol % 7 == 0)  # maintenance every 7 sols
            process_sol(inv, eq, available_power_kwh=100.0, temp_c=-40.0, crew_maintenance=maintain)

        assert eq.excavator_health >= MIN_EQUIPMENT_HEALTH
        assert eq.sintering_health >= MIN_EQUIPMENT_HEALTH
        assert inv.bricks_count > 0

    def test_365_sol_full_year(self) -> None:
        """Full Mars year of processing — colony builds real infrastructure."""
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        total_bricks = 0

        for sol in range(365):
            maintain = (sol % 10 == 0)
            temp = -60 + 30 * math.sin(2 * math.pi * sol / 668)  # seasonal variation
            result = process_sol(inv, eq, available_power_kwh=120.0, temp_c=temp, crew_maintenance=maintain)
            total_bricks += result.bricks_made

        # After a full year, should have enough for real construction
        assert total_bricks > 1000, f"365 sols only produced {total_bricks} bricks"
        habitat_built = build_habitat(inv, 100.0)
        assert habitat_built > 0, "A year of brick-making should enable some construction"


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_extreme_cold(self) -> None:
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=100.0, temp_c=-200.0)
        assert result.excavated_kg >= 0
        assert result.energy_used_kwh >= 0

    def test_extreme_heat(self) -> None:
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=100.0, temp_c=100.0)
        assert result.excavated_kg >= 0

    def test_massive_power_budget(self) -> None:
        inv = RegolithInventory()
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=100000.0, temp_c=0.0)
        # Should be equipment-limited, not power-limited
        assert result.energy_used_kwh < 100000.0

    def test_pre_stocked_inventory(self) -> None:
        inv = RegolithInventory(raw_kg=999999.0)
        eq = ProcessorEquipment()
        result = process_sol(inv, eq, available_power_kwh=200.0, temp_c=0.0)
        assert result.bricks_made > 0
        assert inv.raw_kg > 0  # didn't consume all

    def test_nearly_dead_equipment(self) -> None:
        eq = ProcessorEquipment(
            excavator_health=MIN_EQUIPMENT_HEALTH,
            sintering_health=MIN_EQUIPMENT_HEALTH,
        )
        inv = RegolithInventory(raw_kg=1000.0)
        result = process_sol(inv, eq, available_power_kwh=100.0, temp_c=0.0)
        # Still produces something, just slowly
        assert result.energy_used_kwh >= 0
