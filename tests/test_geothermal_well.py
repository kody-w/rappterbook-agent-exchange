"""Tests for geothermal_well.py — Mars Geothermal Energy Extraction.

Organised into sections:
  1. Physical constants
  2. Temperature at depth
  3. Fluid outlet temperature
  4. ORC / Carnot efficiency
  5. Thermal power calculations
  6. Pump parasitic power
  7. Net power
  8. Scaling / degradation
  9. Thermal drawdown
 10. Thermal energy (kWh)
 11. State dataclass
 12. Serialisation round-trip
 13. Tick engine
 14. Conservation of energy
 15. Multi-sol simulation
 16. Property-based invariants (parametrize)
 17. Edge cases
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# ── path setup ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from geothermal_well import (
    # Constants
    MARS_SURFACE_TEMP_K,
    GEOTHERMAL_GRADIENT_K_PER_KM,
    EARTH_GRADIENT_K_PER_KM,
    ROCK_THERMAL_CONDUCTIVITY_W_MK,
    ROCK_HEAT_CAPACITY_J_KGK,
    ROCK_DENSITY_KG_M3,
    FLUID_SPECIFIC_HEAT_J_KGK,
    DEFAULT_WELL_DEPTH_M,
    WELL_BORE_RADIUS_M,
    DEFAULT_FLOW_RATE_KG_S,
    DEFAULT_HEAT_EXCHANGE_EFFICIENCY,
    ORC_CARNOT_FRACTION,
    PUMP_PARASITIC_FRACTION,
    SCALING_DEGRADATION_PER_SOL,
    THERMAL_DRAWDOWN_FRACTION_PER_SOL,
    HOURS_PER_SOL,
    SECONDS_PER_SOL,
    PUMP_EFFICIENCY,
    PRESSURE_DROP_PA,
    MIN_THERMAL_POWER_KW,
    # Functions
    rock_temperature_at_depth,
    fluid_outlet_temperature,
    carnot_efficiency,
    orc_efficiency,
    thermal_power_kw,
    electrical_power_kw,
    pump_power_kw,
    net_power_kw,
    thermal_drawdown,
    apply_scaling_degradation,
    thermal_energy_kwh,
    # Classes
    GeothermalWell,
    TickResult,
    # Engine
    tick,
    run_simulation,
)


# =====================================================================
# 1. Physical constants
# =====================================================================

class TestPhysicalConstants:
    """Validate physical constants against known reference values."""

    def test_mars_surface_temp(self):
        assert MARS_SURFACE_TEMP_K == 210.0

    def test_mars_surface_temp_reasonable(self):
        assert 130.0 <= MARS_SURFACE_TEMP_K <= 310.0

    def test_geothermal_gradient_tharsis(self):
        assert GEOTHERMAL_GRADIENT_K_PER_KM == 15.0

    def test_geothermal_gradient_positive(self):
        assert GEOTHERMAL_GRADIENT_K_PER_KM > 0.0

    def test_earth_gradient_higher_than_mars(self):
        assert EARTH_GRADIENT_K_PER_KM > GEOTHERMAL_GRADIENT_K_PER_KM

    def test_earth_gradient_value(self):
        assert EARTH_GRADIENT_K_PER_KM == 25.0

    def test_rock_thermal_conductivity_basalt(self):
        assert 1.5 <= ROCK_THERMAL_CONDUCTIVITY_W_MK <= 2.5

    def test_rock_heat_capacity(self):
        assert ROCK_HEAT_CAPACITY_J_KGK == 840.0

    def test_rock_density_basalt(self):
        assert 2800 <= ROCK_DENSITY_KG_M3 <= 3100

    def test_fluid_specific_heat(self):
        assert FLUID_SPECIFIC_HEAT_J_KGK == 2400.0

    def test_default_well_depth(self):
        assert DEFAULT_WELL_DEPTH_M == 3000.0

    def test_well_bore_radius(self):
        assert WELL_BORE_RADIUS_M == 0.15

    def test_default_flow_rate(self):
        assert DEFAULT_FLOW_RATE_KG_S == 5.0

    def test_heat_exchange_efficiency_range(self):
        assert 0.60 <= DEFAULT_HEAT_EXCHANGE_EFFICIENCY <= 0.80

    def test_orc_carnot_fraction_range(self):
        assert 0.40 <= ORC_CARNOT_FRACTION <= 0.50

    def test_pump_parasitic_fraction_range(self):
        assert 0.05 <= PUMP_PARASITIC_FRACTION <= 0.15

    def test_hours_per_sol(self):
        assert HOURS_PER_SOL == pytest.approx(24.66, abs=0.01)

    def test_seconds_per_sol(self):
        assert SECONDS_PER_SOL == pytest.approx(HOURS_PER_SOL * 3600.0)

    def test_scaling_degradation_small(self):
        assert 0.0 < SCALING_DEGRADATION_PER_SOL < 0.01

    def test_thermal_drawdown_small(self):
        assert 0.0 < THERMAL_DRAWDOWN_FRACTION_PER_SOL < 0.01

    def test_pump_efficiency_reasonable(self):
        assert 0.5 <= PUMP_EFFICIENCY <= 1.0

    def test_pressure_drop_positive(self):
        assert PRESSURE_DROP_PA > 0.0


# =====================================================================
# 2. Temperature at depth
# =====================================================================

class TestRockTemperature:
    """Test rock_temperature_at_depth calculations."""

    def test_surface_is_surface_temp(self):
        assert rock_temperature_at_depth(0.0) == MARS_SURFACE_TEMP_K

    def test_1km_depth(self):
        expected = MARS_SURFACE_TEMP_K + GEOTHERMAL_GRADIENT_K_PER_KM
        assert rock_temperature_at_depth(1000.0) == pytest.approx(expected)

    def test_3km_default(self):
        expected = 210.0 + 15.0 * 3.0  # 255 K
        assert rock_temperature_at_depth(3000.0) == pytest.approx(expected)

    def test_10km_depth(self):
        expected = 210.0 + 15.0 * 10.0  # 360 K
        assert rock_temperature_at_depth(10000.0) == pytest.approx(expected)

    def test_custom_gradient(self):
        result = rock_temperature_at_depth(2000.0, gradient_k_per_km=5.0)
        expected = 210.0 + 5.0 * 2.0  # 220 K
        assert result == pytest.approx(expected)

    def test_custom_surface_temp(self):
        result = rock_temperature_at_depth(1000.0, surface_temp_k=200.0)
        expected = 200.0 + 15.0
        assert result == pytest.approx(expected)

    def test_negative_depth_clamped(self):
        result = rock_temperature_at_depth(-500.0)
        assert result == MARS_SURFACE_TEMP_K

    def test_zero_gradient(self):
        result = rock_temperature_at_depth(5000.0, gradient_k_per_km=0.0)
        assert result == MARS_SURFACE_TEMP_K

    def test_increases_with_depth(self):
        t1 = rock_temperature_at_depth(1000.0)
        t2 = rock_temperature_at_depth(2000.0)
        t3 = rock_temperature_at_depth(3000.0)
        assert t1 < t2 < t3

    def test_linear_scaling(self):
        t1 = rock_temperature_at_depth(1000.0) - MARS_SURFACE_TEMP_K
        t2 = rock_temperature_at_depth(2000.0) - MARS_SURFACE_TEMP_K
        assert t2 == pytest.approx(2.0 * t1)


# =====================================================================
# 3. Fluid outlet temperature
# =====================================================================

class TestFluidOutletTemperature:
    """Test fluid_outlet_temperature calculations."""

    def test_perfect_exchange(self):
        result = fluid_outlet_temperature(255.0, 210.0, 1.0, 1.0)
        assert result == pytest.approx(255.0)

    def test_zero_exchange(self):
        result = fluid_outlet_temperature(255.0, 210.0, 0.0, 1.0)
        assert result == pytest.approx(210.0)

    def test_75_percent_exchange(self):
        result = fluid_outlet_temperature(255.0, 210.0, 0.75, 1.0)
        expected = 210.0 + 45.0 * 0.75  # 243.75
        assert result == pytest.approx(expected)

    def test_scaling_degrades_output(self):
        full = fluid_outlet_temperature(255.0, 210.0, 0.75, 1.0)
        degraded = fluid_outlet_temperature(255.0, 210.0, 0.75, 0.80)
        assert degraded < full

    def test_zero_scaling_returns_surface(self):
        result = fluid_outlet_temperature(255.0, 210.0, 0.75, 0.0)
        assert result == pytest.approx(210.0)

    def test_never_below_surface(self):
        result = fluid_outlet_temperature(200.0, 210.0, 0.75, 1.0)
        assert result >= 210.0

    def test_clamping_eff_above_one(self):
        result = fluid_outlet_temperature(255.0, 210.0, 1.5, 1.0)
        assert result == pytest.approx(255.0)

    def test_combined_scaling_and_exchange(self):
        result = fluid_outlet_temperature(300.0, 210.0, 0.80, 0.90)
        expected = 210.0 + 90.0 * 0.80 * 0.90
        assert result == pytest.approx(expected)


# =====================================================================
# 4. ORC / Carnot efficiency
# =====================================================================

class TestEfficiency:
    """Test Carnot and ORC efficiency calculations."""

    def test_carnot_basic(self):
        result = carnot_efficiency(255.0, 210.0)
        assert result == pytest.approx(1.0 - 210.0 / 255.0)

    def test_carnot_equal_temps(self):
        assert carnot_efficiency(300.0, 300.0) == 0.0

    def test_carnot_cold_greater(self):
        assert carnot_efficiency(200.0, 300.0) == 0.0

    def test_carnot_zero_hot(self):
        assert carnot_efficiency(0.0, 200.0) == 0.0

    def test_carnot_max_one(self):
        result = carnot_efficiency(1000.0, 1.0)
        assert result < 1.0

    def test_carnot_increases_with_delta(self):
        e1 = carnot_efficiency(250.0, 210.0)
        e2 = carnot_efficiency(300.0, 210.0)
        assert e2 > e1

    def test_orc_below_carnot(self):
        c = carnot_efficiency(255.0, 210.0)
        o = orc_efficiency(255.0, 210.0)
        assert o < c

    def test_orc_is_fraction_of_carnot(self):
        c = carnot_efficiency(255.0, 210.0)
        o = orc_efficiency(255.0, 210.0)
        assert o == pytest.approx(ORC_CARNOT_FRACTION * c)

    def test_orc_equal_temps(self):
        assert orc_efficiency(300.0, 300.0) == 0.0

    def test_orc_custom_fraction(self):
        c = carnot_efficiency(300.0, 210.0)
        o = orc_efficiency(300.0, 210.0, carnot_fraction=0.5)
        assert o == pytest.approx(0.5 * c)

    def test_orc_zero_fraction(self):
        assert orc_efficiency(300.0, 210.0, carnot_fraction=0.0) == 0.0


# =====================================================================
# 5. Thermal power calculations
# =====================================================================

class TestThermalPower:
    """Test thermal_power_kw calculations."""

    def test_basic(self):
        result = thermal_power_kw(5.0, 2400.0, 243.75, 210.0)
        expected = 5.0 * 2400.0 * 33.75 / 1000.0
        assert result == pytest.approx(expected)

    def test_zero_flow(self):
        assert thermal_power_kw(0.0, 2400.0, 300.0, 200.0) == 0.0

    def test_equal_temps(self):
        assert thermal_power_kw(5.0, 2400.0, 210.0, 210.0) == 0.0

    def test_cold_exceeds_hot(self):
        assert thermal_power_kw(5.0, 2400.0, 200.0, 210.0) == 0.0

    def test_power_increases_with_flow(self):
        p1 = thermal_power_kw(5.0, 2400.0, 250.0, 210.0)
        p2 = thermal_power_kw(10.0, 2400.0, 250.0, 210.0)
        assert p2 == pytest.approx(2.0 * p1)

    def test_power_increases_with_delta_t(self):
        p1 = thermal_power_kw(5.0, 2400.0, 230.0, 210.0)
        p2 = thermal_power_kw(5.0, 2400.0, 250.0, 210.0)
        assert p2 > p1

    def test_negative_flow_clamped(self):
        assert thermal_power_kw(-5.0, 2400.0, 300.0, 200.0) == 0.0

    def test_result_in_kw(self):
        # 5 kg/s × 2400 J/(kg·K) × 40 K = 480,000 W = 480 kW
        result = thermal_power_kw(5.0, 2400.0, 250.0, 210.0)
        assert result == pytest.approx(480.0)


# =====================================================================
# 6. Pump parasitic power
# =====================================================================

class TestPumpPower:
    """Test pump_power_kw calculations."""

    def test_positive(self):
        p = pump_power_kw(5.0)
        assert p > 0.0

    def test_zero_flow(self):
        assert pump_power_kw(0.0) == 0.0

    def test_increases_with_flow(self):
        p1 = pump_power_kw(5.0)
        p2 = pump_power_kw(10.0)
        assert p2 == pytest.approx(2.0 * p1)

    def test_increases_with_pressure_drop(self):
        p1 = pump_power_kw(5.0, pressure_drop_pa=500_000.0)
        p2 = pump_power_kw(5.0, pressure_drop_pa=1_000_000.0)
        assert p2 > p1

    def test_negative_flow_clamped(self):
        assert pump_power_kw(-5.0) == 0.0

    def test_known_value(self):
        # P = (5/550) × 500000 / 0.75 / 1000 ≈ 6.06 kW
        result = pump_power_kw(5.0, 550.0, 500_000.0, 0.75)
        expected = (5.0 / 550.0) * 500_000.0 / 0.75 / 1000.0
        assert result == pytest.approx(expected)

    def test_lower_efficiency_more_power(self):
        p1 = pump_power_kw(5.0, pump_eff=0.75)
        p2 = pump_power_kw(5.0, pump_eff=0.50)
        assert p2 > p1


# =====================================================================
# 7. Net power
# =====================================================================

class TestNetPower:
    """Test net_power_kw calculations."""

    def test_basic_subtraction(self):
        result = net_power_kw(50.0, 6.0)
        assert result == pytest.approx(44.0)

    def test_pump_exceeds_generation(self):
        assert net_power_kw(5.0, 10.0) == 0.0

    def test_zero_generation(self):
        assert net_power_kw(0.0, 5.0) == 0.0

    def test_zero_pump(self):
        assert net_power_kw(50.0, 0.0) == pytest.approx(50.0)

    def test_never_negative(self):
        assert net_power_kw(1.0, 100.0) >= 0.0


# =====================================================================
# 8. Scaling / degradation
# =====================================================================

class TestScaling:
    """Test scaling degradation over time."""

    def test_one_sol_degradation(self):
        result = apply_scaling_degradation(1.0)
        assert result < 1.0
        assert result == pytest.approx(1.0 * (1.0 - SCALING_DEGRADATION_PER_SOL))

    def test_monotonically_decreasing(self):
        s = 1.0
        for _ in range(100):
            new_s = apply_scaling_degradation(s)
            assert new_s <= s
            s = new_s

    def test_stays_positive(self):
        s = 1.0
        for _ in range(100_000):
            s = apply_scaling_degradation(s)
        assert s > 0.0

    def test_clamped_above_one(self):
        result = apply_scaling_degradation(1.5)
        assert result <= 1.0

    def test_zero_stays_zero(self):
        assert apply_scaling_degradation(0.0) == 0.0

    def test_zero_degradation_rate(self):
        assert apply_scaling_degradation(0.8, degradation_per_sol=0.0) == 0.8

    def test_full_degradation_rate(self):
        assert apply_scaling_degradation(0.8, degradation_per_sol=1.0) == 0.0

    def test_custom_rate(self):
        result = apply_scaling_degradation(1.0, degradation_per_sol=0.01)
        assert result == pytest.approx(0.99)


# =====================================================================
# 9. Thermal drawdown
# =====================================================================

class TestThermalDrawdown:
    """Test thermal drawdown of rock temperature."""

    def test_one_sol(self):
        result = thermal_drawdown(255.0, 210.0)
        delta = 255.0 - 210.0
        expected = 210.0 + delta * (1.0 - THERMAL_DRAWDOWN_FRACTION_PER_SOL)
        assert result == pytest.approx(expected)

    def test_never_below_surface(self):
        result = thermal_drawdown(211.0, 210.0)
        assert result >= 210.0

    def test_surface_temp_unchanged(self):
        result = thermal_drawdown(210.0, 210.0)
        assert result == pytest.approx(210.0)

    def test_monotonically_decreasing(self):
        t = 300.0
        for _ in range(1000):
            new_t = thermal_drawdown(t, 210.0)
            assert new_t <= t
            t = new_t

    def test_approaches_surface(self):
        t = 300.0
        for _ in range(1_000_000):
            t = thermal_drawdown(t, 210.0)
        assert t == pytest.approx(210.0, abs=1.0)

    def test_zero_drawdown_no_change(self):
        result = thermal_drawdown(300.0, 210.0, drawdown_fraction=0.0)
        assert result == pytest.approx(300.0)

    def test_full_drawdown_to_surface(self):
        result = thermal_drawdown(300.0, 210.0, drawdown_fraction=1.0)
        assert result == pytest.approx(210.0)

    def test_higher_temp_larger_drop(self):
        drop1 = 250.0 - thermal_drawdown(250.0, 210.0)
        drop2 = 350.0 - thermal_drawdown(350.0, 210.0)
        assert drop2 > drop1


# =====================================================================
# 10. Thermal energy (kWh)
# =====================================================================

class TestThermalEnergy:
    """Test energy conversion from power × time."""

    def test_one_sol(self):
        result = thermal_energy_kwh(100.0)
        assert result == pytest.approx(100.0 * HOURS_PER_SOL)

    def test_zero_power(self):
        assert thermal_energy_kwh(0.0) == 0.0

    def test_negative_power_clamped(self):
        assert thermal_energy_kwh(-10.0) == 0.0

    def test_custom_hours(self):
        result = thermal_energy_kwh(100.0, hours=10.0)
        assert result == pytest.approx(1000.0)


# =====================================================================
# 11. State dataclass
# =====================================================================

class TestGeothermalWellState:
    """Test GeothermalWell dataclass construction and defaults."""

    def test_default_creation(self):
        well = GeothermalWell()
        assert well.sol == 0
        assert well.well_depth_m == DEFAULT_WELL_DEPTH_M
        assert well.flow_rate_kg_s == DEFAULT_FLOW_RATE_KG_S

    def test_rock_temp_computed(self):
        well = GeothermalWell()
        expected = rock_temperature_at_depth(DEFAULT_WELL_DEPTH_M)
        assert well.rock_temp_at_depth_k == pytest.approx(expected)

    def test_custom_depth(self):
        well = GeothermalWell(well_depth_m=5000.0)
        expected = rock_temperature_at_depth(5000.0)
        assert well.rock_temp_at_depth_k == pytest.approx(expected)

    def test_scaling_starts_at_one(self):
        well = GeothermalWell()
        assert well.scaling_factor == 1.0

    def test_cumulative_starts_zero(self):
        well = GeothermalWell()
        assert well.cumulative_heat_extracted_kwh == 0.0
        assert well.cumulative_electrical_kwh == 0.0

    def test_events_starts_empty(self):
        well = GeothermalWell()
        assert well.events == []

    def test_negative_depth_clamped(self):
        well = GeothermalWell(well_depth_m=-100.0)
        assert well.well_depth_m == 0.0
        assert well.rock_temp_at_depth_k >= well.surface_temp_k

    def test_efficiency_clamped(self):
        well = GeothermalWell(heat_exchange_efficiency=1.5)
        assert well.heat_exchange_efficiency == 1.0

    def test_rock_temp_never_below_surface(self):
        well = GeothermalWell(well_depth_m=0.0, gradient_k_per_km=0.0)
        assert well.rock_temp_at_depth_k >= well.surface_temp_k


# =====================================================================
# 12. Serialisation round-trip
# =====================================================================

class TestSerialisation:
    """Test to_dict / from_dict round-trip."""

    def test_round_trip_default(self):
        well = GeothermalWell()
        tick(well)
        data = well.to_dict()
        restored = GeothermalWell.from_dict(data)
        assert restored.sol == well.sol
        assert restored.rock_temp_at_depth_k == pytest.approx(well.rock_temp_at_depth_k)
        assert restored.scaling_factor == pytest.approx(well.scaling_factor)

    def test_round_trip_preserves_cumulative(self):
        well = GeothermalWell()
        for _ in range(10):
            tick(well)
        data = well.to_dict()
        restored = GeothermalWell.from_dict(data)
        assert restored.cumulative_heat_extracted_kwh == pytest.approx(
            well.cumulative_heat_extracted_kwh
        )
        assert restored.cumulative_electrical_kwh == pytest.approx(
            well.cumulative_electrical_kwh
        )

    def test_round_trip_events(self):
        well = GeothermalWell()
        tick(well)
        data = well.to_dict()
        restored = GeothermalWell.from_dict(data)
        assert restored.events == well.events

    def test_to_dict_keys(self):
        well = GeothermalWell()
        d = well.to_dict()
        required_keys = {
            "well_depth_m", "gradient_k_per_km", "surface_temp_k",
            "flow_rate_kg_s", "rock_temp_at_depth_k", "fluid_outlet_temp_k",
            "heat_exchange_efficiency", "orc_efficiency_fraction",
            "scaling_factor", "thermal_power_kw", "gross_electrical_power_kw",
            "pump_power_kw", "net_power_kw", "cumulative_heat_extracted_kwh",
            "cumulative_electrical_kwh", "sol", "events",
        }
        assert set(d.keys()) == required_keys

    def test_from_dict_empty(self):
        well = GeothermalWell.from_dict({})
        assert well.well_depth_m == DEFAULT_WELL_DEPTH_M


# =====================================================================
# 13. Tick engine
# =====================================================================

class TestTick:
    """Test per-sol tick function."""

    def test_sol_increments(self):
        well = GeothermalWell()
        tick(well)
        assert well.sol == 1

    def test_sol_increments_multiple(self):
        well = GeothermalWell()
        for _ in range(10):
            tick(well)
        assert well.sol == 10

    def test_returns_tick_result(self):
        well = GeothermalWell()
        result = tick(well)
        assert isinstance(result, TickResult)
        assert result.sol == 1

    def test_thermal_power_positive(self):
        well = GeothermalWell()
        result = tick(well)
        assert result.thermal_power_kw > 0.0

    def test_net_power_positive(self):
        well = GeothermalWell()
        result = tick(well)
        assert result.net_kw > 0.0

    def test_pump_power_positive(self):
        well = GeothermalWell()
        result = tick(well)
        assert result.pump_kw > 0.0

    def test_orc_eff_positive(self):
        well = GeothermalWell()
        result = tick(well)
        assert result.orc_eff > 0.0

    def test_scaling_decreases_after_tick(self):
        well = GeothermalWell()
        tick(well)
        assert well.scaling_factor < 1.0

    def test_rock_temp_decreases_after_tick(self):
        well = GeothermalWell()
        initial = well.rock_temp_at_depth_k
        tick(well)
        assert well.rock_temp_at_depth_k < initial

    def test_cumulative_increases(self):
        well = GeothermalWell()
        tick(well)
        assert well.cumulative_heat_extracted_kwh > 0.0
        assert well.cumulative_electrical_kwh > 0.0

    def test_result_matches_state(self):
        well = GeothermalWell()
        result = tick(well)
        assert result.rock_temp_k == pytest.approx(well.rock_temp_at_depth_k)
        assert result.scaling_factor == pytest.approx(well.scaling_factor)


# =====================================================================
# 14. Conservation of energy
# =====================================================================

class TestConservation:
    """Test thermodynamic conservation laws."""

    def test_electrical_leq_thermal(self):
        well = GeothermalWell()
        for _ in range(100):
            result = tick(well)
            assert result.gross_electrical_kw <= result.thermal_power_kw + 1e-9

    def test_net_leq_gross(self):
        well = GeothermalWell()
        for _ in range(100):
            result = tick(well)
            assert result.net_kw <= result.gross_electrical_kw + 1e-9

    def test_orc_leq_carnot(self):
        well = GeothermalWell()
        for _ in range(100):
            result = tick(well)
            assert result.orc_eff <= result.carnot_eff + 1e-9

    def test_rock_temp_geq_surface(self):
        well = GeothermalWell()
        for _ in range(1000):
            tick(well)
            assert well.rock_temp_at_depth_k >= well.surface_temp_k - 1e-9

    def test_all_powers_non_negative(self):
        well = GeothermalWell()
        for _ in range(500):
            result = tick(well)
            assert result.thermal_power_kw >= 0.0
            assert result.gross_electrical_kw >= 0.0
            assert result.pump_kw >= 0.0
            assert result.net_kw >= 0.0

    def test_scaling_in_unit_interval(self):
        well = GeothermalWell()
        for _ in range(500):
            tick(well)
            assert 0.0 <= well.scaling_factor <= 1.0

    def test_cumulative_monotonically_increasing(self):
        well = GeothermalWell()
        prev_heat = 0.0
        prev_elec = 0.0
        for _ in range(100):
            tick(well)
            assert well.cumulative_heat_extracted_kwh >= prev_heat
            assert well.cumulative_electrical_kwh >= prev_elec
            prev_heat = well.cumulative_heat_extracted_kwh
            prev_elec = well.cumulative_electrical_kwh

    def test_thermal_energy_matches_power(self):
        well = GeothermalWell()
        result = tick(well)
        expected_kwh = result.thermal_power_kw * HOURS_PER_SOL
        assert result.thermal_energy_kwh == pytest.approx(expected_kwh)


# =====================================================================
# 15. Multi-sol simulation
# =====================================================================

class TestRunSimulation:
    """Test run_simulation function."""

    def test_returns_correct_count(self):
        results = run_simulation(sols=100)
        assert len(results) == 100

    def test_sol_numbers_sequential(self):
        results = run_simulation(sols=50)
        for i, r in enumerate(results):
            assert r.sol == i + 1

    def test_power_declines_over_time(self):
        results = run_simulation(sols=365)
        assert results[-1].net_kw < results[0].net_kw

    def test_rock_temp_declines_over_time(self):
        results = run_simulation(sols=365)
        assert results[-1].rock_temp_k < results[0].rock_temp_k

    def test_scaling_declines_over_time(self):
        results = run_simulation(sols=365)
        assert results[-1].scaling_factor < results[0].scaling_factor

    def test_cumulative_energy_grows(self):
        results = run_simulation(sols=100)
        total_kwh = sum(r.net_energy_kwh for r in results)
        assert total_kwh > 0.0

    def test_deep_well_more_power(self):
        shallow = run_simulation(sols=10, well_depth_m=1000.0)
        deep = run_simulation(sols=10, well_depth_m=5000.0)
        assert deep[0].net_kw > shallow[0].net_kw

    def test_high_gradient_more_power(self):
        low = run_simulation(sols=10, gradient_k_per_km=5.0)
        high = run_simulation(sols=10, gradient_k_per_km=20.0)
        assert high[0].net_kw > low[0].net_kw

    def test_higher_flow_more_thermal(self):
        low_flow = run_simulation(sols=10, flow_rate_kg_s=2.0)
        high_flow = run_simulation(sols=10, flow_rate_kg_s=10.0)
        assert high_flow[0].thermal_power_kw > low_flow[0].thermal_power_kw

    def test_zero_sols_empty(self):
        results = run_simulation(sols=0)
        assert results == []

    def test_one_sol(self):
        results = run_simulation(sols=1)
        assert len(results) == 1
        assert results[0].sol == 1

    def test_deterministic(self):
        r1 = run_simulation(sols=50)
        r2 = run_simulation(sols=50)
        for a, b in zip(r1, r2):
            assert a.net_kw == pytest.approx(b.net_kw)
            assert a.rock_temp_k == pytest.approx(b.rock_temp_k)

    def test_long_run_stability(self):
        """Well should still produce some energy after a Mars year."""
        results = run_simulation(sols=668)
        assert results[-1].thermal_power_kw > 0.0


# =====================================================================
# 16. Property-based invariants (parametrize)
# =====================================================================

class TestParametrizedInvariants:
    """Parametrized tests over ranges of depths and gradients."""

    @pytest.mark.parametrize("depth_m", [100, 500, 1000, 2000, 3000, 5000, 10000])
    def test_rock_temp_increases_with_depth(self, depth_m: int):
        t = rock_temperature_at_depth(float(depth_m))
        assert t >= MARS_SURFACE_TEMP_K

    @pytest.mark.parametrize("depth_m", [100, 500, 1000, 2000, 3000, 5000, 10000])
    def test_tick_all_powers_non_negative_by_depth(self, depth_m: int):
        well = GeothermalWell(well_depth_m=float(depth_m))
        result = tick(well)
        assert result.thermal_power_kw >= 0.0
        assert result.net_kw >= 0.0

    @pytest.mark.parametrize("gradient", [1.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    def test_tick_gradient_range(self, gradient: float):
        well = GeothermalWell(gradient_k_per_km=gradient)
        result = tick(well)
        assert result.thermal_power_kw >= 0.0
        assert result.orc_eff <= result.carnot_eff + 1e-9

    @pytest.mark.parametrize("flow", [0.1, 1.0, 5.0, 10.0, 20.0, 50.0])
    def test_tick_flow_rate_range(self, flow: float):
        well = GeothermalWell(flow_rate_kg_s=flow)
        result = tick(well)
        assert result.thermal_power_kw >= 0.0

    @pytest.mark.parametrize("depth_m,gradient", [
        (1000, 5.0),
        (2000, 10.0),
        (3000, 15.0),
        (5000, 20.0),
        (10000, 5.0),
    ])
    def test_multi_sol_conservation(self, depth_m: int, gradient: float):
        well = GeothermalWell(
            well_depth_m=float(depth_m),
            gradient_k_per_km=gradient,
        )
        for _ in range(50):
            result = tick(well)
            assert result.gross_electrical_kw <= result.thermal_power_kw + 1e-9
            assert result.net_kw <= result.gross_electrical_kw + 1e-9
            assert well.rock_temp_at_depth_k >= well.surface_temp_k - 1e-9

    @pytest.mark.parametrize("eff", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_heat_exchange_range(self, eff: float):
        well = GeothermalWell(heat_exchange_efficiency=eff)
        result = tick(well)
        assert result.thermal_power_kw >= 0.0


# =====================================================================
# 17. Edge cases
# =====================================================================

class TestEdgeCases:
    """Boundary and degenerate input tests."""

    def test_zero_depth_well(self):
        well = GeothermalWell(well_depth_m=0.0)
        result = tick(well)
        assert result.thermal_power_kw == 0.0
        assert result.net_kw == 0.0

    def test_zero_gradient(self):
        well = GeothermalWell(gradient_k_per_km=0.0)
        result = tick(well)
        assert result.thermal_power_kw == 0.0

    def test_zero_flow_rate(self):
        well = GeothermalWell(flow_rate_kg_s=0.0)
        result = tick(well)
        assert result.thermal_power_kw == 0.0
        assert result.pump_kw == 0.0

    def test_very_deep_well(self):
        well = GeothermalWell(well_depth_m=50000.0)
        result = tick(well)
        assert result.thermal_power_kw > 0.0
        assert result.orc_eff > 0.0

    def test_extreme_gradient(self):
        well = GeothermalWell(gradient_k_per_km=100.0)
        result = tick(well)
        assert result.thermal_power_kw > 0.0

    def test_very_small_flow(self):
        well = GeothermalWell(flow_rate_kg_s=0.001)
        result = tick(well)
        assert result.thermal_power_kw >= 0.0
        assert result.net_kw >= 0.0

    def test_zero_heat_exchange(self):
        well = GeothermalWell(heat_exchange_efficiency=0.0)
        result = tick(well)
        assert result.thermal_power_kw == 0.0

    def test_exhausted_well_event(self):
        """A zero-depth well should report exhaustion."""
        well = GeothermalWell(well_depth_m=0.0)
        result = tick(well)
        assert any("EXHAUSTED" in e for e in result.events)

    def test_electrical_power_function_clamps_efficiency(self):
        assert electrical_power_kw(100.0, 1.5) == pytest.approx(100.0)
        assert electrical_power_kw(100.0, -0.5) == 0.0

    def test_carnot_negative_temp(self):
        assert carnot_efficiency(-100.0, 200.0) == 0.0

    def test_thermal_drawdown_rock_below_surface(self):
        """If rock temp somehow below surface, clamp."""
        result = thermal_drawdown(200.0, 210.0)
        assert result >= 210.0

    def test_fluid_outlet_when_rock_equals_surface(self):
        result = fluid_outlet_temperature(210.0, 210.0, 0.75, 1.0)
        assert result == pytest.approx(210.0)

    def test_tick_result_events_list(self):
        well = GeothermalWell()
        result = tick(well)
        assert isinstance(result.events, list)

    def test_negative_gradient_clamped(self):
        t = rock_temperature_at_depth(3000.0, gradient_k_per_km=-5.0)
        assert t == MARS_SURFACE_TEMP_K
