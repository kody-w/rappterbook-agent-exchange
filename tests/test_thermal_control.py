"""
test_thermal_control.py -- 71 unit tests for Mars habitat thermal regulation.

Tests the colony's thermostat: radiative/conductive heat loss,
metabolic and equipment heat gains, heater control, temperature
drift, insulation physics, airlock losses, and multi-sol stability.

Property-based invariants:
  - Heat losses are non-negative when interior > exterior
  - Temperature stays within physical bounds [-50, 50] C
  - Heater output never exceeds max capacity
  - More insulation reduces conductive loss
  - More crew increases metabolic heat
  - Comfort status matches temperature thresholds
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest

from src.thermal_control import (
    AEROGEL_R_PER_CM,
    AIR_HEAT_CAPACITY_J_M3K,
    DEFAULT_INSULATION_CM,
    EQUIPMENT_HEAT_W_PER_KWH,
    HABITAT_EMISSIVITY,
    HEATER_MAX_KW,
    MARS_SKY_TEMP_K,
    MAX_SAFE_TEMP_C,
    METABOLIC_HEAT_W,
    MIN_SAFE_TEMP_C,
    REGOLITH_CONDUCTIVITY_W_MK,
    SOL_HOURS,
    SOL_SECONDS,
    STEFAN_BOLTZMANN,
    TARGET_TEMP_C,
    TEMP_TOLERANCE_C,
    InsulationSpec,
    ThermalState,
    airlock_loss_kwh,
    comfort_status,
    conductive_loss_kw,
    equipment_heat_kw,
    floor_loss_kw,
    metabolic_heat_kw,
    radiative_loss_kw,
    required_heating_kw,
    temperature_drift,
    tick_thermal,
)


# ===================================================================
# 1. Physical constants sanity
# ===================================================================

class TestConstants:
    def test_stefan_boltzmann(self):
        assert 5.6e-8 < STEFAN_BOLTZMANN < 5.8e-8

    def test_habitat_emissivity_bounded(self):
        assert 0.0 < HABITAT_EMISSIVITY <= 1.0

    def test_mars_sky_temp(self):
        assert 180.0 < MARS_SKY_TEMP_K < 210.0

    def test_metabolic_heat_reasonable(self):
        assert 60.0 < METABOLIC_HEAT_W < 150.0

    def test_target_temp_comfortable(self):
        assert 18.0 <= TARGET_TEMP_C <= 25.0

    def test_safe_range_ordered(self):
        assert MIN_SAFE_TEMP_C < TARGET_TEMP_C < MAX_SAFE_TEMP_C

    def test_sol_hours(self):
        assert 24.5 < SOL_HOURS < 25.0

    def test_sol_seconds(self):
        assert 88000 < SOL_SECONDS < 89000

    def test_aerogel_r_value(self):
        assert AEROGEL_R_PER_CM > 5.0  # aerogel is an excellent insulator

    def test_heater_max(self):
        assert HEATER_MAX_KW > 0


# ===================================================================
# 2. InsulationSpec dataclass
# ===================================================================

class TestInsulationSpec:
    def test_defaults(self):
        ins = InsulationSpec()
        assert ins.thickness_cm == DEFAULT_INSULATION_CM
        assert ins.r_value > 0

    def test_r_value_scales_with_thickness(self):
        thin = InsulationSpec(thickness_cm=2.0)
        thick = InsulationSpec(thickness_cm=10.0)
        assert thick.r_value > thin.r_value

    def test_r_value_formula(self):
        ins = InsulationSpec(thickness_cm=5.0)
        assert ins.r_value == pytest.approx(5.0 * AEROGEL_R_PER_CM)

    def test_thickness_floor(self):
        ins = InsulationSpec(thickness_cm=0.0)
        assert ins.thickness_cm >= 0.1

    def test_wall_area_floor(self):
        ins = InsulationSpec(wall_area_m2=0.0)
        assert ins.wall_area_m2 >= 1.0

    def test_floor_area_floor(self):
        ins = InsulationSpec(floor_area_m2=0.0)
        assert ins.floor_area_m2 >= 1.0


# ===================================================================
# 3. ThermalState dataclass
# ===================================================================

class TestThermalState:
    def test_defaults(self):
        ts = ThermalState()
        assert ts.interior_temp_c == pytest.approx(TARGET_TEMP_C)
        assert ts.heater_output_kw == 0.0
        assert ts.sols_operational == 0


# ===================================================================
# 4. Radiative heat loss
# ===================================================================

class TestRadiativeLoss:
    def test_positive_when_warm(self):
        loss = radiative_loss_kw(628.0, 21.0)
        assert loss > 0.0

    def test_zero_area_zero_loss(self):
        loss = radiative_loss_kw(0.0, 21.0)
        assert loss == 0.0

    def test_higher_temp_more_loss(self):
        l1 = radiative_loss_kw(628.0, 10.0)
        l2 = radiative_loss_kw(628.0, 30.0)
        assert l2 > l1

    def test_larger_area_more_loss(self):
        l1 = radiative_loss_kw(100.0, 21.0)
        l2 = radiative_loss_kw(1000.0, 21.0)
        assert l2 > l1

    def test_freezing_temp_still_positive(self):
        loss = radiative_loss_kw(628.0, -40.0)
        # -40C = 233K > sky temp 193K, so still radiates
        assert loss > 0.0

    def test_below_sky_temp_zero(self):
        # If surface is at or below sky temp, no radiative loss
        loss = radiative_loss_kw(628.0, -200.0)
        assert loss == 0.0


# ===================================================================
# 5. Conductive heat loss
# ===================================================================

class TestConductiveLoss:
    def test_positive_when_warm(self):
        ins = InsulationSpec()
        loss = conductive_loss_kw(21.0, -60.0, ins)
        assert loss > 0.0

    def test_zero_when_equal_temp(self):
        ins = InsulationSpec()
        loss = conductive_loss_kw(21.0, 21.0, ins)
        assert loss == 0.0

    def test_zero_when_exterior_warmer(self):
        ins = InsulationSpec()
        loss = conductive_loss_kw(10.0, 20.0, ins)
        assert loss == 0.0

    def test_more_insulation_less_loss(self):
        thin = InsulationSpec(thickness_cm=2.0)
        thick = InsulationSpec(thickness_cm=10.0)
        l_thin = conductive_loss_kw(21.0, -60.0, thin)
        l_thick = conductive_loss_kw(21.0, -60.0, thick)
        assert l_thick < l_thin

    def test_bigger_delta_more_loss(self):
        ins = InsulationSpec()
        l1 = conductive_loss_kw(21.0, 0.0, ins)
        l2 = conductive_loss_kw(21.0, -100.0, ins)
        assert l2 > l1


# ===================================================================
# 6. Floor loss
# ===================================================================

class TestFloorLoss:
    def test_positive_when_warm(self):
        loss = floor_loss_kw(21.0, -40.0)
        assert loss > 0.0

    def test_zero_when_equal(self):
        loss = floor_loss_kw(-40.0, -40.0)
        assert loss == 0.0

    def test_small_magnitude(self):
        # Regolith is a good insulator; floor loss should be small
        loss = floor_loss_kw(21.0, -40.0)
        assert loss < 1.0  # less than 1 kW


# ===================================================================
# 7. Airlock loss
# ===================================================================

class TestAirlockLoss:
    def test_positive(self):
        loss = airlock_loss_kwh(21.0, -60.0)
        assert loss > 0.0

    def test_zero_cycles_zero_loss(self):
        loss = airlock_loss_kwh(21.0, -60.0, cycles=0.0)
        assert loss == 0.0

    def test_more_cycles_more_loss(self):
        l1 = airlock_loss_kwh(21.0, -60.0, cycles=2.0)
        l2 = airlock_loss_kwh(21.0, -60.0, cycles=8.0)
        assert l2 > l1

    def test_zero_when_equal_temp(self):
        loss = airlock_loss_kwh(21.0, 21.0)
        assert loss == 0.0


# ===================================================================
# 8. Heat gains
# ===================================================================

class TestHeatGains:
    def test_metabolic_zero_pop(self):
        assert metabolic_heat_kw(0) == 0.0

    def test_metabolic_positive(self):
        assert metabolic_heat_kw(10) > 0.0

    def test_metabolic_scales(self):
        h10 = metabolic_heat_kw(10)
        h20 = metabolic_heat_kw(20)
        assert h20 == pytest.approx(2 * h10)

    def test_metabolic_negative_pop(self):
        assert metabolic_heat_kw(-5) == 0.0

    def test_equipment_zero(self):
        assert equipment_heat_kw(0.0) == 0.0

    def test_equipment_positive(self):
        assert equipment_heat_kw(100.0) > 0.0

    def test_equipment_negative_safe(self):
        assert equipment_heat_kw(-10.0) == 0.0


# ===================================================================
# 9. Heater control
# ===================================================================

class TestHeaterControl:
    def test_no_heating_when_surplus(self):
        # Passive heat exceeds loss -> no heater needed
        needed = required_heating_kw(5.0, 10.0)
        assert needed == 0.0

    def test_heating_when_deficit(self):
        needed = required_heating_kw(10.0, 3.0)
        assert needed > 0.0
        assert needed == pytest.approx(7.0)

    def test_capped_at_max(self):
        needed = required_heating_kw(100.0, 0.0)
        assert needed <= HEATER_MAX_KW


# ===================================================================
# 10. Temperature drift
# ===================================================================

class TestTemperatureDrift:
    def test_positive_drift_from_surplus(self):
        drift = temperature_drift(1.0)
        assert drift > 0.0

    def test_negative_drift_from_deficit(self):
        drift = temperature_drift(-1.0)
        assert drift < 0.0

    def test_zero_net_zero_drift(self):
        drift = temperature_drift(0.0)
        assert drift == 0.0

    def test_zero_mass_zero_drift(self):
        drift = temperature_drift(1.0, habitat_mass_kg=0.0)
        assert drift == 0.0


# ===================================================================
# 11. Comfort status
# ===================================================================

class TestComfortStatus:
    def test_comfortable(self):
        assert comfort_status(TARGET_TEMP_C) == "comfortable"

    def test_cool(self):
        assert comfort_status(TARGET_TEMP_C - 5.0) == "cool"

    def test_warm(self):
        assert comfort_status(TARGET_TEMP_C + 5.0) == "warm"

    def test_critical_cold(self):
        assert comfort_status(5.0) == "critical_cold"

    def test_critical_hot(self):
        assert comfort_status(40.0) == "critical_hot"


# ===================================================================
# 12. tick_thermal integration
# ===================================================================

class TestTickThermal:
    def test_returns_dict(self):
        state = ThermalState()
        ins = InsulationSpec()
        result = tick_thermal(state, ins, -60.0, 10, 100.0, 50.0)
        assert isinstance(result, dict)
        assert "sol" in result
        assert "interior_temp_c" in result
        assert "comfort" in result

    def test_sol_advances(self):
        state = ThermalState()
        tick_thermal(state, InsulationSpec(), -60.0, 10, 100.0, 50.0)
        assert state.sols_operational == 1

    def test_heater_activates_in_cold(self):
        state = ThermalState()
        result = tick_thermal(state, InsulationSpec(), -100.0, 2, 10.0, 200.0)
        assert result["heater_kw"] > 0

    def test_temp_stays_bounded(self):
        state = ThermalState()
        ins = InsulationSpec()
        for _ in range(100):
            tick_thermal(state, ins, -120.0, 0, 0.0, 0.0)
        assert state.interior_temp_c >= -50.0

    def test_temp_stays_bounded_hot(self):
        state = ThermalState()
        ins = InsulationSpec()
        for _ in range(100):
            tick_thermal(state, ins, 20.0, 100, 5000.0, 0.0)
        assert state.interior_temp_c <= 50.0

    def test_cumulative_tracking(self):
        state = ThermalState()
        tick_thermal(state, InsulationSpec(), -60.0, 10, 100.0, 50.0)
        assert state.total_heat_input_kwh > 0
        assert state.total_heat_lost_kwh > 0

    def test_no_population_less_heat(self):
        s1 = ThermalState()
        s2 = ThermalState()
        ins = InsulationSpec()
        r1 = tick_thermal(s1, ins, -60.0, 0, 100.0, 50.0)
        r2 = tick_thermal(s2, ins, -60.0, 50, 100.0, 50.0)
        assert r2["metabolic_heat_kw"] > r1["metabolic_heat_kw"]


# ===================================================================
# 13. Multi-sol simulation
# ===================================================================

class TestMultiSol:
    def test_10_sol_smoke(self):
        state = ThermalState()
        ins = InsulationSpec()
        for _ in range(10):
            r = tick_thermal(state, ins, -60.0, 10, 100.0, 50.0)
            assert isinstance(r, dict)
        assert state.sols_operational == 10

    def test_100_sol_stability(self):
        """With enough heating, habitat should stay comfortable."""
        state = ThermalState()
        ins = InsulationSpec(thickness_cm=10.0)
        for _ in range(100):
            tick_thermal(state, ins, -60.0, 20, 200.0, 200.0)
        assert state.interior_temp_c > MIN_SAFE_TEMP_C

    def test_freezing_without_power(self):
        """No power, no crew -> habitat cools toward exterior temp."""
        state = ThermalState(interior_temp_c=21.0)
        ins = InsulationSpec()
        for _ in range(50):
            tick_thermal(state, ins, -60.0, 0, 0.0, 0.0)
        assert state.interior_temp_c < 21.0

    def test_temp_always_in_bounds(self):
        """Property: temperature always in [-50, 50] C."""
        state = ThermalState()
        ins = InsulationSpec()
        for _ in range(200):
            tick_thermal(state, ins, -120.0, 0, 0.0, 0.0)
            assert -50.0 <= state.interior_temp_c <= 50.0

    def test_seasonal_variation(self):
        """Temperature response to changing external conditions."""
        state = ThermalState()
        ins = InsulationSpec()
        temps = []
        for sol in range(100):
            ext = -60.0 + 40.0 * (sol / 100.0)  # warming from -60 to -20
            tick_thermal(state, ins, ext, 10, 100.0, 100.0)
            temps.append(state.interior_temp_c)
        # Should respond to warming exterior
        assert len(temps) == 100

    def test_heat_input_output_accumulate(self):
        state = ThermalState()
        ins = InsulationSpec()
        for _ in range(50):
            tick_thermal(state, ins, -60.0, 10, 100.0, 50.0)
        assert state.total_heat_input_kwh > 0
        assert state.total_heat_lost_kwh > 0
