"""Tests for src/thermal_control.py — Mars habitat thermal regulation."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.thermal_control import (
    HabitatThermal,
    conduction_loss_kwh,
    radiation_loss_kwh,
    metabolic_heat_kwh,
    rtg_heat_kwh,
    equipment_heat_kwh,
    airlock_loss_kwh,
    heater_demand_kwh,
    comfort_score,
    tick_thermal,
    STEFAN_BOLTZMANN,
    MARS_SOL_HOURS,
    METABOLIC_HEAT_W,
    RTG_THERMAL_W,
    RTG_ELECTRICAL_W,
    RTG_DECAY_PER_SOL,
    DEFAULT_WALL_U,
    REGOLITH_BERM_U,
    MINIMAL_INSULATION_U,
    EQUIPMENT_HEAT_FRACTION,
    AIRLOCK_HEAT_LOSS_KWH,
    RADIATOR_EMISSIVITY,
    RADIATOR_AREA_M2_PER_UNIT,
    THERMAL_MASS_KWH_PER_C,
    COMFORT_MIN_C,
    COMFORT_MAX_C,
    TARGET_TEMP_C,
    HYPOTHERMIA_C,
    HYPERTHERMIA_C,
)


# ===================================================================
# HabitatThermal dataclass
# ===================================================================

class TestHabitatThermal:
    def test_default_values(self) -> None:
        h = HabitatThermal()
        assert h.interior_temp_c == TARGET_TEMP_C
        assert h.wall_area_m2 == 400.0
        assert h.wall_u_value == DEFAULT_WALL_U
        assert h.rtg_count == 2
        assert h.rtg_efficiency == 1.0
        assert h.radiator_units == 1

    def test_custom_values(self) -> None:
        h = HabitatThermal(interior_temp_c=25.0, wall_area_m2=800.0, rtg_count=4)
        assert h.interior_temp_c == 25.0
        assert h.wall_area_m2 == 800.0
        assert h.rtg_count == 4

    def test_clamping_temp(self) -> None:
        h = HabitatThermal(interior_temp_c=-100.0)
        assert h.interior_temp_c == -50.0
        h2 = HabitatThermal(interior_temp_c=100.0)
        assert h2.interior_temp_c == 60.0

    def test_clamping_wall_area(self) -> None:
        h = HabitatThermal(wall_area_m2=-50.0)
        assert h.wall_area_m2 == 1.0

    def test_clamping_u_value(self) -> None:
        h = HabitatThermal(wall_u_value=-1.0)
        assert h.wall_u_value == 0.01
        h2 = HabitatThermal(wall_u_value=5.0)
        assert h2.wall_u_value == 2.0

    def test_clamping_rtg_count(self) -> None:
        h = HabitatThermal(rtg_count=-3)
        assert h.rtg_count == 0

    def test_clamping_rtg_efficiency(self) -> None:
        h = HabitatThermal(rtg_efficiency=1.5)
        assert h.rtg_efficiency == 1.0
        h2 = HabitatThermal(rtg_efficiency=-0.5)
        assert h2.rtg_efficiency == 0.0

    def test_clamping_thermal_mass_factor(self) -> None:
        h = HabitatThermal(thermal_mass_factor=0.0)
        assert h.thermal_mass_factor == 0.1


# ===================================================================
# Conduction loss
# ===================================================================

class TestConductionLoss:
    def test_positive_when_warmer_inside(self) -> None:
        loss = conduction_loss_kwh(400.0, 0.25, 22.0, -60.0)
        assert loss > 0

    def test_negative_when_warmer_outside(self) -> None:
        loss = conduction_loss_kwh(400.0, 0.25, 10.0, 20.0)
        assert loss < 0

    def test_zero_when_same_temp(self) -> None:
        loss = conduction_loss_kwh(400.0, 0.25, 20.0, 20.0)
        assert loss == 0.0

    def test_scales_with_area(self) -> None:
        loss1 = conduction_loss_kwh(200.0, 0.25, 22.0, -60.0)
        loss2 = conduction_loss_kwh(400.0, 0.25, 22.0, -60.0)
        assert loss2 == pytest.approx(2.0 * loss1, rel=1e-6)

    def test_scales_with_u_value(self) -> None:
        loss1 = conduction_loss_kwh(400.0, 0.25, 22.0, -60.0)
        loss2 = conduction_loss_kwh(400.0, 0.50, 22.0, -60.0)
        assert loss2 == pytest.approx(2.0 * loss1, rel=1e-6)

    def test_scales_with_delta_t(self) -> None:
        loss1 = conduction_loss_kwh(400.0, 0.25, 22.0, -18.0)  # ΔT = 40
        loss2 = conduction_loss_kwh(400.0, 0.25, 22.0, -58.0)  # ΔT = 80
        assert loss2 == pytest.approx(2.0 * loss1, rel=1e-6)

    def test_formula_matches_manual(self) -> None:
        """Q = U × A × ΔT × hours / 1000"""
        expected = 0.25 * 400.0 * 82.0 * MARS_SOL_HOURS / 1000.0
        actual = conduction_loss_kwh(400.0, 0.25, 22.0, -60.0)
        assert actual == pytest.approx(expected, rel=1e-4)


# ===================================================================
# Radiation loss (radiators)
# ===================================================================

class TestRadiationLoss:
    def test_positive_with_units(self) -> None:
        loss = radiation_loss_kwh(1)
        assert loss > 0

    def test_zero_with_no_units(self) -> None:
        assert radiation_loss_kwh(0) == 0.0

    def test_scales_with_units(self) -> None:
        loss1 = radiation_loss_kwh(1)
        loss2 = radiation_loss_kwh(3)
        assert loss2 == pytest.approx(3.0 * loss1, rel=1e-6)

    def test_stefan_boltzmann_formula(self) -> None:
        """P = ε σ A T⁴ × hours / 1000"""
        area = RADIATOR_AREA_M2_PER_UNIT
        watts = RADIATOR_EMISSIVITY * STEFAN_BOLTZMANN * area * (300.0 ** 4)
        expected = watts * MARS_SOL_HOURS / 1000.0
        actual = radiation_loss_kwh(1, 300.0)
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_higher_temp_more_radiation(self) -> None:
        """T⁴ dependence: hotter radiator dumps more heat."""
        loss_cool = radiation_loss_kwh(1, 280.0)
        loss_hot = radiation_loss_kwh(1, 320.0)
        assert loss_hot > loss_cool


# ===================================================================
# Metabolic heat
# ===================================================================

class TestMetabolicHeat:
    def test_positive(self) -> None:
        h = metabolic_heat_kwh(10)
        assert h > 0

    def test_zero_population(self) -> None:
        assert metabolic_heat_kwh(0) == 0.0

    def test_negative_population(self) -> None:
        assert metabolic_heat_kwh(-5) == 0.0

    def test_linear_scaling(self) -> None:
        h1 = metabolic_heat_kwh(1)
        h10 = metabolic_heat_kwh(10)
        assert h10 == pytest.approx(10.0 * h1, rel=1e-6)

    def test_formula_manual(self) -> None:
        expected = 10 * METABOLIC_HEAT_W * MARS_SOL_HOURS / 1000.0
        assert metabolic_heat_kwh(10) == pytest.approx(expected, rel=1e-4)


# ===================================================================
# RTG heat
# ===================================================================

class TestRtgHeat:
    def test_positive(self) -> None:
        h = rtg_heat_kwh(2, 1.0)
        assert h > 0

    def test_zero_count(self) -> None:
        assert rtg_heat_kwh(0, 1.0) == 0.0

    def test_zero_efficiency(self) -> None:
        assert rtg_heat_kwh(2, 0.0) == 0.0

    def test_waste_heat_is_thermal_minus_electrical(self) -> None:
        """Waste heat = thermal - electrical output."""
        waste_per_unit = RTG_THERMAL_W - RTG_ELECTRICAL_W
        expected = 2 * waste_per_unit * MARS_SOL_HOURS / 1000.0
        actual = rtg_heat_kwh(2, 1.0)
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_scales_with_count(self) -> None:
        h1 = rtg_heat_kwh(1, 1.0)
        h3 = rtg_heat_kwh(3, 1.0)
        assert h3 == pytest.approx(3.0 * h1, rel=1e-6)

    def test_degraded_rtg(self) -> None:
        full = rtg_heat_kwh(2, 1.0)
        half = rtg_heat_kwh(2, 0.5)
        assert half == pytest.approx(0.5 * full, rel=1e-6)

    def test_negative_count(self) -> None:
        assert rtg_heat_kwh(-1, 1.0) == 0.0


# ===================================================================
# Equipment heat
# ===================================================================

class TestEquipmentHeat:
    def test_positive(self) -> None:
        h = equipment_heat_kwh(100.0)
        assert h > 0

    def test_zero_power(self) -> None:
        assert equipment_heat_kwh(0.0) == 0.0

    def test_negative_power(self) -> None:
        assert equipment_heat_kwh(-50.0) == 0.0

    def test_fraction(self) -> None:
        assert equipment_heat_kwh(100.0) == pytest.approx(100.0 * EQUIPMENT_HEAT_FRACTION, rel=1e-6)


# ===================================================================
# Airlock loss
# ===================================================================

class TestAirlockLoss:
    def test_positive(self) -> None:
        assert airlock_loss_kwh(4.0) > 0

    def test_zero_cycles(self) -> None:
        assert airlock_loss_kwh(0.0) == 0.0

    def test_negative_cycles(self) -> None:
        assert airlock_loss_kwh(-1.0) == 0.0

    def test_scales_linearly(self) -> None:
        l1 = airlock_loss_kwh(1.0)
        l4 = airlock_loss_kwh(4.0)
        assert l4 == pytest.approx(4.0 * l1, rel=1e-6)

    def test_formula(self) -> None:
        assert airlock_loss_kwh(1.0) == pytest.approx(AIRLOCK_HEAT_LOSS_KWH, rel=1e-6)


# ===================================================================
# Heater demand
# ===================================================================

class TestHeaterDemand:
    def test_positive_deficit(self) -> None:
        d = heater_demand_kwh(10.0)
        assert d > 0

    def test_zero_deficit(self) -> None:
        assert heater_demand_kwh(0.0) == 0.0

    def test_negative_deficit(self) -> None:
        assert heater_demand_kwh(-5.0) == 0.0

    def test_efficiency_factor(self) -> None:
        """95% efficient heater needs slightly more power than heat output."""
        d = heater_demand_kwh(10.0, 0.95)
        assert d == pytest.approx(10.0 / 0.95, rel=1e-4)

    def test_lower_efficiency_more_power(self) -> None:
        d_high = heater_demand_kwh(10.0, 0.95)
        d_low = heater_demand_kwh(10.0, 0.50)
        assert d_low > d_high


# ===================================================================
# Comfort score
# ===================================================================

class TestComfortScore:
    def test_perfect_in_range(self) -> None:
        assert comfort_score(20.0) == 1.0
        assert comfort_score(22.0) == 1.0
        assert comfort_score(COMFORT_MIN_C) == 1.0
        assert comfort_score(COMFORT_MAX_C) == 1.0

    def test_zero_at_hypothermia(self) -> None:
        assert comfort_score(HYPOTHERMIA_C) == 0.0

    def test_zero_at_hyperthermia(self) -> None:
        assert comfort_score(HYPERTHERMIA_C) == 0.0

    def test_zero_below_hypothermia(self) -> None:
        assert comfort_score(-20.0) == 0.0

    def test_zero_above_hyperthermia(self) -> None:
        assert comfort_score(50.0) == 0.0

    def test_mid_cold(self) -> None:
        """Midpoint between hypothermia and comfort should be ~0.5."""
        mid = (HYPOTHERMIA_C + COMFORT_MIN_C) / 2.0
        score = comfort_score(mid)
        assert 0.4 < score < 0.6

    def test_mid_hot(self) -> None:
        """Midpoint between comfort and hyperthermia should be ~0.5."""
        mid = (COMFORT_MAX_C + HYPERTHERMIA_C) / 2.0
        score = comfort_score(mid)
        assert 0.4 < score < 0.6

    def test_monotonic_cold(self) -> None:
        """Comfort should increase monotonically from hypothermia to comfort min."""
        temps = [HYPOTHERMIA_C + i for i in range(int(COMFORT_MIN_C - HYPOTHERMIA_C) + 1)]
        scores = [comfort_score(t) for t in temps]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]

    def test_monotonic_hot(self) -> None:
        """Comfort should decrease monotonically from comfort max to hyperthermia."""
        temps = [COMFORT_MAX_C + i for i in range(int(HYPERTHERMIA_C - COMFORT_MAX_C) + 1)]
        scores = [comfort_score(t) for t in temps]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]

    def test_range_always_0_to_1(self) -> None:
        """Score must always be in [0, 1]."""
        for temp in range(-60, 70):
            s = comfort_score(float(temp))
            assert 0.0 <= s <= 1.0


# ===================================================================
# tick_thermal — integration tests
# ===================================================================

class TestTickThermal:
    def _nominal(self) -> HabitatThermal:
        return HabitatThermal()

    def test_nominal_returns_dict(self) -> None:
        h = self._nominal()
        result = tick_thermal(h, -60.0, 20, 200.0)
        assert isinstance(result, dict)
        assert "interior_temp_c" in result
        assert "comfort_score" in result

    def test_nominal_temp_stays_near_target(self) -> None:
        """With RTGs + heater power, habitat should stay near 22°C."""
        h = self._nominal()
        for _ in range(10):
            result = tick_thermal(h, -60.0, 20, 200.0)
        assert abs(h.interior_temp_c - TARGET_TEMP_C) < 5.0

    def test_no_heater_cools_down(self) -> None:
        """Without heating power on Mars, habitat cools toward exterior."""
        h = HabitatThermal(rtg_count=0)
        for _ in range(50):
            result = tick_thermal(h, -60.0, 0, 0.0, heater_power_available_kwh=0.0)
        assert h.interior_temp_c < TARGET_TEMP_C

    def test_rtg_provides_passive_heating(self) -> None:
        """RTGs should warm habitat vs no-RTG baseline."""
        h_rtg = HabitatThermal(rtg_count=4, interior_temp_c=0.0)
        h_none = HabitatThermal(rtg_count=0, interior_temp_c=0.0)
        for _ in range(20):
            tick_thermal(h_rtg, -60.0, 0, 0.0, heater_power_available_kwh=0.0)
            tick_thermal(h_none, -60.0, 0, 0.0, heater_power_available_kwh=0.0)
        assert h_rtg.interior_temp_c > h_none.interior_temp_c

    def test_crew_metabolic_heat(self) -> None:
        """More crew = more metabolic heat = warmer habitat."""
        h1 = HabitatThermal(rtg_count=0, interior_temp_c=20.0)
        h2 = HabitatThermal(rtg_count=0, interior_temp_c=20.0)
        tick_thermal(h1, -60.0, 10, 0.0, heater_power_available_kwh=0.0)
        tick_thermal(h2, -60.0, 100, 0.0, heater_power_available_kwh=0.0)
        assert h2.interior_temp_c > h1.interior_temp_c

    def test_rtg_decays(self) -> None:
        """RTG efficiency should decrease each sol."""
        h = self._nominal()
        initial = h.rtg_efficiency
        tick_thermal(h, -60.0, 10, 100.0)
        assert h.rtg_efficiency < initial
        assert h.rtg_efficiency == pytest.approx(initial - RTG_DECAY_PER_SOL, rel=1e-6)

    def test_airlock_cycles_increase_heat_loss(self) -> None:
        """More EVA cycles = more airlock heat loss."""
        h1 = HabitatThermal(interior_temp_c=22.0)
        h2 = HabitatThermal(interior_temp_c=22.0)
        r1 = tick_thermal(h1, -60.0, 10, 100.0, airlock_cycles=2.0)
        r2 = tick_thermal(h2, -60.0, 10, 100.0, airlock_cycles=20.0)
        assert r2["loss_airlock_kwh"] > r1["loss_airlock_kwh"]

    def test_better_insulation_less_loss(self) -> None:
        """Lower U-value = better insulation = warmer habitat."""
        h1 = HabitatThermal(wall_u_value=MINIMAL_INSULATION_U, interior_temp_c=22.0)
        h2 = HabitatThermal(wall_u_value=REGOLITH_BERM_U, interior_temp_c=22.0)
        tick_thermal(h1, -60.0, 10, 100.0, heater_power_available_kwh=0.0)
        tick_thermal(h2, -60.0, 10, 100.0, heater_power_available_kwh=0.0)
        assert h2.interior_temp_c > h1.interior_temp_c

    def test_comfort_nominal(self) -> None:
        """Nominal conditions should yield high comfort."""
        h = self._nominal()
        result = tick_thermal(h, -60.0, 20, 200.0)
        assert result["comfort_score"] > 0.8

    def test_zero_population_still_works(self) -> None:
        """Empty habitat should not crash."""
        h = self._nominal()
        result = tick_thermal(h, -60.0, 0, 0.0)
        assert result["heat_metabolic_kwh"] == 0.0

    def test_negative_population(self) -> None:
        h = self._nominal()
        result = tick_thermal(h, -60.0, -5, 100.0)
        assert result["heat_metabolic_kwh"] == 0.0

    def test_heater_activates_on_cold(self) -> None:
        """Heater should draw power when habitat is cold."""
        h = HabitatThermal(rtg_count=0, interior_temp_c=10.0)
        result = tick_thermal(h, -80.0, 5, 50.0, heater_power_available_kwh=200.0)
        assert result["heater_kwh"] > 0

    def test_radiator_activates_on_hot(self) -> None:
        """Radiator should activate when habitat is too hot."""
        h = HabitatThermal(
            rtg_count=10, interior_temp_c=30.0,
            radiator_units=3, wall_u_value=0.01,
        )
        result = tick_thermal(h, 10.0, 200, 1000.0)
        assert result["radiator_rejection_kwh"] > 0


# ===================================================================
# Physical invariants
# ===================================================================

class TestPhysicalInvariants:
    def test_temp_clamped(self) -> None:
        """Interior temp must stay in [-50, 60]°C."""
        h = HabitatThermal(interior_temp_c=-45.0, rtg_count=0)
        for _ in range(100):
            tick_thermal(h, -120.0, 0, 0.0, heater_power_available_kwh=0.0)
        assert h.interior_temp_c >= -50.0

    def test_rtg_efficiency_never_negative(self) -> None:
        h = HabitatThermal(rtg_efficiency=0.001)
        for _ in range(1000):
            tick_thermal(h, -60.0, 10, 100.0)
        assert h.rtg_efficiency >= 0.0

    def test_comfort_always_bounded(self) -> None:
        h = HabitatThermal()
        for ext in [-120, -60, -20, 0, 10, 20]:
            result = tick_thermal(h, float(ext), 10, 100.0)
            assert 0.0 <= result["comfort_score"] <= 1.0

    def test_constants_physically_reasonable(self) -> None:
        assert 50 < METABOLIC_HEAT_W < 500
        assert 1000 < RTG_THERMAL_W < 5000
        assert RTG_ELECTRICAL_W < RTG_THERMAL_W
        assert 0.0 < RTG_DECAY_PER_SOL < 0.001
        assert 0.0 < DEFAULT_WALL_U < 1.0
        assert REGOLITH_BERM_U < DEFAULT_WALL_U < MINIMAL_INSULATION_U
        assert 15.0 < COMFORT_MIN_C < COMFORT_MAX_C < 30.0
        assert HYPOTHERMIA_C < COMFORT_MIN_C
        assert HYPERTHERMIA_C > COMFORT_MAX_C

    def test_conduction_loss_is_major_on_mars(self) -> None:
        """On Mars (-60°C exterior), conduction should dominate heat loss."""
        loss = conduction_loss_kwh(400.0, DEFAULT_WALL_U, 22.0, -60.0)
        # Should be significant — tens to hundreds of kWh/sol
        assert loss > 10.0

    def test_rtg_waste_heat_always_positive(self) -> None:
        """RTG waste heat must be non-negative (thermal > electrical)."""
        for eff in [0.0, 0.5, 1.0]:
            for count in [0, 1, 5]:
                h = rtg_heat_kwh(count, eff)
                assert h >= 0.0


# ===================================================================
# Smoke tests
# ===================================================================

class TestSmoke:
    def test_10_sol_nominal(self) -> None:
        """10 sols with standard crew, no crash."""
        h = HabitatThermal()
        for _ in range(10):
            result = tick_thermal(h, -60.0, 20, 200.0)
            assert -50.0 <= h.interior_temp_c <= 60.0
            assert result["comfort_score"] >= 0.0

    def test_100_sol_stability(self) -> None:
        """100 sols: temp should stabilize, not diverge."""
        h = HabitatThermal()
        temps: list[float] = []
        for _ in range(100):
            tick_thermal(h, -60.0, 20, 200.0)
            temps.append(h.interior_temp_c)
        # Last 50 temps should not vary more than 2°C (stable)
        last_50 = temps[50:]
        assert max(last_50) - min(last_50) < 2.0

    def test_365_sol_mars_year(self) -> None:
        """Full Mars year: seasonal temp variation, RTG decay, no crash."""
        h = HabitatThermal()
        for sol in range(365):
            # Simulate seasonal variation
            ext_temp = -60.0 + 30.0 * math.sin(2 * math.pi * sol / 668.6)
            result = tick_thermal(h, ext_temp, 20, 200.0)
        assert h.interior_temp_c > HYPOTHERMIA_C
        assert h.rtg_efficiency < 1.0  # should have decayed

    def test_power_outage_recovery(self) -> None:
        """Lose heater power, then restore — should recover."""
        h = HabitatThermal()
        # 10 sols nominal
        for _ in range(10):
            tick_thermal(h, -60.0, 20, 200.0)
        # 5 sols no power
        for _ in range(5):
            tick_thermal(h, -60.0, 20, 0.0, heater_power_available_kwh=0.0)
        cold_temp = h.interior_temp_c
        # 10 sols restored
        for _ in range(10):
            tick_thermal(h, -60.0, 20, 200.0)
        assert h.interior_temp_c > cold_temp

    def test_empty_habitat_still_stable(self) -> None:
        """Unoccupied habitat: RTGs keep it above freezing."""
        h = HabitatThermal(rtg_count=2)
        for _ in range(100):
            tick_thermal(h, -60.0, 0, 0.0, heater_power_available_kwh=0.0)
        # RTGs should keep it above hypothermia
        assert h.interior_temp_c > HYPOTHERMIA_C


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_extreme_cold_exterior(self) -> None:
        h = HabitatThermal()
        result = tick_thermal(h, -120.0, 20, 200.0)
        assert result["loss_conduction_kwh"] > 0

    def test_warm_exterior(self) -> None:
        """Mars summer day near equator: ~0°C."""
        h = HabitatThermal()
        result = tick_thermal(h, 0.0, 20, 200.0)
        assert result["loss_conduction_kwh"] < conduction_loss_kwh(400.0, 0.25, 22.0, -60.0)

    def test_huge_population(self) -> None:
        """500 people generate serious metabolic heat."""
        h = HabitatThermal()
        result = tick_thermal(h, -60.0, 500, 500.0)
        assert result["heat_metabolic_kwh"] > 100.0

    def test_no_rtg_no_power(self) -> None:
        """Worst case: no RTGs, no power, Mars winter."""
        h = HabitatThermal(rtg_count=0)
        for _ in range(20):
            tick_thermal(h, -80.0, 5, 0.0, heater_power_available_kwh=0.0)
        # Should be very cold but not crash
        assert h.interior_temp_c >= -50.0
