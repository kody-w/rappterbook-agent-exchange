"""
test_thermal.py — 80+ tests for Mars colony thermal management.

Tests physical bounds, heat balance conservation, edge cases,
comfort scoring, and multi-sol simulation.
"""
from __future__ import annotations

import math
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from thermal import (
    # Constants
    STEFAN_BOLTZMANN,
    MARS_MEAN_TEMP_C,
    KELVIN_OFFSET,
    SOL_HOURS,
    SECONDS_PER_SOL,
    HABITAT_TARGET_TEMP_C,
    HABITAT_TEMP_TOLERANCE_C,
    HABITAT_MIN_TEMP_C,
    HABITAT_MAX_TEMP_C,
    METABOLIC_HEAT_KWH_SOL,
    AEROGEL_R_VALUE,
    DEFAULT_WALL_AREA_M2,
    SOCE_WASTE_HEAT_FRACTION,
    NUCLEAR_WASTE_HEAT_RATIO,
    SOLAR_INVERTER_WASTE,
    RADIATOR_EMISSIVITY,
    RADIATOR_EFFICIENCY_FLOOR,
    STORAGE_LEAK_RATE,
    # Data structures
    HabitatThermal,
    Radiator,
    # Functions
    conductive_loss_kwh,
    radiative_rejection_kwh,
    metabolic_heat_kwh,
    waste_heat_kwh,
    temperature_delta,
    heating_power_needed,
    comfort_score,
    tick_thermal,
)


# ===========================================================================
# Physical constants validation
# ===========================================================================

class TestPhysicalConstants:
    """Verify constants match known physical values."""

    def test_stefan_boltzmann(self):
        """Stefan-Boltzmann constant is ~5.67e-8 W/m²/K⁴."""
        assert abs(STEFAN_BOLTZMANN - 5.67e-8) < 1e-10

    def test_mars_mean_temp(self):
        """Mars mean surface temp is approximately -60°C."""
        assert -70 <= MARS_MEAN_TEMP_C <= -50

    def test_kelvin_offset(self):
        """0°C = 273.15 K."""
        assert abs(KELVIN_OFFSET - 273.15) < 0.01

    def test_sol_hours(self):
        """Mars sol is ~24.66 hours."""
        assert 24.5 <= SOL_HOURS <= 24.8

    def test_seconds_per_sol(self):
        """Derived from sol_hours."""
        assert abs(SECONDS_PER_SOL - SOL_HOURS * 3600) < 1

    def test_habitat_target_comfortable(self):
        """Target temp is in human comfort zone (18-24°C)."""
        assert 18 <= HABITAT_TARGET_TEMP_C <= 24

    def test_metabolic_heat(self):
        """~100W average = ~2.4 kWh/sol per person."""
        # 100W × 24.66h / 1000 = 2.466 kWh
        expected = 0.100 * SOL_HOURS
        assert abs(METABOLIC_HEAT_KWH_SOL - expected) < 0.2

    def test_nuclear_waste_ratio(self):
        """Kilopower: ~7 kW thermal per 1 kW electric."""
        assert 5 <= NUCLEAR_WASTE_HEAT_RATIO <= 10

    def test_aerogel_r_value(self):
        """Multi-layer aerogel R-value ~10 m²K/W."""
        assert 5 <= AEROGEL_R_VALUE <= 20

    def test_freeze_below_stroke(self):
        """Pipe freeze temp < heat stroke temp (sanity)."""
        assert HABITAT_MIN_TEMP_C < HABITAT_MAX_TEMP_C


# ===========================================================================
# HabitatThermal dataclass
# ===========================================================================

class TestHabitatThermal:
    """Tests for habitat thermal state."""

    def test_default_creation(self):
        hab = HabitatThermal()
        assert hab.interior_temp_c == HABITAT_TARGET_TEMP_C
        assert hab.wall_area_m2 == DEFAULT_WALL_AREA_M2
        assert hab.insulation_r == AEROGEL_R_VALUE

    def test_custom_creation(self):
        hab = HabitatThermal(interior_temp_c=15.0, wall_area_m2=300.0)
        assert hab.interior_temp_c == 15.0
        assert hab.wall_area_m2 == 300.0

    def test_wall_area_clamped(self):
        """Wall area must be at least 1.0."""
        hab = HabitatThermal(wall_area_m2=-5.0)
        assert hab.wall_area_m2 == 1.0

    def test_insulation_clamped(self):
        """Insulation R-value must be at least 0.1."""
        hab = HabitatThermal(insulation_r=-1.0)
        assert hab.insulation_r == 0.1

    def test_thermal_mass_clamped(self):
        """Thermal mass must be at least 0.01."""
        hab = HabitatThermal(thermal_mass_kwh_c=-0.5)
        assert hab.thermal_mass_kwh_c == 0.01


# ===========================================================================
# Radiator dataclass
# ===========================================================================

class TestRadiator:
    """Tests for heat rejection radiator."""

    def test_creation(self):
        rad = Radiator(area_m2=50.0)
        assert rad.area_m2 == 50.0
        assert rad.emissivity == RADIATOR_EMISSIVITY
        assert rad.dust_fraction == 0.0

    def test_negative_area_clamped(self):
        rad = Radiator(area_m2=-10.0)
        assert rad.area_m2 == 0.0

    def test_emissivity_clamped(self):
        rad = Radiator(area_m2=10.0, emissivity=1.5)
        assert rad.emissivity == 1.0
        rad2 = Radiator(area_m2=10.0, emissivity=-0.5)
        assert rad2.emissivity == 0.0

    def test_dust_fraction_clamped(self):
        rad = Radiator(area_m2=10.0, dust_fraction=2.0)
        assert rad.dust_fraction == 1.0

    def test_effective_emissivity_clean(self):
        """Clean radiator has full emissivity."""
        rad = Radiator(area_m2=10.0, emissivity=0.9)
        assert abs(rad.effective_emissivity() - 0.9) < 1e-6

    def test_effective_emissivity_dusty(self):
        """Dusty radiator has reduced emissivity but not below floor."""
        rad = Radiator(area_m2=10.0, emissivity=0.9, dust_fraction=0.5)
        eff = rad.effective_emissivity()
        assert eff < 0.9
        assert eff >= RADIATOR_EFFICIENCY_FLOOR * 0.9

    def test_effective_emissivity_fully_dusty(self):
        """Fully dust-covered radiator hits floor."""
        rad = Radiator(area_m2=10.0, emissivity=0.9, dust_fraction=1.0)
        eff = rad.effective_emissivity()
        assert abs(eff - RADIATOR_EFFICIENCY_FLOOR * 0.9) < 1e-6


# ===========================================================================
# Core physics functions
# ===========================================================================

class TestConductiveLoss:
    """Tests for conductive_loss_kwh()."""

    def test_no_gradient_no_loss(self):
        """Equal temps = zero loss."""
        loss = conductive_loss_kwh(20.0, 20.0, 200.0, 10.0)
        assert abs(loss) < 1e-6

    def test_warm_interior_loses_heat(self):
        """Warm interior → positive loss (heat flowing out)."""
        loss = conductive_loss_kwh(21.0, -60.0, 200.0, 10.0)
        assert loss > 0

    def test_cold_interior_gains_heat(self):
        """If exterior is warmer, loss is negative (heat flows in)."""
        loss = conductive_loss_kwh(-60.0, 20.0, 200.0, 10.0)
        assert loss < 0

    def test_better_insulation_less_loss(self):
        """Higher R-value → less loss."""
        loss_low_r = conductive_loss_kwh(21.0, -60.0, 200.0, 5.0)
        loss_high_r = conductive_loss_kwh(21.0, -60.0, 200.0, 20.0)
        assert loss_high_r < loss_low_r

    def test_larger_area_more_loss(self):
        """Larger wall area → more loss."""
        loss_small = conductive_loss_kwh(21.0, -60.0, 100.0, 10.0)
        loss_large = conductive_loss_kwh(21.0, -60.0, 400.0, 10.0)
        assert loss_large > loss_small

    def test_loss_scales_linearly_with_delta_t(self):
        """Double the temperature difference → double the loss."""
        loss1 = conductive_loss_kwh(21.0, 1.0, 200.0, 10.0)  # ΔT=20
        loss2 = conductive_loss_kwh(21.0, -19.0, 200.0, 10.0)  # ΔT=40
        assert abs(loss2 / loss1 - 2.0) < 0.01

    def test_realistic_mars_loss(self):
        """Sanity check: Mars habitat (21°C in, -60°C out, 200m², R=10)."""
        loss = conductive_loss_kwh(21.0, -60.0, 200.0, 10.0)
        # 200 × 81 / 10 = 1620 W = 1.62 kW × 24.66h ≈ 40 kWh/sol
        assert 35 < loss < 45


class TestRadiativeRejection:
    """Tests for radiative_rejection_kwh()."""

    def test_zero_area_zero_rejection(self):
        rad = Radiator(area_m2=0.0)
        assert radiative_rejection_kwh(rad, 21.0) == 0.0

    def test_positive_rejection(self):
        """Warm radiator rejects heat."""
        rad = Radiator(area_m2=50.0)
        rejection = radiative_rejection_kwh(rad, 21.0)
        assert rejection > 0

    def test_hotter_rejects_more(self):
        """T⁴ law: hotter radiator rejects dramatically more."""
        rad = Radiator(area_m2=50.0)
        rej_cool = radiative_rejection_kwh(rad, 0.0)
        rej_warm = radiative_rejection_kwh(rad, 40.0)
        assert rej_warm > rej_cool

    def test_dust_reduces_rejection(self):
        """Dusty radiator rejects less heat."""
        clean = Radiator(area_m2=50.0, dust_fraction=0.0)
        dusty = Radiator(area_m2=50.0, dust_fraction=0.8)
        rej_clean = radiative_rejection_kwh(clean, 21.0)
        rej_dusty = radiative_rejection_kwh(dusty, 21.0)
        assert rej_dusty < rej_clean

    def test_t4_scaling(self):
        """Rejection scales as T⁴ (Stefan-Boltzmann)."""
        rad = Radiator(area_m2=50.0)
        # Compare T₁=300K (27°C) vs T₂=600K (327°C)
        rej1 = radiative_rejection_kwh(rad, 27.0)
        rej2 = radiative_rejection_kwh(rad, 327.0)
        # 600⁴/300⁴ = 16
        ratio = rej2 / rej1
        assert 15 < ratio < 17


class TestMetabolicHeat:
    """Tests for metabolic_heat_kwh()."""

    def test_zero_population(self):
        assert metabolic_heat_kwh(0) == 0.0

    def test_negative_population(self):
        assert metabolic_heat_kwh(-5) == 0.0

    def test_one_person(self):
        assert abs(metabolic_heat_kwh(1) - METABOLIC_HEAT_KWH_SOL) < 1e-6

    def test_linear_scaling(self):
        assert abs(metabolic_heat_kwh(10) - 10 * METABOLIC_HEAT_KWH_SOL) < 1e-6


class TestWasteHeat:
    """Tests for waste_heat_kwh()."""

    def test_zero_everything(self):
        assert waste_heat_kwh(0.0, 0.0, 0.0) == 0.0

    def test_solar_waste(self):
        """Solar inverter waste is 5% of generation."""
        wh = waste_heat_kwh(100.0, 0.0, 0.0)
        assert abs(wh - 100.0 * SOLAR_INVERTER_WASTE) < 1e-6

    def test_nuclear_dominates(self):
        """Nuclear waste heat is massive (7x electric output)."""
        wh = waste_heat_kwh(0.0, 100.0, 0.0)
        assert abs(wh - 100.0 * NUCLEAR_WASTE_HEAT_RATIO) < 1e-6
        assert wh > 500  # 700 kWh thermal from 100 kWh electric

    def test_soce_waste(self):
        """SOCE wastes 60% of consumed power."""
        wh = waste_heat_kwh(0.0, 0.0, 100.0)
        assert abs(wh - 100.0 * SOCE_WASTE_HEAT_FRACTION) < 1e-6

    def test_negative_inputs_clamped(self):
        """Negative power inputs produce zero waste."""
        wh = waste_heat_kwh(-10.0, -20.0, -30.0)
        assert wh == 0.0

    def test_all_sources_sum(self):
        """Total is sum of all three sources."""
        wh = waste_heat_kwh(100.0, 50.0, 200.0)
        expected = (100 * SOLAR_INVERTER_WASTE +
                    50 * NUCLEAR_WASTE_HEAT_RATIO +
                    200 * SOCE_WASTE_HEAT_FRACTION)
        assert abs(wh - expected) < 1e-6


class TestTemperatureDelta:
    """Tests for temperature_delta()."""

    def test_zero_heat_no_change(self):
        assert temperature_delta(0.0, 0.5) == 0.0

    def test_positive_heat_raises_temp(self):
        dt = temperature_delta(5.0, 0.5)
        assert dt == 10.0  # 5 kWh / 0.5 kWh/°C = 10°C

    def test_negative_heat_lowers_temp(self):
        dt = temperature_delta(-5.0, 0.5)
        assert dt == -10.0

    def test_zero_thermal_mass(self):
        """Zero thermal mass → no change (safety)."""
        assert temperature_delta(100.0, 0.0) == 0.0

    def test_higher_mass_smaller_delta(self):
        """More thermal mass → less temperature change."""
        dt1 = temperature_delta(10.0, 0.5)
        dt2 = temperature_delta(10.0, 5.0)
        assert abs(dt2) < abs(dt1)


class TestComfortScore:
    """Tests for comfort_score()."""

    def test_at_target(self):
        """Target temperature = perfect comfort."""
        assert comfort_score(HABITAT_TARGET_TEMP_C) == 1.0

    def test_within_tolerance(self):
        """Within ±3°C = perfect comfort."""
        assert comfort_score(HABITAT_TARGET_TEMP_C + 2.0) == 1.0
        assert comfort_score(HABITAT_TARGET_TEMP_C - 2.0) == 1.0

    def test_at_tolerance_boundary(self):
        """Exactly at tolerance boundary = still perfect."""
        assert comfort_score(HABITAT_TARGET_TEMP_C + HABITAT_TEMP_TOLERANCE_C) == 1.0
        assert comfort_score(HABITAT_TARGET_TEMP_C - HABITAT_TEMP_TOLERANCE_C) == 1.0

    def test_below_freeze_zero(self):
        """At or below pipe freeze temp = zero comfort."""
        assert comfort_score(HABITAT_MIN_TEMP_C) == 0.0
        assert comfort_score(HABITAT_MIN_TEMP_C - 10.0) == 0.0

    def test_above_stroke_zero(self):
        """At or above heat stroke temp = zero comfort."""
        assert comfort_score(HABITAT_MAX_TEMP_C) == 0.0
        assert comfort_score(HABITAT_MAX_TEMP_C + 10.0) == 0.0

    def test_mid_range_between_zero_and_one(self):
        """Halfway between tolerance and limit = ~0.5."""
        mid_cold = (HABITAT_TARGET_TEMP_C - HABITAT_TEMP_TOLERANCE_C + HABITAT_MIN_TEMP_C) / 2
        score = comfort_score(mid_cold)
        assert 0.3 < score < 0.7

    def test_monotonic_decreasing_below_target(self):
        """Comfort decreases as temp drops below tolerance."""
        temps = [18, 15, 10, 5]
        scores = [comfort_score(t) for t in temps]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_always_in_bounds(self):
        """Comfort is always in [0, 1] for any temperature."""
        for t in range(-200, 200, 5):
            score = comfort_score(float(t))
            assert 0.0 <= score <= 1.0


class TestHeatingNeeded:
    """Tests for heating_power_needed()."""

    def test_waste_covers_losses(self):
        """If waste heat exceeds losses, no active heating needed."""
        hab = HabitatThermal()
        needed = heating_power_needed(hab, -60.0, 500.0, 24.0)
        assert needed == 0.0

    def test_deficit_needs_heating(self):
        """If losses exceed free heat, active heating is needed."""
        hab = HabitatThermal()
        needed = heating_power_needed(hab, -60.0, 0.0, 0.0)
        assert needed > 0

    def test_colder_exterior_more_heating(self):
        """Colder outside → more heating needed."""
        hab = HabitatThermal()
        h1 = heating_power_needed(hab, -40.0, 0.0, 0.0)
        h2 = heating_power_needed(hab, -100.0, 0.0, 0.0)
        assert h2 > h1


# ===========================================================================
# tick_thermal integration
# ===========================================================================

class TestTickThermal:
    """Integration tests for the per-sol tick."""

    def _make_system(self):
        hab = HabitatThermal()
        rad = Radiator(area_m2=50.0)
        return hab, rad

    def test_basic_tick(self):
        """Tick returns valid snapshot."""
        hab, rad = self._make_system()
        snap = tick_thermal(
            hab, rad, exterior_temp_c=-60.0, population=10,
            solar_kwh=200.0, nuclear_kwh=100.0, soce_kwh=50.0,
        )
        assert "interior_temp_c" in snap
        assert "comfort_score" in snap
        assert "net_heat_kwh" in snap

    def test_all_fields_present(self):
        """Snapshot contains all expected fields."""
        hab, rad = self._make_system()
        snap = tick_thermal(
            hab, rad, exterior_temp_c=-60.0, population=5,
            solar_kwh=100.0, nuclear_kwh=50.0, soce_kwh=30.0,
        )
        expected = {
            "interior_temp_c", "exterior_temp_c", "temp_delta_c",
            "metabolic_heat_kwh", "waste_heat_kwh", "active_heating_kwh",
            "total_heat_in_kwh", "conductive_loss_kwh",
            "radiator_rejection_kwh", "net_heat_kwh",
            "comfort_score", "heating_needed_kwh",
            "pipe_freeze_risk", "heat_stroke_risk",
        }
        assert set(snap.keys()) == expected

    def test_nuclear_heats_habitat(self):
        """Nuclear waste heat warms the habitat significantly."""
        hab1 = HabitatThermal(interior_temp_c=10.0)
        hab2 = HabitatThermal(interior_temp_c=10.0)
        rad = Radiator(area_m2=50.0)

        snap_no_nuke = tick_thermal(
            hab1, rad, exterior_temp_c=-60.0, population=10,
            solar_kwh=0.0, nuclear_kwh=0.0, soce_kwh=0.0,
        )
        snap_nuke = tick_thermal(
            hab2, rad, exterior_temp_c=-60.0, population=10,
            solar_kwh=0.0, nuclear_kwh=100.0, soce_kwh=0.0,
        )
        assert snap_nuke["interior_temp_c"] > snap_no_nuke["interior_temp_c"]

    def test_freezing_without_heat(self):
        """Without heat sources, habitat cools toward exterior."""
        hab = HabitatThermal(interior_temp_c=21.0, thermal_mass_kwh_c=0.5)
        rad = Radiator(area_m2=0.0)  # no radiator
        snap = tick_thermal(
            hab, rad, exterior_temp_c=-60.0, population=0,
            solar_kwh=0.0, nuclear_kwh=0.0, soce_kwh=0.0,
        )
        assert snap["interior_temp_c"] < 21.0
        assert snap["temp_delta_c"] < 0

    def test_cannot_cool_below_exterior(self):
        """Interior temp never drops below exterior (thermodynamics)."""
        hab = HabitatThermal(interior_temp_c=-50.0, thermal_mass_kwh_c=0.01)
        rad = Radiator(area_m2=0.0)
        snap = tick_thermal(
            hab, rad, exterior_temp_c=-60.0, population=0,
            solar_kwh=0.0, nuclear_kwh=0.0, soce_kwh=0.0,
        )
        assert snap["interior_temp_c"] >= -60.0

    def test_comfort_at_target(self):
        """With balanced heat inputs, comfort stays reasonable."""
        hab = HabitatThermal(interior_temp_c=21.0, thermal_mass_kwh_c=5.0)
        rad = Radiator(area_m2=50.0)
        snap = tick_thermal(
            hab, rad, exterior_temp_c=-60.0, population=10,
            solar_kwh=50.0, nuclear_kwh=0.0, soce_kwh=0.0,
            active_heating_kwh=40.0,
        )
        # Moderate heating should keep temp in a livable range
        assert snap["comfort_score"] > 0.0

    def test_zero_population_less_heat(self):
        """No crew = no metabolic heat."""
        hab, rad = self._make_system()
        snap = tick_thermal(
            hab, rad, exterior_temp_c=-60.0, population=0,
            solar_kwh=0.0, nuclear_kwh=0.0, soce_kwh=0.0,
        )
        assert snap["metabolic_heat_kwh"] == 0.0

    def test_pipe_freeze_risk(self):
        """Pipe freeze risk when interior <= 5°C."""
        hab = HabitatThermal(interior_temp_c=5.0, thermal_mass_kwh_c=100.0)
        rad = Radiator(area_m2=0.0)
        snap = tick_thermal(
            hab, rad, exterior_temp_c=-60.0, population=0,
            solar_kwh=0.0, nuclear_kwh=0.0, soce_kwh=0.0,
        )
        assert snap["pipe_freeze_risk"]


# ===========================================================================
# Multi-sol simulation
# ===========================================================================

class TestMultiSolSimulation:
    """Run thermal system for many sols and check invariants."""

    def test_10_sol_smoke(self):
        """10 sols without crash."""
        hab = HabitatThermal()
        rad = Radiator(area_m2=50.0)
        for _ in range(10):
            snap = tick_thermal(
                hab, rad, exterior_temp_c=-60.0, population=10,
                solar_kwh=150.0, nuclear_kwh=100.0, soce_kwh=40.0,
                active_heating_kwh=20.0,
            )
            assert snap["comfort_score"] >= 0.0
            assert snap["comfort_score"] <= 1.0

    def test_50_sol_thermal_equilibrium(self):
        """With constant inputs, temperature should stabilize."""
        hab = HabitatThermal(interior_temp_c=0.0)
        rad = Radiator(area_m2=50.0)
        temps = []
        for _ in range(50):
            snap = tick_thermal(
                hab, rad, exterior_temp_c=-60.0, population=20,
                solar_kwh=200.0, nuclear_kwh=100.0, soce_kwh=50.0,
                active_heating_kwh=10.0,
            )
            temps.append(snap["interior_temp_c"])
        # Temperature should converge (delta shrinks)
        early_delta = abs(temps[5] - temps[0])
        late_delta = abs(temps[49] - temps[44])
        assert late_delta <= early_delta + 0.1  # approaches equilibrium

    def test_100_sol_bounded(self):
        """Interior temp stays in physical bounds over 100 sols."""
        hab = HabitatThermal()
        rad = Radiator(area_m2=50.0)
        for _ in range(100):
            tick_thermal(
                hab, rad, exterior_temp_c=-60.0, population=10,
                solar_kwh=100.0, nuclear_kwh=50.0, soce_kwh=30.0,
            )
            assert hab.interior_temp_c >= -120.0  # can't be colder than Mars min
            assert hab.interior_temp_c <= 80.0


# ===========================================================================
# Property-based invariants
# ===========================================================================

class TestInvariants:
    """Property-based checks that must hold for any input."""

    @pytest.mark.parametrize("pop", [0, 1, 10, 50])
    def test_metabolic_nonnegative(self, pop):
        assert metabolic_heat_kwh(pop) >= 0.0

    @pytest.mark.parametrize("solar,nuclear,soce", [
        (0, 0, 0), (100, 0, 0), (0, 100, 0), (0, 0, 100), (100, 100, 100),
    ])
    def test_waste_heat_nonnegative(self, solar, nuclear, soce):
        assert waste_heat_kwh(solar, nuclear, soce) >= 0.0

    @pytest.mark.parametrize("temp", [-120, -60, -30, 0, 10, 21, 30, 40, 50])
    def test_comfort_in_bounds(self, temp):
        score = comfort_score(float(temp))
        assert 0.0 <= score <= 1.0

    def test_energy_conservation(self):
        """Net heat = total_in - total_out (accounting check)."""
        hab = HabitatThermal(interior_temp_c=30.0)
        rad = Radiator(area_m2=50.0)
        snap = tick_thermal(
            hab, rad, exterior_temp_c=-60.0, population=10,
            solar_kwh=100.0, nuclear_kwh=50.0, soce_kwh=30.0,
        )
        total_in = snap["total_heat_in_kwh"]
        total_out = snap["conductive_loss_kwh"] + snap["radiator_rejection_kwh"]
        net = total_in - total_out
        assert abs(snap["net_heat_kwh"] - net) < 0.01

    @pytest.mark.parametrize("ext", [-120.0, -60.0, -30.0, 0.0, 20.0])
    def test_interior_above_exterior(self, ext):
        """Interior never goes below exterior temperature."""
        hab = HabitatThermal(interior_temp_c=ext + 1, thermal_mass_kwh_c=0.01)
        rad = Radiator(area_m2=0.0)
        tick_thermal(
            hab, rad, exterior_temp_c=ext, population=0,
            solar_kwh=0.0, nuclear_kwh=0.0, soce_kwh=0.0,
        )
        assert hab.interior_temp_c >= ext
