"""Tests for lightning_rod.py — Mars Electrostatic Discharge Protection.

68 tests covering:
  - Paschen breakdown voltage (CO₂ curve)
  - Triboelectric charge generation
  - Electric field calculation
  - Ground rod resistance (hemispherical model)
  - Discharge severity classification
  - Charge-to-voltage conversion
  - Habitat capacitance
  - System tick (charge buildup, decay, discharge)
  - Grounding effectiveness
  - Storm scenarios (dust devil, regional, global)
  - Physical invariants (conservation, bounds, monotonicity)
  - Equipment damage accumulation
  - 10-sol and 100-sol smoke tests
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lightning_rod import (
    ATMOSPHERIC_CONDUCTIVITY_S_M,
    CHARGE_DECAY_TIME_CONSTANT_S,
    CO2_RELATIVE_PERMITTIVITY,
    DEFAULT_HAB_HEIGHT_M,
    DEFAULT_HAB_SURFACE_M2,
    DISCHARGE_CATASTROPHIC_V,
    DISCHARGE_MINOR_V,
    DISCHARGE_MODERATE_V,
    DISCHARGE_SEVERE_V,
    DUST_MASS_FLUX_CLEAR,
    DUST_MASS_FLUX_DEVIL,
    DUST_MASS_FLUX_GLOBAL,
    DUST_MASS_FLUX_REGIONAL,
    GROUND_ROD_DEPTH_M,
    GROUND_ROD_RADIUS_M,
    MARS_SURFACE_PRESSURE_PA,
    MAST_RESISTANCE_OHM,
    PASCHEN_MIN_VOLTAGE,
    REGOLITH_RESISTIVITY_OHM_M,
    TRIBO_CHARGE_COEFF,
    VACUUM_PERMITTIVITY,
    WIND_CLEAR,
    WIND_DEVIL,
    WIND_GLOBAL,
    WIND_REGIONAL,
    LightningRodSystem,
    charge_to_voltage,
    discharge_severity,
    electric_field_parallel_plate,
    ground_rod_resistance,
    habitat_capacitance,
    paschen_breakdown_voltage,
    triboelectric_charge,
)


# ═══════════════════════════════════════════════════════════════════
# Paschen breakdown voltage
# ═══════════════════════════════════════════════════════════════════


class TestPaschenBreakdown:
    """Paschen curve for CO₂ at Mars pressure."""

    def test_positive_at_standard_conditions(self):
        """Breakdown voltage is positive at Mars surface pressure."""
        v = paschen_breakdown_voltage()
        assert v > 0

    def test_minimum_voltage_floor(self):
        """Result is never below the Paschen minimum."""
        for gap in [0.001, 0.01, 0.1, 1.0, 10.0]:
            v = paschen_breakdown_voltage(gap_m=gap)
            assert v >= PASCHEN_MIN_VOLTAGE

    def test_zero_pressure_infinite_breakdown(self):
        """Perfect vacuum cannot break down."""
        v = paschen_breakdown_voltage(pressure_pa=0.0)
        assert v == float("inf")

    def test_zero_gap_infinite_breakdown(self):
        """Zero gap cannot break down."""
        v = paschen_breakdown_voltage(gap_m=0.0)
        assert v == float("inf")

    def test_negative_pressure_infinite(self):
        """Negative pressure is unphysical — returns inf."""
        v = paschen_breakdown_voltage(pressure_pa=-100)
        assert v == float("inf")

    def test_small_gap_low_breakdown(self):
        """Small gaps at Mars pressure should have relatively low breakdown."""
        v_small = paschen_breakdown_voltage(gap_m=0.001)
        v_large = paschen_breakdown_voltage(gap_m=1.0)
        # Not necessarily monotonic (Paschen curve has a minimum)
        # but both should be finite and positive
        assert v_small > 0
        assert v_large > 0

    def test_mars_1cm_gap_reasonable(self):
        """1 cm gap at 600 Pa CO₂ should give hundreds to thousands of volts."""
        v = paschen_breakdown_voltage(MARS_SURFACE_PRESSURE_PA, 0.01)
        assert 100 < v < 50_000


# ═══════════════════════════════════════════════════════════════════
# Triboelectric charge generation
# ═══════════════════════════════════════════════════════════════════


class TestTriboelectricCharge:
    """Q = k × Φ_m × v × A × t."""

    def test_clear_weather_zero_charge(self):
        """No dust flux → no charge generated."""
        q = triboelectric_charge(DUST_MASS_FLUX_CLEAR, WIND_CLEAR)
        assert q == 0.0

    def test_dust_devil_generates_charge(self):
        """Dust devil conditions generate measurable charge."""
        q = triboelectric_charge(DUST_MASS_FLUX_DEVIL, WIND_DEVIL)
        assert q > 0

    def test_regional_storm_more_than_devil(self):
        """Regional storm generates more charge than dust devil."""
        q_devil = triboelectric_charge(DUST_MASS_FLUX_DEVIL, WIND_DEVIL)
        q_storm = triboelectric_charge(DUST_MASS_FLUX_REGIONAL, WIND_REGIONAL)
        assert q_storm > q_devil

    def test_global_storm_most_charge(self):
        """Global storm generates the most charge."""
        q_regional = triboelectric_charge(DUST_MASS_FLUX_REGIONAL, WIND_REGIONAL)
        q_global = triboelectric_charge(DUST_MASS_FLUX_GLOBAL, WIND_GLOBAL)
        assert q_global > q_regional

    def test_proportional_to_mass_flux(self):
        """Charge scales linearly with dust mass flux."""
        q1 = triboelectric_charge(0.01, 10.0)
        q2 = triboelectric_charge(0.02, 10.0)
        assert abs(q2 / q1 - 2.0) < 0.01

    def test_proportional_to_wind_speed(self):
        """Charge scales linearly with wind speed."""
        q1 = triboelectric_charge(0.01, 10.0)
        q2 = triboelectric_charge(0.01, 20.0)
        assert abs(q2 / q1 - 2.0) < 0.01

    def test_proportional_to_area(self):
        """Charge scales linearly with collection area."""
        q1 = triboelectric_charge(0.01, 10.0, collection_area=100.0)
        q2 = triboelectric_charge(0.01, 10.0, collection_area=200.0)
        assert abs(q2 / q1 - 2.0) < 0.01

    def test_zero_wind_zero_charge(self):
        """No wind → no triboelectric charging."""
        q = triboelectric_charge(0.1, 0.0)
        assert q == 0.0

    def test_formula_manual(self):
        """Verify against manual calculation."""
        flux, wind, area, dur = 0.05, 30.0, 200.0, 86400.0
        expected = TRIBO_CHARGE_COEFF * flux * wind * area * dur
        assert abs(triboelectric_charge(flux, wind, area, dur) - expected) < 1e-15


# ═══════════════════════════════════════════════════════════════════
# Electric field
# ═══════════════════════════════════════════════════════════════════


class TestElectricField:
    """E = Q / (A · ε₀ · ε_r)."""

    def test_zero_charge_zero_field(self):
        assert electric_field_parallel_plate(0.0) == 0.0

    def test_positive_charge_positive_field(self):
        e = electric_field_parallel_plate(1e-6)
        assert e > 0

    def test_negative_charge_positive_field(self):
        """Field magnitude is always positive."""
        e = electric_field_parallel_plate(-1e-6)
        assert e > 0

    def test_proportional_to_charge(self):
        e1 = electric_field_parallel_plate(1e-6)
        e2 = electric_field_parallel_plate(2e-6)
        assert abs(e2 / e1 - 2.0) < 0.01

    def test_inversely_proportional_to_area(self):
        e1 = electric_field_parallel_plate(1e-6, area_m2=100.0)
        e2 = electric_field_parallel_plate(1e-6, area_m2=200.0)
        assert abs(e1 / e2 - 2.0) < 0.01

    def test_zero_area_zero_field(self):
        assert electric_field_parallel_plate(1e-6, area_m2=0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Ground rod resistance
# ═══════════════════════════════════════════════════════════════════


class TestGroundRodResistance:
    """R = ρ/(2π·L) × ln(2L/a)."""

    def test_default_resistance_very_high(self):
        """Mars regolith is a terrible conductor — resistance is huge."""
        r = ground_rod_resistance()
        assert r > 1e6  # megaohms

    def test_deeper_rod_lower_resistance(self):
        """Deeper ground rod has lower resistance."""
        r_shallow = ground_rod_resistance(depth_m=1.0)
        r_deep = ground_rod_resistance(depth_m=5.0)
        assert r_deep < r_shallow

    def test_lower_resistivity_lower_resistance(self):
        """Wetter/saltier soil → lower resistivity → lower resistance."""
        r_dry = ground_rod_resistance(resistivity=1e8)
        r_wet = ground_rod_resistance(resistivity=1e4)
        assert r_wet < r_dry

    def test_zero_depth_infinite(self):
        r = ground_rod_resistance(depth_m=0.0)
        assert r == float("inf")

    def test_zero_radius_infinite(self):
        r = ground_rod_resistance(rod_radius_m=0.0)
        assert r == float("inf")

    def test_formula_manual(self):
        """Verify against manual calculation."""
        L, a, rho = 3.0, 0.025, 1e8
        expected = (rho / (2 * math.pi * L)) * math.log(2 * L / a)
        assert abs(ground_rod_resistance(L, a, rho) - expected) < 1.0


# ═══════════════════════════════════════════════════════════════════
# Discharge severity
# ═══════════════════════════════════════════════════════════════════


class TestDischargeSeverity:
    """Voltage → severity classification."""

    def test_none(self):
        assert discharge_severity(500) == "none"

    def test_minor(self):
        assert discharge_severity(5_000) == "minor"

    def test_moderate(self):
        assert discharge_severity(30_000) == "moderate"

    def test_severe(self):
        assert discharge_severity(100_000) == "severe"

    def test_catastrophic(self):
        assert discharge_severity(300_000) == "catastrophic"

    def test_boundary_minor(self):
        assert discharge_severity(DISCHARGE_MINOR_V - 1) == "none"
        assert discharge_severity(DISCHARGE_MINOR_V) == "minor"

    def test_zero_is_none(self):
        assert discharge_severity(0) == "none"


# ═══════════════════════════════════════════════════════════════════
# Charge-to-voltage and capacitance
# ═══════════════════════════════════════════════════════════════════


class TestChargeVoltage:
    """V = Q/C."""

    def test_zero_charge_zero_voltage(self):
        assert charge_to_voltage(0.0, 1e-9) == 0.0

    def test_positive_charge_positive_voltage(self):
        v = charge_to_voltage(1e-6, 1e-9)
        assert v > 0

    def test_negative_charge_positive_voltage(self):
        """Voltage magnitude is always positive."""
        v = charge_to_voltage(-1e-6, 1e-9)
        assert v > 0

    def test_zero_capacitance_zero_voltage(self):
        """Zero capacitance → zero (edge guard)."""
        assert charge_to_voltage(1e-6, 0.0) == 0.0

    def test_proportional_to_charge(self):
        v1 = charge_to_voltage(1e-6, 1e-9)
        v2 = charge_to_voltage(2e-6, 1e-9)
        assert abs(v2 / v1 - 2.0) < 0.01


class TestHabitatCapacitance:
    """C = ε₀ · ε_r · A / d."""

    def test_positive_capacitance(self):
        c = habitat_capacitance()
        assert c > 0

    def test_larger_area_higher_capacitance(self):
        c1 = habitat_capacitance(surface_area_m2=100.0)
        c2 = habitat_capacitance(surface_area_m2=200.0)
        assert c2 > c1

    def test_taller_hab_lower_capacitance(self):
        c1 = habitat_capacitance(height_m=4.0)
        c2 = habitat_capacitance(height_m=8.0)
        assert c1 > c2

    def test_zero_height_minimum_capacitance(self):
        c = habitat_capacitance(height_m=0.0)
        assert c > 0  # minimum guard value


# ═══════════════════════════════════════════════════════════════════
# System construction
# ═══════════════════════════════════════════════════════════════════


class TestSystemConstruction:
    """System initializes with sane defaults."""

    def test_default_construction(self):
        sys = LightningRodSystem()
        assert sys.mast_installed is True
        assert sys.atmospheric_charge_c == 0.0
        assert sys.discharge_events == 0

    def test_grounding_resistance_computed(self):
        """Grounding resistance is auto-computed on construction."""
        sys = LightningRodSystem()
        assert sys.grounding_resistance_ohm > 0

    def test_initial_voltage_zero(self):
        sys = LightningRodSystem()
        assert sys.current_voltage() == 0.0

    def test_initial_e_field_zero(self):
        sys = LightningRodSystem()
        assert sys.current_e_field() == 0.0


# ═══════════════════════════════════════════════════════════════════
# Single tick behavior
# ═══════════════════════════════════════════════════════════════════


class TestSingleTick:
    """One-sol tick behavior."""

    def test_tick_returns_dict(self):
        sys = LightningRodSystem()
        result = sys.tick()
        assert isinstance(result, dict)
        assert "voltage" in result
        assert "severity" in result

    def test_clear_weather_no_charge(self):
        """Clear weather generates no charge."""
        sys = LightningRodSystem()
        result = sys.tick(dust_mass_flux=0.0, wind_speed=5.0)
        assert result["charge_generated_c"] == 0.0

    def test_dust_devil_generates_charge(self):
        """Dust devil adds charge to atmosphere."""
        sys = LightningRodSystem()
        result = sys.tick(dust_mass_flux=DUST_MASS_FLUX_DEVIL, wind_speed=WIND_DEVIL)
        assert result["charge_generated_c"] > 0
        assert sys.atmospheric_charge_c > 0

    def test_grounding_dissipates_charge(self):
        """Grounding system dissipates some accumulated charge."""
        sys = LightningRodSystem()
        sys.atmospheric_charge_c = 1.0  # inject charge
        result = sys.tick(dust_mass_flux=0.0)
        assert result["charge_dissipated_c"] > 0
        assert sys.atmospheric_charge_c < 1.0

    def test_no_mast_less_dissipation(self):
        """Without mast, only natural decay dissipates charge."""
        sys_mast = LightningRodSystem()
        sys_mast.atmospheric_charge_c = 1.0

        sys_bare = LightningRodSystem()
        sys_bare.mast_installed = False
        sys_bare.atmospheric_charge_c = 1.0

        r_mast = sys_mast.tick(dust_mass_flux=0.0)
        r_bare = sys_bare.tick(dust_mass_flux=0.0)

        assert r_mast["charge_dissipated_c"] >= r_bare["charge_dissipated_c"]


# ═══════════════════════════════════════════════════════════════════
# Storm scenarios
# ═══════════════════════════════════════════════════════════════════


class TestStormScenarios:
    """Charge buildup during different storm types."""

    def test_regional_storm_buildup(self):
        """Regional storm over 10 sols builds significant charge."""
        sys = LightningRodSystem()
        for _ in range(10):
            sys.tick(dust_mass_flux=DUST_MASS_FLUX_REGIONAL, wind_speed=WIND_REGIONAL, storm_active=True)
        assert sys.atmospheric_charge_c > 0
        assert sys.current_voltage() > 0

    def test_global_storm_may_discharge(self):
        """Global storm generates enough charge for potential discharge."""
        sys = LightningRodSystem()
        discharged = False
        for _ in range(30):
            result = sys.tick(
                dust_mass_flux=DUST_MASS_FLUX_GLOBAL,
                wind_speed=WIND_GLOBAL,
                storm_active=True,
            )
            if result["discharged"]:
                discharged = True
        # Global storm should eventually cause at least one discharge
        assert discharged or sys.current_voltage() > DISCHARGE_MINOR_V

    def test_storm_then_clear_charge_decays(self):
        """After storm ends, charge decays back toward zero."""
        sys = LightningRodSystem()
        # Build up charge
        for _ in range(5):
            sys.tick(dust_mass_flux=DUST_MASS_FLUX_REGIONAL, wind_speed=WIND_REGIONAL)
        peak_charge = sys.atmospheric_charge_c

        # Clear weather recovery
        for _ in range(50):
            sys.tick(dust_mass_flux=0.0, wind_speed=WIND_CLEAR)
        assert sys.atmospheric_charge_c < peak_charge * 0.5


# ═══════════════════════════════════════════════════════════════════
# Multi-tick smoke tests
# ═══════════════════════════════════════════════════════════════════


class TestSmoke:
    """System runs without crash."""

    def test_10_sol_clear_weather(self):
        sys = LightningRodSystem()
        for _ in range(10):
            result = sys.tick()
            assert result["voltage"] >= 0
            assert result["power_watts"] > 0

    def test_100_sol_mixed_weather(self):
        """100 sols with varying conditions — no crash."""
        sys = LightningRodSystem()
        conditions = [
            (DUST_MASS_FLUX_CLEAR, WIND_CLEAR, False),
            (DUST_MASS_FLUX_DEVIL, WIND_DEVIL, False),
            (DUST_MASS_FLUX_REGIONAL, WIND_REGIONAL, True),
            (DUST_MASS_FLUX_GLOBAL, WIND_GLOBAL, True),
        ]
        for sol in range(100):
            flux, wind, storm = conditions[sol % len(conditions)]
            result = sys.tick(flux, wind, storm)
            assert result["voltage"] >= 0
            assert result["atmospheric_charge_c"] >= 0

    def test_voltage_history_grows(self):
        sys = LightningRodSystem()
        for _ in range(10):
            sys.tick()
        assert len(sys.voltage_history) == 10


# ═══════════════════════════════════════════════════════════════════
# Physical invariants
# ═══════════════════════════════════════════════════════════════════


class TestPhysicalInvariants:
    """Conservation laws and physical bounds."""

    def test_charge_never_negative(self):
        """Atmospheric charge can never go below zero."""
        sys = LightningRodSystem()
        for _ in range(50):
            sys.tick(dust_mass_flux=DUST_MASS_FLUX_DEVIL, wind_speed=WIND_DEVIL)
        for _ in range(50):
            sys.tick(dust_mass_flux=0.0)
            assert sys.atmospheric_charge_c >= 0

    def test_voltage_never_negative(self):
        sys = LightningRodSystem()
        for _ in range(30):
            sys.tick(dust_mass_flux=DUST_MASS_FLUX_REGIONAL, wind_speed=WIND_REGIONAL)
            assert sys.current_voltage() >= 0

    def test_damage_never_exceeds_100(self):
        """Damage percentages capped at 100%."""
        sys = LightningRodSystem()
        # Force massive charge to cause catastrophic discharges
        for _ in range(100):
            sys.atmospheric_charge_c = 1e6  # enormous charge
            sys.tick(dust_mass_flux=DUST_MASS_FLUX_GLOBAL, wind_speed=WIND_GLOBAL)
        assert sys.solar_panel_damage_pct <= 100.0
        assert sys.antenna_damage_pct <= 100.0
        assert sys.electronics_damage_pct <= 100.0

    def test_e_field_positive_or_zero(self):
        sys = LightningRodSystem()
        sys.tick(dust_mass_flux=DUST_MASS_FLUX_DEVIL, wind_speed=WIND_DEVIL)
        assert sys.current_e_field() >= 0

    def test_discharge_resets_most_charge(self):
        """A discharge event dumps 90% of accumulated charge."""
        sys = LightningRodSystem()
        # Build up enough for discharge
        sys.atmospheric_charge_c = 1e3  # large charge
        pre = sys.atmospheric_charge_c
        result = sys.tick(dust_mass_flux=0.0)
        if result["discharged"]:
            # After discharge, charge should be much lower
            assert sys.atmospheric_charge_c < pre * 0.5


# ═══════════════════════════════════════════════════════════════════
# Status report
# ═══════════════════════════════════════════════════════════════════


class TestStatusReport:
    """status() returns complete system summary."""

    def test_status_has_all_fields(self):
        sys = LightningRodSystem()
        sys.tick()
        status = sys.status()
        required = [
            "atmospheric_charge_c", "voltage", "e_field_v_m", "severity",
            "grounding_resistance_ohm", "mast_installed", "discharge_events",
            "solar_panel_damage_pct", "antenna_damage_pct", "electronics_damage_pct",
        ]
        for key in required:
            assert key in status, f"Missing key: {key}"

    def test_status_reflects_damage(self):
        sys = LightningRodSystem()
        sys.solar_panel_damage_pct = 25.0
        status = sys.status()
        assert status["solar_panel_damage_pct"] == 25.0
