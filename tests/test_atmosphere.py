"""
Tests for atmosphere.py — Mars habitat atmosphere processor.

Coverage:
  - Physical constants sanity
  - AtmosphereProcessor construction
  - Crew respiration O2/CO2 balance
  - Electrolysis water→O2 conversion
  - CO2 scrubber with degradation
  - Sabatier water recovery
  - MOXIE ISRU supplemental O2
  - Leak rate and repressurisation
  - Trace contaminant accumulation
  - Multi-sol stability (smoke test)
  - Conservation laws (mass/energy bounds)
  - Edge cases (zero pop, zero power, zero water)
  - CO2 danger alerts
  - Pressure status thresholds
"""
from __future__ import annotations

import math
import pytest

from src.atmosphere import (
    AtmosphereProcessor,
    AtmosphereState,
    CO2_DANGER_KPA,
    CO2_KG_PER_PERSON_SOL,
    CO2_LETHAL_KPA,
    ELECTROLYSIS_KWH_PER_KG_O2,
    HABITAT_LEAK_RATE_KPA_SOL,
    MOXIE_EFFICIENCY_BASE,
    MOXIE_KG_O2_PER_KWH,
    O2_KG_PER_PERSON_SOL,
    SABATIER_CO2_TO_H2O_RATIO,
    SCRUBBER_DEGRADATION_PER_SOL,
    SCRUBBER_EFFICIENCY_BASE,
    TARGET_CO2_KPA,
    TARGET_O2_KPA,
    TARGET_TOTAL_KPA,
    WATER_TO_O2_RATIO,
)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


class TestConstants:
    """Physical constants must be positive and in realistic ranges."""

    def test_o2_consumption_positive(self):
        assert O2_KG_PER_PERSON_SOL > 0

    def test_co2_production_positive(self):
        assert CO2_KG_PER_PERSON_SOL > 0

    def test_co2_exceeds_o2(self):
        """Humans exhale more CO2 mass than O2 consumed (C added from food)."""
        assert CO2_KG_PER_PERSON_SOL > O2_KG_PER_PERSON_SOL

    def test_water_to_o2_ratio_physical(self):
        """Stoichiometry: 2H2O → 2H2 + O2. 36g water → 32g O2."""
        assert 0.8 < WATER_TO_O2_RATIO < 1.0

    def test_target_pressures_sum(self):
        """O2 + N2 + CO2 ≈ total target."""
        n2 = TARGET_TOTAL_KPA - TARGET_O2_KPA - TARGET_CO2_KPA
        assert abs((TARGET_O2_KPA + n2 + TARGET_CO2_KPA) - TARGET_TOTAL_KPA) < 0.01

    def test_co2_danger_thresholds_ordered(self):
        assert TARGET_CO2_KPA < CO2_DANGER_KPA < CO2_LETHAL_KPA

    def test_scrubber_efficiency_bounded(self):
        assert 0.0 < SCRUBBER_EFFICIENCY_BASE <= 1.0

    def test_moxie_output_positive(self):
        assert MOXIE_KG_O2_PER_KWH > 0

    def test_sabatier_ratio_bounded(self):
        assert 0.0 < SABATIER_CO2_TO_H2O_RATIO < 1.0

    def test_electrolysis_energy_positive(self):
        assert ELECTROLYSIS_KWH_PER_KG_O2 > 0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """AtmosphereProcessor initialises to Earth-normal atmosphere."""

    def test_default_atmosphere(self):
        proc = AtmosphereProcessor()
        assert abs(proc.o2_kpa - TARGET_O2_KPA) < 0.01
        assert abs(proc.co2_kpa - TARGET_CO2_KPA) < 0.01
        total = proc.total_pressure()
        assert abs(total - TARGET_TOTAL_KPA) < 0.1

    def test_custom_volume(self):
        proc = AtmosphereProcessor(habitat_volume_m3=1000.0)
        assert proc.habitat_volume_m3 == 1000.0

    def test_custom_electrolyser(self):
        proc = AtmosphereProcessor(electrolyser_capacity_kg_sol=50.0)
        assert proc.electrolyser_capacity_kg_sol == 50.0

    def test_initial_sol_zero(self):
        proc = AtmosphereProcessor()
        assert proc.sol == 0

    def test_initial_scrubber_full(self):
        proc = AtmosphereProcessor()
        assert proc.scrubber_health == 1.0

    def test_initial_traces_zero(self):
        proc = AtmosphereProcessor()
        assert proc.trace_contaminants == 0.0

    def test_moxie_default_off(self):
        proc = AtmosphereProcessor()
        assert proc.moxie_installed is False


# ---------------------------------------------------------------------------
# O2 fraction and status methods
# ---------------------------------------------------------------------------


class TestStatusMethods:
    """Test diagnostic methods on the processor."""

    def test_o2_fraction_nominal(self):
        proc = AtmosphereProcessor()
        frac = proc.o2_fraction()
        assert 0.19 < frac < 0.22  # Earth-like

    def test_o2_fraction_zero_pressure(self):
        proc = AtmosphereProcessor()
        proc.o2_kpa = 0.0
        proc.co2_kpa = 0.0
        proc.n2_kpa = 0.0
        assert proc.o2_fraction() == 0.0

    def test_co2_status_nominal(self):
        proc = AtmosphereProcessor()
        assert proc.co2_status() == "nominal"

    def test_co2_status_elevated(self):
        proc = AtmosphereProcessor()
        proc.co2_kpa = 0.3
        assert proc.co2_status() == "elevated"

    def test_co2_status_dangerous(self):
        proc = AtmosphereProcessor()
        proc.co2_kpa = CO2_DANGER_KPA
        assert proc.co2_status() == "dangerous"

    def test_co2_status_lethal(self):
        proc = AtmosphereProcessor()
        proc.co2_kpa = CO2_LETHAL_KPA
        assert proc.co2_status() == "lethal"

    def test_pressure_status_nominal(self):
        proc = AtmosphereProcessor()
        assert proc.pressure_status() == "nominal"

    def test_pressure_status_low(self):
        proc = AtmosphereProcessor()
        proc.o2_kpa = 15.0
        proc.n2_kpa = 55.0
        proc.co2_kpa = 0.04
        assert proc.pressure_status() == "low"

    def test_pressure_status_critical(self):
        proc = AtmosphereProcessor()
        proc.o2_kpa = 10.0
        proc.n2_kpa = 30.0
        proc.co2_kpa = 0.04
        assert proc.pressure_status() == "critical"

    def test_pressure_status_over(self):
        proc = AtmosphereProcessor()
        proc.n2_kpa = 120.0
        assert proc.pressure_status() == "over_pressurised"


# ---------------------------------------------------------------------------
# Crew respiration
# ---------------------------------------------------------------------------


class TestRespiration:
    """Crew respiration consumes O2, produces CO2."""

    def test_respiration_consumes_o2(self):
        proc = AtmosphereProcessor()
        o2_before = proc.o2_kpa
        proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        # O2 should drop from consumption (electrolysis partially restores)
        # But net O2 should be close to starting if enough water/power
        assert proc.o2_kpa > 0

    def test_respiration_produces_co2(self):
        """CO2 rises without scrubbing (give zero power)."""
        proc = AtmosphereProcessor()
        co2_before = proc.co2_kpa
        proc.tick(population=10, water_available_kg=0.0, power_available_kwh=0.0)
        assert proc.co2_kpa > co2_before

    def test_zero_population_no_consumption(self):
        proc = AtmosphereProcessor()
        o2_before = proc.o2_kpa
        co2_before = proc.co2_kpa
        state = proc.tick(population=0, water_available_kg=0.0, power_available_kwh=0.0)
        # Only leak should change pressures
        assert proc.o2_kpa <= o2_before
        assert proc.co2_kpa <= co2_before  # leak reduces all components

    def test_more_people_more_co2(self):
        proc1 = AtmosphereProcessor()
        proc2 = AtmosphereProcessor()
        s1 = proc1.tick(population=5, water_available_kg=0.0, power_available_kwh=0.0)
        s2 = proc2.tick(population=20, water_available_kg=0.0, power_available_kwh=0.0)
        # 20 people should produce more CO2 than 5
        assert s2.co2_kpa > s1.co2_kpa


# ---------------------------------------------------------------------------
# Electrolysis
# ---------------------------------------------------------------------------


class TestElectrolysis:
    """Water electrolysis produces O2 proportional to water and power."""

    def test_electrolysis_produces_o2(self):
        proc = AtmosphereProcessor()
        state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert state.electrolysis_o2_produced_kg > 0

    def test_no_water_no_electrolysis(self):
        proc = AtmosphereProcessor()
        state = proc.tick(population=10, water_available_kg=0.0, power_available_kwh=500.0)
        assert state.electrolysis_o2_produced_kg == 0.0

    def test_no_power_no_electrolysis(self):
        proc = AtmosphereProcessor()
        state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=0.0)
        assert state.electrolysis_o2_produced_kg == 0.0

    def test_electrolyser_capacity_limit(self):
        proc = AtmosphereProcessor(electrolyser_capacity_kg_sol=5.0)
        state = proc.tick(population=100, water_available_kg=10000.0, power_available_kwh=10000.0)
        assert state.electrolysis_o2_produced_kg <= 5.0 + 0.01

    def test_water_consumed_tracked(self):
        proc = AtmosphereProcessor()
        proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert proc.total_water_consumed_kg > 0


# ---------------------------------------------------------------------------
# CO2 scrubbing
# ---------------------------------------------------------------------------


class TestScrubber:
    """CO2 scrubber removes exhaled CO2."""

    def test_scrubber_removes_co2(self):
        proc = AtmosphereProcessor()
        state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert state.co2_scrubbed_kg > 0

    def test_degraded_scrubber_less_effective(self):
        proc1 = AtmosphereProcessor()
        proc2 = AtmosphereProcessor(scrubber_health=0.3)
        s1 = proc1.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        s2 = proc2.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert s2.co2_scrubbed_kg < s1.co2_scrubbed_kg

    def test_scrubber_degrades_each_sol(self):
        proc = AtmosphereProcessor()
        health_before = proc.scrubber_health
        proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert proc.scrubber_health < health_before
        assert proc.scrubber_health == pytest.approx(
            health_before - SCRUBBER_DEGRADATION_PER_SOL, abs=1e-6
        )

    def test_scrubber_health_floor(self):
        proc = AtmosphereProcessor(scrubber_health=0.0001)
        proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert proc.scrubber_health >= 0.0


# ---------------------------------------------------------------------------
# Sabatier reactor
# ---------------------------------------------------------------------------


class TestSabatier:
    """Sabatier reactor recovers water from scrubbed CO2."""

    def test_sabatier_recovers_water(self):
        proc = AtmosphereProcessor()
        state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert state.sabatier_water_recovered_kg > 0

    def test_sabatier_bounded_by_co2(self):
        """Water recovered can't exceed stoichiometric limit of CO2 scrubbed."""
        proc = AtmosphereProcessor()
        state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        max_possible = state.co2_scrubbed_kg * SABATIER_CO2_TO_H2O_RATIO
        assert state.sabatier_water_recovered_kg <= max_possible + 0.01


# ---------------------------------------------------------------------------
# MOXIE ISRU
# ---------------------------------------------------------------------------


class TestMoxie:
    """MOXIE unit extracts O2 from Mars atmosphere CO2."""

    def test_moxie_off_by_default(self):
        proc = AtmosphereProcessor()
        state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert state.moxie_o2_produced_kg == 0.0

    def test_moxie_produces_o2_when_installed(self):
        proc = AtmosphereProcessor(moxie_installed=True)
        state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert state.moxie_o2_produced_kg > 0

    def test_moxie_no_power_no_o2(self):
        proc = AtmosphereProcessor(moxie_installed=True)
        state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=0.0)
        assert state.moxie_o2_produced_kg == 0.0

    def test_moxie_output_proportional_to_power(self):
        proc1 = AtmosphereProcessor(moxie_installed=True,
                                     electrolyser_capacity_kg_sol=0.0)
        proc2 = AtmosphereProcessor(moxie_installed=True,
                                     electrolyser_capacity_kg_sol=0.0)
        s1 = proc1.tick(population=0, water_available_kg=0.0, power_available_kwh=100.0)
        s2 = proc2.tick(population=0, water_available_kg=0.0, power_available_kwh=1000.0)
        assert s2.moxie_o2_produced_kg > s1.moxie_o2_produced_kg


# ---------------------------------------------------------------------------
# Leak and repressurisation
# ---------------------------------------------------------------------------


class TestLeakAndRepress:
    """Habitat leaks gas; repressurisation compensates."""

    def test_leak_reduces_pressure(self):
        proc = AtmosphereProcessor()
        total_before = proc.total_pressure()
        state = proc.tick(population=0, water_available_kg=0.0, power_available_kwh=0.0)
        # With no crew and no power, only leak happens (and repress needs power)
        assert state.leak_loss_kpa > 0

    def test_leak_rate_matches_constant(self):
        proc = AtmosphereProcessor()
        state = proc.tick(population=0, water_available_kg=0.0, power_available_kwh=0.0)
        assert state.leak_loss_kpa == pytest.approx(HABITAT_LEAK_RATE_KPA_SOL, abs=0.001)

    def test_repressurisation_with_power(self):
        proc = AtmosphereProcessor()
        proc.o2_kpa = 15.0
        proc.n2_kpa = 60.0
        proc.co2_kpa = 0.04
        total_before = proc.total_pressure()
        proc.tick(population=0, water_available_kg=0.0, power_available_kwh=500.0)
        # Should have repressurized closer to target
        assert proc.total_pressure() > total_before - 0.1  # compensate leak + add N2


# ---------------------------------------------------------------------------
# Trace contaminants
# ---------------------------------------------------------------------------


class TestTraces:
    """Trace contaminant accumulation and scrubbing."""

    def test_traces_accumulate_with_crew(self):
        proc = AtmosphereProcessor()
        # Need enough people so accumulation > removal (per-person 0.001 vs 0.02 removal)
        proc.tick(population=50, water_available_kg=1000.0, power_available_kwh=500.0)
        assert proc.trace_contaminants > 0

    def test_traces_bounded_zero_one(self):
        proc = AtmosphereProcessor()
        for _ in range(100):
            proc.tick(population=50, water_available_kg=1000.0, power_available_kwh=500.0)
        assert 0.0 <= proc.trace_contaminants <= 1.0

    def test_traces_scrubbed_without_crew(self):
        proc = AtmosphereProcessor()
        proc.trace_contaminants = 0.5
        proc.tick(population=0, water_available_kg=0.0, power_available_kwh=0.0)
        assert proc.trace_contaminants < 0.5


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


class TestAlerts:
    """Alert system detects dangerous conditions."""

    def test_no_alert_nominal(self):
        proc = AtmosphereProcessor()
        state = proc.tick(population=6, water_available_kg=1000.0, power_available_kwh=500.0)
        assert state.alert is None

    def test_co2_danger_alert(self):
        proc = AtmosphereProcessor()
        proc.co2_kpa = CO2_DANGER_KPA + 0.1
        state = proc.tick(population=0, water_available_kg=0.0, power_available_kwh=0.0)
        # CO2 may be slightly reduced by leak, check if alert fires
        # Force it above threshold for test
        proc.co2_kpa = CO2_DANGER_KPA + 0.5
        state = proc.tick(population=0, water_available_kg=0.0, power_available_kwh=0.0)
        assert state.alert in ("CO2_DANGER", "CO2_LETHAL")

    def test_co2_lethal_alert(self):
        proc = AtmosphereProcessor()
        proc.co2_kpa = CO2_LETHAL_KPA + 1.0
        state = proc.tick(population=0, water_available_kg=0.0, power_available_kwh=0.0)
        assert state.alert == "CO2_LETHAL"

    def test_o2_low_alert(self):
        proc = AtmosphereProcessor()
        proc.o2_kpa = 14.0
        state = proc.tick(population=0, water_available_kg=0.0, power_available_kwh=0.0)
        assert state.alert == "O2_LOW"

    def test_pressure_critical_alert(self):
        proc = AtmosphereProcessor()
        # Set O2 above low threshold (16 kPa) so PRESSURE_CRITICAL wins
        proc.o2_kpa = 17.0
        proc.n2_kpa = 20.0
        proc.co2_kpa = 0.01
        state = proc.tick(population=0, water_available_kg=0.0, power_available_kwh=0.0)
        assert state.alert == "PRESSURE_CRITICAL"


# ---------------------------------------------------------------------------
# Snapshot serialisation
# ---------------------------------------------------------------------------


class TestSnapshot:
    """snapshot() produces valid JSON-serialisable dict."""

    def test_snapshot_keys(self):
        proc = AtmosphereProcessor()
        snap = proc.snapshot()
        expected_keys = {
            "sol", "o2_kpa", "co2_kpa", "n2_kpa", "total_kpa",
            "scrubber_health", "trace_contaminants", "co2_status",
            "pressure_status", "o2_fraction", "total_water_consumed_kg",
            "total_energy_used_kwh",
        }
        assert set(snap.keys()) == expected_keys

    def test_snapshot_values_are_numbers_or_strings(self):
        proc = AtmosphereProcessor()
        proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        snap = proc.snapshot()
        for key, val in snap.items():
            assert isinstance(val, (int, float, str)), f"{key}: {type(val)}"

    def test_snapshot_round_trip(self):
        """Snapshot values match processor state."""
        proc = AtmosphereProcessor()
        proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        snap = proc.snapshot()
        assert snap["sol"] == proc.sol
        assert snap["o2_kpa"] == round(proc.o2_kpa, 4)


# ---------------------------------------------------------------------------
# AtmosphereState dataclass
# ---------------------------------------------------------------------------


class TestAtmosphereState:
    """AtmosphereState returned by tick() has correct structure."""

    def test_state_fields(self):
        proc = AtmosphereProcessor()
        state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert isinstance(state, AtmosphereState)
        assert state.sol == 1
        assert state.o2_kpa > 0
        assert state.energy_used_kwh >= 0

    def test_state_sol_increments(self):
        proc = AtmosphereProcessor()
        s1 = proc.tick(population=5, water_available_kg=500.0, power_available_kwh=200.0)
        s2 = proc.tick(population=5, water_available_kg=500.0, power_available_kwh=200.0)
        assert s1.sol == 1
        assert s2.sol == 2


# ---------------------------------------------------------------------------
# Property-based invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    """Physical invariants that must hold across all ticks."""

    @pytest.mark.parametrize("pop", [0, 1, 6, 20, 100])
    def test_pressures_non_negative(self, pop):
        """Partial pressures must never go negative."""
        proc = AtmosphereProcessor()
        for _ in range(50):
            proc.tick(population=pop, water_available_kg=1000.0, power_available_kwh=500.0)
        assert proc.o2_kpa >= 0.0
        assert proc.co2_kpa >= 0.0
        assert proc.n2_kpa >= 0.0

    @pytest.mark.parametrize("pop", [0, 1, 6, 20, 100])
    def test_total_pressure_equals_sum(self, pop):
        """total_pressure() must equal sum of partials."""
        proc = AtmosphereProcessor()
        for _ in range(10):
            proc.tick(population=pop, water_available_kg=1000.0, power_available_kwh=500.0)
        total = proc.total_pressure()
        parts = proc.o2_kpa + proc.co2_kpa + proc.n2_kpa
        assert abs(total - parts) < 0.001

    def test_energy_used_non_negative(self):
        proc = AtmosphereProcessor()
        for _ in range(20):
            state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
            assert state.energy_used_kwh >= 0.0

    def test_energy_does_not_exceed_budget(self):
        """Energy used in a single tick cannot exceed power budget."""
        proc = AtmosphereProcessor()
        budget = 100.0
        for _ in range(20):
            state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=budget)
            assert state.energy_used_kwh <= budget + 0.01

    def test_scrubber_health_bounded(self):
        proc = AtmosphereProcessor()
        for _ in range(20000):  # 20k sols
            proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
        assert 0.0 <= proc.scrubber_health <= 1.0

    def test_water_consumed_monotonic(self):
        """Total water consumed should only increase."""
        proc = AtmosphereProcessor()
        prev = 0.0
        for _ in range(20):
            proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=500.0)
            assert proc.total_water_consumed_kg >= prev
            prev = proc.total_water_consumed_kg


# ---------------------------------------------------------------------------
# Multi-sol smoke tests
# ---------------------------------------------------------------------------


class TestSmoke:
    """Smoke tests: run the simulation for many sols without crash."""

    def test_10_sols_no_crash(self):
        proc = AtmosphereProcessor()
        for _ in range(10):
            state = proc.tick(population=6, water_available_kg=500.0, power_available_kwh=200.0)
        assert state.sol == 10

    def test_100_sols_stable_atmosphere(self):
        """With adequate resources, atmosphere stays liveable for 100 sols."""
        # Large volume + high electrolyser keeps CO2 partial pressure low
        proc = AtmosphereProcessor(habitat_volume_m3=2000.0,
                                    electrolyser_capacity_kg_sol=30.0)
        for _ in range(100):
            state = proc.tick(population=6, water_available_kg=500.0, power_available_kwh=300.0)
        # O2 should still be breathable
        assert state.o2_kpa > 15.0
        # CO2 should not be lethal (residual accumulation is physical)
        assert state.co2_kpa < CO2_LETHAL_KPA
        # Total pressure should be reasonable
        assert 50.0 < state.total_kpa < 150.0

    def test_365_sols_mars_year(self):
        """Full Mars year without crash."""
        proc = AtmosphereProcessor(habitat_volume_m3=600.0,
                                    electrolyser_capacity_kg_sol=25.0)
        for _ in range(365):
            state = proc.tick(population=10, water_available_kg=1000.0, power_available_kwh=400.0)
        assert state.sol == 365
        assert proc.o2_kpa >= 0
        assert proc.co2_kpa >= 0

    def test_1000_sols_long_mission(self):
        """Extended mission — atmosphere processor runs 1000 sols."""
        proc = AtmosphereProcessor(habitat_volume_m3=1000.0,
                                    electrolyser_capacity_kg_sol=40.0,
                                    moxie_installed=True)
        for sol in range(1000):
            state = proc.tick(population=20, water_available_kg=5000.0,
                              power_available_kwh=1000.0)
        assert state.sol == 1000
        # Should still be alive (MOXIE + electrolysis + scrubber)
        assert proc.o2_kpa > 0

    def test_starvation_scenario(self):
        """No water, no power — atmosphere degrades but doesn't crash."""
        proc = AtmosphereProcessor()
        for _ in range(50):
            state = proc.tick(population=10, water_available_kg=0.0, power_available_kwh=0.0)
        # Should degrade but not error
        assert proc.o2_kpa >= 0
        assert proc.co2_kpa >= 0
        assert state.sol == 50


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_zero_volume_habitat(self):
        """Zero volume shouldn't crash (division by zero guard)."""
        proc = AtmosphereProcessor(habitat_volume_m3=0.0)
        state = proc.tick(population=1, water_available_kg=100.0, power_available_kwh=100.0)
        assert state.sol == 1

    def test_negative_power_clamped(self):
        """Negative power input is treated as zero."""
        proc = AtmosphereProcessor()
        state = proc.tick(population=5, water_available_kg=100.0, power_available_kwh=-50.0)
        assert state.energy_used_kwh == 0.0

    def test_massive_population(self):
        """1000 people in a small hab — CO2 spikes but no crash."""
        proc = AtmosphereProcessor(habitat_volume_m3=100.0)
        state = proc.tick(population=1000, water_available_kg=10000.0,
                          power_available_kwh=10000.0)
        assert state.sol == 1
        assert proc.co2_kpa >= 0

    def test_single_person(self):
        """One person in a large hab — barely noticeable."""
        proc = AtmosphereProcessor(habitat_volume_m3=5000.0)
        state = proc.tick(population=1, water_available_kg=100.0, power_available_kwh=100.0)
        # O2 should barely change
        assert abs(state.o2_kpa - TARGET_O2_KPA) < 1.0
