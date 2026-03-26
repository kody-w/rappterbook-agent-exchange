"""Tests for src/atmo_recycler.py — Mars habitat atmosphere management."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.atmo_recycler import (
    Atmosphere,
    LifeSupport,
    AtmoTickResult,
    o2_demand,
    co2_production,
    water_for_electrolysis,
    electrolysis_power,
    scrubber_power,
    sabatier_water_recovery,
    co2_alert_level,
    pressure_to_mass,
    apply_leakage,
    tick_atmosphere,
    O2_KG_PER_PERSON_SOL,
    CO2_KG_PER_PERSON_SOL,
    H2O_KG_PER_KG_O2,
    ELECTROLYSIS_KWH_PER_KG_O2,
    ELECTROLYSIS_EFFICIENCY,
    SCRUBBER_KWH_PER_KG_CO2,
    SCRUBBER_CAPACITY_KG_SOL,
    SABATIER_H2O_RECOVERY,
    TARGET_O2_KPA,
    TARGET_CO2_KPA,
    TOTAL_PRESSURE_KPA,
    CO2_WARNING_KPA,
    CO2_DANGER_KPA,
    CO2_LETHAL_KPA,
    LEAK_RATE_KPA_SOL,
    AIRLOCK_LOSS_KPA,
    TRACE_CONTAMINANT_KG_PERSON_SOL,
    TRACE_FILTER_EFFICIENCY,
)


# ===================================================================
# O2 demand
# ===================================================================

class TestO2Demand:
    def test_zero_population(self) -> None:
        assert o2_demand(0) == 0.0

    def test_one_person(self) -> None:
        assert o2_demand(1) == pytest.approx(O2_KG_PER_PERSON_SOL)

    def test_scales_linearly(self) -> None:
        assert o2_demand(10) == pytest.approx(10 * O2_KG_PER_PERSON_SOL)

    def test_large_crew(self) -> None:
        demand = o2_demand(200)
        assert demand == pytest.approx(200 * O2_KG_PER_PERSON_SOL)
        assert demand > 0


# ===================================================================
# CO2 production
# ===================================================================

class TestCO2Production:
    def test_zero_population(self) -> None:
        assert co2_production(0) == 0.0

    def test_one_person(self) -> None:
        assert co2_production(1) == pytest.approx(CO2_KG_PER_PERSON_SOL)

    def test_co2_exceeds_o2(self) -> None:
        """Humans exhale more CO2 mass than O2 consumed (carbon added)."""
        assert CO2_KG_PER_PERSON_SOL > O2_KG_PER_PERSON_SOL

    def test_scales_linearly(self) -> None:
        assert co2_production(50) == pytest.approx(50 * CO2_KG_PER_PERSON_SOL)


# ===================================================================
# Water for electrolysis
# ===================================================================

class TestWaterForElectrolysis:
    def test_zero_o2(self) -> None:
        assert water_for_electrolysis(0.0) == 0.0

    def test_stoichiometry(self) -> None:
        """2H2O → 2H2 + O2. Molar: 36g H2O per 32g O2 = 1.125 ratio."""
        assert water_for_electrolysis(1.0) == pytest.approx(H2O_KG_PER_KG_O2)

    def test_scales_linearly(self) -> None:
        w1 = water_for_electrolysis(1.0)
        w5 = water_for_electrolysis(5.0)
        assert w5 == pytest.approx(5.0 * w1)


# ===================================================================
# Electrolysis power
# ===================================================================

class TestElectrolysisPower:
    def test_zero_o2(self) -> None:
        assert electrolysis_power(0.0) == 0.0

    def test_positive_power(self) -> None:
        p = electrolysis_power(1.0)
        assert p > 0.0

    def test_efficiency_increases_power(self) -> None:
        """Lower efficiency means more power needed."""
        p = electrolysis_power(1.0)
        ideal = ELECTROLYSIS_KWH_PER_KG_O2
        assert p > ideal  # efficiency < 1 → actual > ideal

    def test_scales_linearly(self) -> None:
        p1 = electrolysis_power(1.0)
        p10 = electrolysis_power(10.0)
        assert p10 == pytest.approx(10.0 * p1)


# ===================================================================
# Scrubber power
# ===================================================================

class TestScrubberPower:
    def test_zero_co2(self) -> None:
        assert scrubber_power(0.0) == 0.0

    def test_positive(self) -> None:
        assert scrubber_power(1.0) == pytest.approx(SCRUBBER_KWH_PER_KG_CO2)

    def test_scales_linearly(self) -> None:
        assert scrubber_power(3.0) == pytest.approx(3.0 * SCRUBBER_KWH_PER_KG_CO2)


# ===================================================================
# Sabatier water recovery
# ===================================================================

class TestSabatierWaterRecovery:
    def test_zero_co2(self) -> None:
        assert sabatier_water_recovery(0.0) == 0.0

    def test_positive_recovery(self) -> None:
        r = sabatier_water_recovery(1.0)
        assert r > 0.0

    def test_recovery_less_than_input_water(self) -> None:
        """Can't recover more water than electrolysis consumed."""
        o2_for_one_person = O2_KG_PER_PERSON_SOL
        water_consumed = water_for_electrolysis(o2_for_one_person)
        co2_made = CO2_KG_PER_PERSON_SOL
        water_recovered = sabatier_water_recovery(co2_made)
        assert water_recovered < water_consumed


# ===================================================================
# CO2 alert level
# ===================================================================

class TestCO2AlertLevel:
    def test_nominal(self) -> None:
        assert co2_alert_level(0.04) == "nominal"

    def test_just_below_warning(self) -> None:
        assert co2_alert_level(CO2_WARNING_KPA - 0.01) == "nominal"

    def test_warning(self) -> None:
        assert co2_alert_level(CO2_WARNING_KPA) == "warning"

    def test_danger(self) -> None:
        assert co2_alert_level(CO2_DANGER_KPA) == "danger"

    def test_lethal(self) -> None:
        assert co2_alert_level(CO2_LETHAL_KPA) == "lethal"

    def test_above_lethal(self) -> None:
        assert co2_alert_level(50.0) == "lethal"

    def test_zero(self) -> None:
        assert co2_alert_level(0.0) == "nominal"


# ===================================================================
# Pressure to mass (ideal gas law)
# ===================================================================

class TestPressureToMass:
    def test_zero_pressure(self) -> None:
        assert pressure_to_mass(0.0, 500.0, 32.0) == 0.0

    def test_zero_volume(self) -> None:
        assert pressure_to_mass(21.0, 0.0, 32.0) == 0.0

    def test_positive(self) -> None:
        m = pressure_to_mass(21.3, 500.0, 32.0)
        assert m > 0.0

    def test_heavier_gas_more_mass(self) -> None:
        """Same partial pressure → heavier gas = more mass."""
        m_o2 = pressure_to_mass(21.0, 500.0, 32.0)  # O2
        m_co2 = pressure_to_mass(21.0, 500.0, 44.0)  # CO2
        assert m_co2 > m_o2

    def test_scales_with_volume(self) -> None:
        m1 = pressure_to_mass(21.0, 100.0, 32.0)
        m5 = pressure_to_mass(21.0, 500.0, 32.0)
        assert m5 == pytest.approx(5.0 * m1)

    def test_scales_with_pressure(self) -> None:
        m1 = pressure_to_mass(10.0, 500.0, 32.0)
        m2 = pressure_to_mass(20.0, 500.0, 32.0)
        assert m2 == pytest.approx(2.0 * m1)

    def test_earth_o2_mass_sanity(self) -> None:
        """500 m³ at 21.3 kPa O2 should hold ~140 kg O2 (sanity check)."""
        m = pressure_to_mass(21.3, 500.0, 32.0)
        assert 100.0 < m < 200.0


# ===================================================================
# Atmosphere
# ===================================================================

class TestAtmosphere:
    def test_default_pressure(self) -> None:
        a = Atmosphere()
        assert a.total_pressure() == pytest.approx(TOTAL_PRESSURE_KPA, abs=0.1)

    def test_o2_fraction(self) -> None:
        a = Atmosphere()
        frac = a.o2_fraction()
        assert 0.19 < frac < 0.23  # ~21%

    def test_co2_ppm_default(self) -> None:
        a = Atmosphere()
        ppm = a.co2_ppm()
        assert 300 < ppm < 600  # ~400 ppm nominal

    def test_zero_pressure_safety(self) -> None:
        a = Atmosphere(o2_kpa=0, co2_kpa=0, n2_kpa=0)
        assert a.o2_fraction() == 0.0
        assert a.co2_ppm() == 0.0

    def test_custom_volume(self) -> None:
        a = Atmosphere(volume_m3=1000.0)
        assert a.volume_m3 == 1000.0


# ===================================================================
# LifeSupport
# ===================================================================

class TestLifeSupport:
    def test_defaults(self) -> None:
        ls = LifeSupport()
        assert ls.scrubber_units == 2
        assert ls.electrolyzer_capacity_kg_sol == 10.0
        assert ls.sabatier_active is True

    def test_clamp_negative_scrubbers(self) -> None:
        ls = LifeSupport(scrubber_units=-1)
        assert ls.scrubber_units == 0

    def test_clamp_health(self) -> None:
        ls = LifeSupport(scrubber_health=1.5, electrolyzer_health=-0.1)
        assert ls.scrubber_health == 1.0
        assert ls.electrolyzer_health == 0.0


# ===================================================================
# Apply leakage
# ===================================================================

class TestApplyLeakage:
    def test_baseline_leak(self) -> None:
        a = Atmosphere()
        before = a.total_pressure()
        lost = apply_leakage(a)
        assert lost == pytest.approx(LEAK_RATE_KPA_SOL, abs=0.001)
        assert a.total_pressure() < before

    def test_airlock_adds_leak(self) -> None:
        a1 = Atmosphere()
        a2 = Atmosphere()
        lost1 = apply_leakage(a1, airlock_cycles=0)
        lost2 = apply_leakage(a2, airlock_cycles=5)
        assert lost2 > lost1

    def test_proportional_gas_loss(self) -> None:
        """All gases lose the same fraction."""
        a = Atmosphere()
        o2_before = a.o2_kpa
        n2_before = a.n2_kpa
        apply_leakage(a)
        o2_frac = a.o2_kpa / o2_before
        n2_frac = a.n2_kpa / n2_before
        assert o2_frac == pytest.approx(n2_frac, abs=1e-6)

    def test_zero_pressure_no_crash(self) -> None:
        a = Atmosphere(o2_kpa=0, co2_kpa=0, n2_kpa=0)
        lost = apply_leakage(a)
        assert lost == 0.0

    def test_leak_capped_at_10_percent(self) -> None:
        """Even extreme airlock use can't lose >10% per sol."""
        a = Atmosphere()
        total = a.total_pressure()
        lost = apply_leakage(a, airlock_cycles=100)
        assert lost <= total * 0.1 + 0.001


# ===================================================================
# tick_atmosphere — integration tests
# ===================================================================

class TestTickAtmosphere:
    """Full integration tests for one-sol atmosphere tick."""

    def _standard_tick(self, population: int = 6) -> tuple:
        """Run one standard tick with Earth-like conditions."""
        atmo = Atmosphere()
        ls = LifeSupport()
        result = tick_atmosphere(
            atmo, ls,
            population=population,
            power_available_kwh=100.0,
            water_available_kg=50.0,
        )
        return atmo, result

    def test_nominal_6_crew(self) -> None:
        atmo, result = self._standard_tick(6)
        assert result.o2_consumed_kg > 0
        assert result.co2_produced_kg > 0
        assert result.o2_produced_kg > 0
        assert result.co2_scrubbed_kg > 0
        assert result.power_consumed_kwh > 0
        assert result.water_consumed_kg > 0

    def test_co2_alert_nominal(self) -> None:
        _, result = self._standard_tick(6)
        assert result.co2_alert == "nominal"

    def test_pressure_stays_near_target(self) -> None:
        """After one sol with working life support, pressure is ~101 kPa."""
        atmo, _ = self._standard_tick(6)
        assert 99.0 < atmo.total_pressure() < 103.0

    def test_o2_stays_near_target(self) -> None:
        """O2 partial pressure stays near 21 kPa with working systems."""
        atmo, _ = self._standard_tick(6)
        assert 18.0 < atmo.o2_kpa < 25.0

    # --- Conservation laws ---

    def test_power_conservation(self) -> None:
        """Power consumed ≤ power available."""
        _, result = self._standard_tick(6)
        assert result.power_consumed_kwh <= 100.0 + 0.001

    def test_water_conservation(self) -> None:
        """Water consumed ≤ water available."""
        _, result = self._standard_tick(6)
        assert result.water_consumed_kg <= 50.0 + 0.001

    def test_o2_non_negative(self) -> None:
        """O2 partial pressure never goes negative."""
        atmo, _ = self._standard_tick(200)  # huge crew
        assert atmo.o2_kpa >= 0.0

    def test_co2_non_negative(self) -> None:
        """CO2 partial pressure never goes negative."""
        atmo, _ = self._standard_tick(6)
        assert atmo.co2_kpa >= 0.0

    # --- Edge cases ---

    def test_zero_population(self) -> None:
        atmo = Atmosphere()
        ls = LifeSupport()
        result = tick_atmosphere(atmo, ls, population=0,
                                 power_available_kwh=100.0,
                                 water_available_kg=50.0)
        assert result.o2_consumed_kg == 0.0
        assert result.co2_produced_kg == 0.0
        assert result.o2_produced_kg == 0.0

    def test_zero_power(self) -> None:
        """No power → no scrubbing, no electrolysis, CO2 rises."""
        atmo = Atmosphere()
        ls = LifeSupport()
        result = tick_atmosphere(atmo, ls, population=6,
                                 power_available_kwh=0.0,
                                 water_available_kg=50.0)
        assert result.co2_scrubbed_kg == 0.0
        assert result.o2_produced_kg == 0.0
        assert result.power_consumed_kwh == 0.0

    def test_zero_water(self) -> None:
        """No water → no electrolysis, O2 drops."""
        atmo = Atmosphere()
        ls = LifeSupport()
        result = tick_atmosphere(atmo, ls, population=6,
                                 power_available_kwh=100.0,
                                 water_available_kg=0.0)
        assert result.o2_produced_kg == 0.0
        assert result.water_consumed_kg == 0.0
        # Scrubbing still works (doesn't need water)
        assert result.co2_scrubbed_kg > 0.0

    def test_broken_scrubbers(self) -> None:
        """Health=0 scrubbers → no CO2 removal."""
        atmo = Atmosphere()
        ls = LifeSupport(scrubber_health=0.0)
        result = tick_atmosphere(atmo, ls, population=6,
                                 power_available_kwh=100.0,
                                 water_available_kg=50.0)
        assert result.co2_scrubbed_kg == 0.0

    def test_broken_electrolyzer(self) -> None:
        """Health=0 electrolyzer → no O2 production."""
        atmo = Atmosphere()
        ls = LifeSupport(electrolyzer_health=0.0)
        result = tick_atmosphere(atmo, ls, population=6,
                                 power_available_kwh=100.0,
                                 water_available_kg=50.0)
        assert result.o2_produced_kg == 0.0

    def test_no_sabatier(self) -> None:
        """Sabatier off → no water recovery."""
        atmo = Atmosphere()
        ls = LifeSupport(sabatier_active=False)
        result = tick_atmosphere(atmo, ls, population=6,
                                 power_available_kwh=100.0,
                                 water_available_kg=50.0)
        assert result.water_recovered_kg == 0.0

    def test_sabatier_recovers_water(self) -> None:
        """Sabatier on → some water recovered."""
        _, result = self._standard_tick(6)
        assert result.water_recovered_kg > 0.0

    def test_trace_contaminants_filtered(self) -> None:
        """Trace contaminants get partially removed each sol."""
        _, result = self._standard_tick(6)
        assert result.trace_removed_kg > 0.0

    def test_airlock_increases_pressure_loss(self) -> None:
        atmo1 = Atmosphere()
        ls = LifeSupport()
        r1 = tick_atmosphere(atmo1, ls, population=6,
                              power_available_kwh=100.0,
                              water_available_kg=50.0,
                              airlock_cycles=0)
        atmo2 = Atmosphere()
        r2 = tick_atmosphere(atmo2, ls, population=6,
                              power_available_kwh=100.0,
                              water_available_kg=50.0,
                              airlock_cycles=10)
        assert r2.pressure_lost_kpa > r1.pressure_lost_kpa

    # --- Multi-sol stability tests ---

    def test_10_sol_stability(self) -> None:
        """Run 10 sols with 6 crew — atmosphere stays breathable."""
        atmo = Atmosphere()
        ls = LifeSupport()
        for _ in range(10):
            tick_atmosphere(atmo, ls, population=6,
                           power_available_kwh=100.0,
                           water_available_kg=50.0)
        assert atmo.o2_kpa > 15.0, "O2 dropped too low"
        assert atmo.co2_kpa < CO2_WARNING_KPA, "CO2 rose to warning"
        assert atmo.total_pressure() > 90.0, "pressure dropped too far"

    def test_100_sol_stability(self) -> None:
        """100 sols with adequate resources — system stays stable."""
        atmo = Atmosphere()
        ls = LifeSupport()
        for _ in range(100):
            tick_atmosphere(atmo, ls, population=6,
                           power_available_kwh=100.0,
                           water_available_kg=50.0)
        assert atmo.o2_kpa > 10.0
        assert atmo.co2_kpa < CO2_DANGER_KPA
        assert atmo.total_pressure() > 80.0

    def test_co2_rises_without_power(self) -> None:
        """Without power, CO2 accumulates over multiple sols."""
        atmo = Atmosphere()
        ls = LifeSupport()
        initial_co2 = atmo.co2_kpa
        for _ in range(10):
            tick_atmosphere(atmo, ls, population=6,
                           power_available_kwh=0.0,
                           water_available_kg=0.0)
        assert atmo.co2_kpa > initial_co2 * 5, "CO2 should rise sharply"

    def test_o2_drops_without_water(self) -> None:
        """Without water for electrolysis, O2 depletes."""
        atmo = Atmosphere()
        ls = LifeSupport()
        initial_o2 = atmo.o2_kpa
        for _ in range(10):
            tick_atmosphere(atmo, ls, population=6,
                           power_available_kwh=100.0,
                           water_available_kg=0.0)
        assert atmo.o2_kpa < initial_o2, "O2 should decrease"

    # --- Physical bounds (property-based invariants) ---

    def test_all_results_non_negative(self) -> None:
        """Every field in AtmoTickResult must be ≥ 0."""
        _, result = self._standard_tick(6)
        assert result.o2_produced_kg >= 0.0
        assert result.o2_consumed_kg >= 0.0
        assert result.co2_produced_kg >= 0.0
        assert result.co2_scrubbed_kg >= 0.0
        assert result.water_consumed_kg >= 0.0
        assert result.water_recovered_kg >= 0.0
        assert result.power_consumed_kwh >= 0.0
        assert result.pressure_lost_kpa >= 0.0
        assert result.trace_removed_kg >= 0.0

    def test_mass_balance_o2(self) -> None:
        """O2 produced + O2 in atmosphere should account for O2 consumed."""
        atmo_before = Atmosphere()
        o2_before = atmo_before.o2_kpa
        atmo = Atmosphere()
        ls = LifeSupport()
        result = tick_atmosphere(atmo, ls, population=6,
                                 power_available_kwh=100.0,
                                 water_available_kg=50.0)
        # O2 change = produced - consumed (in kPa terms, approximately)
        # This is a qualitative check: production should offset consumption
        if result.o2_produced_kg >= result.o2_consumed_kg:
            # Production >= consumption → O2 should not drop much
            assert atmo.o2_kpa >= o2_before * 0.9

    def test_water_recovery_less_than_consumed(self) -> None:
        """Sabatier can't recover more water than electrolysis consumed."""
        _, result = self._standard_tick(6)
        assert result.water_recovered_kg <= result.water_consumed_kg

    # --- Extreme scenarios ---

    def test_huge_crew_stress(self) -> None:
        """500 crew in small habitat — system degrades but doesn't crash."""
        atmo = Atmosphere(volume_m3=200.0)
        ls = LifeSupport(scrubber_units=5, electrolyzer_capacity_kg_sol=50.0)
        result = tick_atmosphere(atmo, ls, population=500,
                                 power_available_kwh=1000.0,
                                 water_available_kg=500.0)
        assert atmo.o2_kpa >= 0.0
        assert atmo.co2_kpa >= 0.0
        assert result.power_consumed_kwh <= 1000.0 + 0.001

    def test_tiny_habitat(self) -> None:
        """10 m³ habitat (phone booth) — physics still holds."""
        atmo = Atmosphere(volume_m3=10.0)
        ls = LifeSupport()
        result = tick_atmosphere(atmo, ls, population=1,
                                 power_available_kwh=50.0,
                                 water_available_kg=20.0)
        assert atmo.o2_kpa >= 0.0
        assert result.power_consumed_kwh >= 0.0

    def test_massive_habitat(self) -> None:
        """100,000 m³ dome — scales without overflow."""
        atmo = Atmosphere(volume_m3=100_000.0)
        ls = LifeSupport(scrubber_units=20, electrolyzer_capacity_kg_sol=100.0)
        result = tick_atmosphere(atmo, ls, population=100,
                                 power_available_kwh=5000.0,
                                 water_available_kg=1000.0)
        assert atmo.o2_kpa > 0.0
        assert result.co2_alert == "nominal"


# ===================================================================
# Smoke test: 365-sol simulation
# ===================================================================

class TestSmoke:
    def test_365_sol_no_crash(self) -> None:
        """Full Mars year with 6 crew. Must not crash. Atmosphere must be breathable."""
        atmo = Atmosphere()
        ls = LifeSupport()
        for sol in range(365):
            result = tick_atmosphere(
                atmo, ls,
                population=6,
                power_available_kwh=100.0,
                water_available_kg=50.0,
                airlock_cycles=2,
            )
            # Every single sol: physics hold
            assert atmo.o2_kpa >= 0.0
            assert atmo.co2_kpa >= 0.0
            assert atmo.total_pressure() >= 0.0
            assert result.power_consumed_kwh <= 100.0 + 0.001
            assert result.water_consumed_kg <= 50.0 + 0.001

        # After a full year: still breathable
        assert atmo.o2_kpa > 5.0, f"O2 too low after 365 sols: {atmo.o2_kpa}"
        assert atmo.co2_kpa < CO2_LETHAL_KPA, f"CO2 lethal after 365 sols: {atmo.co2_kpa}"

    def test_10_sol_smoke(self) -> None:
        """Quick 10-sol smoke test."""
        atmo = Atmosphere()
        ls = LifeSupport()
        for _ in range(10):
            tick_atmosphere(atmo, ls, population=6,
                           power_available_kwh=100.0,
                           water_available_kg=50.0)
        assert atmo.o2_kpa > 0.0
        assert atmo.total_pressure() > 0.0
