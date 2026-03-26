"""Tests for ore_smelter.py -- Mars Regolith Metal Extraction.

152 unit tests across 20 sections covering physics, conservation laws,
failure modes, parametrized invariants, edge cases, and smoke tests.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ore_smelter import (
    FARADAY_CONSTANT, STEFAN_BOLTZMANN, FE_MOLAR_MASS_KG, FEO_MOLAR_MASS_KG,
    O2_MOLAR_MASS_KG, FE_ELECTRONS, FEO_GIBBS_KJ_MOL, IRON_OXIDE_FRACTION,
    FE_FRACTION_IN_FEO, DEFAULT_CELL_VOLTAGE_V, MIN_CELL_VOLTAGE_V,
    DEFAULT_CURRENT_A, DEFAULT_CURRENT_EFFICIENCY, CELL_RESISTANCE_OHM,
    MELTING_TEMP_K, MARS_AMBIENT_TEMP_K, ELECTRODE_WEAR_PER_KG_FE,
    ELECTRODE_REPLACEMENT_THRESHOLD, DEFAULT_FEED_RATE_KG_PER_SOL,
    HOURS_PER_SOL, SECONDS_PER_SOL, O2_PER_KG_FE,
    faraday_mass_kg, minimum_voltage, joule_heating_kw, radiation_loss_kw,
    thermal_balance_kw, electrical_power_kw, iron_from_regolith_kg,
    oxygen_byproduct_kg, apply_electrode_wear, energy_kwh,
    OreSmelter, TickResult, tick, run_simulation,
)


class TestPhysicalConstants:
    def test_faraday(self):
        assert abs(FARADAY_CONSTANT - 96485.0) < 1.0

    def test_stefan_boltzmann(self):
        assert abs(STEFAN_BOLTZMANN - 5.670374419e-8) < 1e-15

    def test_fe_molar_mass(self):
        assert abs(FE_MOLAR_MASS_KG - 0.05585) < 0.001

    def test_feo_molar_mass(self):
        assert abs(FEO_MOLAR_MASS_KG - 0.07184) < 0.001

    def test_o2_molar_mass(self):
        assert abs(O2_MOLAR_MASS_KG - 0.032) < 0.001

    def test_fe_electrons(self):
        assert FE_ELECTRONS == 2

    def test_iron_oxide_fraction(self):
        assert abs(IRON_OXIDE_FRACTION - 0.18) < 0.01

    def test_fe_frac_in_feo(self):
        assert abs(FE_FRACTION_IN_FEO - FE_MOLAR_MASS_KG / FEO_MOLAR_MASS_KG) < 0.001

    def test_melting_temp(self):
        assert MELTING_TEMP_K == 1873.0

    def test_min_voltage(self):
        assert abs(MIN_CELL_VOLTAGE_V - 150000.0 / (2 * 96485.0)) < 0.01

    def test_default_above_min(self):
        assert DEFAULT_CELL_VOLTAGE_V > MIN_CELL_VOLTAGE_V

    def test_o2_stoich(self):
        assert abs(O2_PER_KG_FE - (0.5 * 0.032) / 0.05585) < 0.001

    def test_hours_per_sol(self):
        assert abs(HOURS_PER_SOL - 24.66) < 0.01

    def test_seconds_per_sol(self):
        assert abs(SECONDS_PER_SOL - HOURS_PER_SOL * 3600.0) < 1.0


class TestFaradayMass:
    def test_known(self):
        I, t, M, n, eta = 500.0, 88776.0, 0.05585, 2, 0.80
        expected = (I * t * M * eta) / (n * FARADAY_CONSTANT)
        assert abs(faraday_mass_kg(I, t, M, n, eta) - expected) < 1e-6

    def test_zero_current(self):
        assert faraday_mass_kg(0.0, 1000.0) == 0.0

    def test_zero_time(self):
        assert faraday_mass_kg(500.0, 0.0) == 0.0

    def test_higher_current(self):
        assert faraday_mass_kg(500.0, 1000.0) > faraday_mass_kg(100.0, 1000.0)

    def test_efficiency_linear(self):
        m_half = faraday_mass_kg(500.0, 1000.0, efficiency=0.5)
        m_full = faraday_mass_kg(500.0, 1000.0, efficiency=1.0)
        assert abs(m_full / m_half - 2.0) < 0.01

    def test_neg_current_clamped(self):
        assert faraday_mass_kg(-100.0, 1000.0) == 0.0

    def test_eff_clamped_to_one(self):
        m1 = faraday_mass_kg(500.0, 1000.0, efficiency=1.0)
        m2 = faraday_mass_kg(500.0, 1000.0, efficiency=2.0)
        assert abs(m1 - m2) < 1e-10

    def test_one_sol_reasonable(self):
        m = faraday_mass_kg(DEFAULT_CURRENT_A, SECONDS_PER_SOL, efficiency=DEFAULT_CURRENT_EFFICIENCY)
        assert 1.0 < m < 100.0


class TestMinimumVoltage:
    def test_known(self):
        assert abs(minimum_voltage() - 150000.0 / (2 * 96485.0)) < 0.01

    def test_positive(self):
        assert minimum_voltage() > 0.0

    def test_below_practical(self):
        assert minimum_voltage() < DEFAULT_CELL_VOLTAGE_V

    def test_zero_gibbs(self):
        assert minimum_voltage(gibbs_kj_mol=0.0) == 0.0


class TestJouleHeating:
    def test_known(self):
        assert abs(joule_heating_kw(500.0, 0.003) - (500.0**2) * 0.003 / 1000.0) < 1e-6

    def test_zero(self):
        assert joule_heating_kw(0.0) == 0.0

    def test_squared(self):
        assert abs(joule_heating_kw(200.0) / joule_heating_kw(100.0) - 4.0) < 0.01

    def test_non_negative(self):
        assert joule_heating_kw(-100.0) == 0.0


class TestRadiationLoss:
    def test_positive_at_op(self):
        assert radiation_loss_kw(MELTING_TEMP_K) > 0

    def test_zero_at_ambient(self):
        assert abs(radiation_loss_kw(MARS_AMBIENT_TEMP_K, MARS_AMBIENT_TEMP_K)) < 1e-10

    def test_higher_temp(self):
        assert radiation_loss_kw(2000.0) > radiation_loss_kw(1500.0)

    def test_insulation(self):
        assert radiation_loss_kw(MELTING_TEMP_K, insulation=0.15) < radiation_loss_kw(MELTING_TEMP_K, insulation=1.0)

    def test_perfect_insulation(self):
        assert radiation_loss_kw(MELTING_TEMP_K, insulation=0.0) == 0.0

    def test_t4_scaling(self):
        r = radiation_loss_kw(1000.0, ambient_temp_k=0.0, insulation=1.0)
        r2 = radiation_loss_kw(500.0, ambient_temp_k=0.0, insulation=1.0)
        assert abs(r / r2 - 16.0) < 0.5


class TestThermalBalance:
    def test_positive(self):
        assert thermal_balance_kw(5.0, 5.0, 3.0) > 0

    def test_negative(self):
        assert thermal_balance_kw(1.0, 1.0, 10.0) < 0

    def test_zero(self):
        assert abs(thermal_balance_kw(5.0, 0.0, 5.0)) < 1e-10


class TestElectricalPower:
    def test_known(self):
        assert abs(electrical_power_kw(2.0, 500.0) - 1.0) < 1e-6

    def test_zero_v(self):
        assert electrical_power_kw(0.0, 500.0) == 0.0

    def test_zero_i(self):
        assert electrical_power_kw(2.0, 0.0) == 0.0

    def test_neg_clamped(self):
        assert electrical_power_kw(-2.0, 500.0) == 0.0


class TestIronFromRegolith:
    def test_known(self):
        fe = iron_from_regolith_kg(100.0)
        assert abs(fe - 100.0 * IRON_OXIDE_FRACTION * FE_FRACTION_IN_FEO) < 0.01

    def test_zero(self):
        assert iron_from_regolith_kg(0.0) == 0.0

    def test_proportional(self):
        assert abs(iron_from_regolith_kg(100.0) / iron_from_regolith_kg(50.0) - 2.0) < 0.01

    def test_less_than_input(self):
        assert iron_from_regolith_kg(100.0) < 100.0

    def test_neg_clamped(self):
        assert iron_from_regolith_kg(-50.0) == 0.0

    def test_daily_reasonable(self):
        fe = iron_from_regolith_kg(DEFAULT_FEED_RATE_KG_PER_SOL)
        assert 1.0 < fe < 20.0


class TestOxygenByproduct:
    def test_stoich(self):
        assert abs(oxygen_byproduct_kg(1.0) - O2_PER_KG_FE) < 0.001

    def test_zero(self):
        assert oxygen_byproduct_kg(0.0) == 0.0

    def test_proportional(self):
        assert abs(oxygen_byproduct_kg(10.0) / oxygen_byproduct_kg(1.0) - 10.0) < 0.01

    def test_neg_clamped(self):
        assert oxygen_byproduct_kg(-5.0) == 0.0

    def test_less_than_iron(self):
        assert oxygen_byproduct_kg(10.0) < 10.0


class TestElectrodeWear:
    def test_reduces(self):
        assert apply_electrode_wear(1.0, 5.0) < 1.0

    def test_zero_no_wear(self):
        assert apply_electrode_wear(1.0, 0.0) == 1.0

    def test_never_negative(self):
        assert apply_electrode_wear(0.01, 1000.0) >= 0.0

    def test_clamped_to_one(self):
        assert apply_electrode_wear(1.0, 0.0) <= 1.0

    def test_known(self):
        assert abs(apply_electrode_wear(1.0, 5.0) - 0.9995) < 1e-6

    def test_monotonic(self):
        h = 1.0
        for _ in range(50):
            h = apply_electrode_wear(h, 1.0)
        assert 0.0 < h < 1.0


class TestEnergy:
    def test_known(self):
        assert abs(energy_kwh(1.0, HOURS_PER_SOL) - HOURS_PER_SOL) < 0.01

    def test_zero(self):
        assert energy_kwh(0.0) == 0.0

    def test_neg_clamped(self):
        assert energy_kwh(-5.0) == 0.0


class TestState:
    def test_default(self):
        s = OreSmelter()
        assert s.cell_voltage_v == DEFAULT_CELL_VOLTAGE_V
        assert s.electrode_health == 1.0
        assert s.sol == 0

    def test_voltage_clamped(self):
        assert OreSmelter(cell_voltage_v=-1.0).cell_voltage_v == 0.0

    def test_current_clamped(self):
        assert OreSmelter(cell_current_a=-100.0).cell_current_a == 0.0

    def test_eff_clamped(self):
        assert OreSmelter(current_efficiency=1.5).current_efficiency == 1.0

    def test_electrode_clamped(self):
        assert OreSmelter(electrode_health=-0.5).electrode_health == 0.0


class TestSerialisation:
    def test_round_trip(self):
        s = OreSmelter(cell_current_a=300.0, electrode_health=0.75)
        s2 = OreSmelter.from_dict(s.to_dict())
        assert s2.cell_current_a == 300.0
        assert abs(s2.electrode_health - 0.75) < 1e-10

    def test_dict_keys(self):
        data = OreSmelter().to_dict()
        for k in ("cell_voltage_v", "electrode_health", "cumulative_iron_kg", "events"):
            assert k in data

    def test_from_empty(self):
        assert OreSmelter.from_dict({}).cell_current_a == DEFAULT_CURRENT_A

    def test_events_copied(self):
        s = OreSmelter()
        s.events = ["test"]
        data = s.to_dict()
        data["events"].append("extra")
        assert len(s.events) == 1

    def test_cumulative_preserved(self):
        s = OreSmelter()
        s.cumulative_iron_kg = 100.0
        assert OreSmelter.from_dict(s.to_dict()).cumulative_iron_kg == 100.0


class TestTickEngine:
    def test_increments_sol(self):
        s = OreSmelter()
        r = tick(s)
        assert r.sol == 1 and s.sol == 1

    def test_produces_iron(self):
        assert tick(OreSmelter()).iron_kg > 0

    def test_produces_oxygen(self):
        assert tick(OreSmelter()).oxygen_kg > 0

    def test_produces_slag(self):
        assert tick(OreSmelter()).slag_kg > 0

    def test_consumes_power(self):
        assert tick(OreSmelter()).power_consumed_kw > 0

    def test_accumulates_iron(self):
        s = OreSmelter()
        tick(s); tick(s)
        assert s.cumulative_iron_kg > 0

    def test_accumulates_energy(self):
        s = OreSmelter()
        tick(s)
        assert s.cumulative_energy_kwh > 0

    def test_electrode_degrades(self):
        s = OreSmelter()
        tick(s)
        assert s.electrode_health < 1.0

    def test_operational(self):
        assert tick(OreSmelter()).operational is True

    def test_thermal_reported(self):
        assert isinstance(tick(OreSmelter()).thermal_balance_kw, float)


class TestConservation:
    def test_mass_balance(self):
        r = tick(OreSmelter())
        assert r.iron_kg + r.oxygen_kg + r.slag_kg <= r.regolith_consumed_kg + 1e-9

    def test_faraday_limit(self):
        s = OreSmelter()
        r = tick(s)
        fmax = faraday_mass_kg(s.cell_current_a, SECONDS_PER_SOL, efficiency=s.current_efficiency)
        assert r.iron_kg <= fmax + 1e-9

    def test_feed_limit(self):
        s = OreSmelter()
        r = tick(s)
        assert r.iron_kg <= iron_from_regolith_kg(s.regolith_feed_kg_per_sol) + 1e-9

    def test_o2_stoich(self):
        r = tick(OreSmelter())
        assert abs(r.oxygen_kg - r.iron_kg * O2_PER_KG_FE) < 1e-9

    def test_energy_monotonic(self):
        s = OreSmelter()
        prev = 0.0
        for _ in range(50):
            tick(s)
            assert s.cumulative_energy_kwh >= prev
            prev = s.cumulative_energy_kwh

    def test_iron_monotonic(self):
        s = OreSmelter()
        prev = 0.0
        for _ in range(50):
            tick(s)
            assert s.cumulative_iron_kg >= prev
            prev = s.cumulative_iron_kg

    def test_electrode_monotonic(self):
        s = OreSmelter()
        prev = 1.0
        for _ in range(50):
            tick(s)
            assert s.electrode_health <= prev
            prev = s.electrode_health

    def test_non_negative(self):
        s = OreSmelter()
        for _ in range(100):
            r = tick(s)
            assert r.iron_kg >= 0 and r.oxygen_kg >= 0 and r.slag_kg >= 0
            assert r.power_consumed_kw >= 0 and r.energy_consumed_kwh >= 0


class TestSimulation:
    def test_default(self):
        r = run_simulation(365)
        assert len(r) == 365 and r[0].sol == 1 and r[-1].sol == 365

    def test_iron_reasonable(self):
        total = sum(r.iron_kg for r in run_simulation(365))
        assert 100 < total < 10000

    def test_o2_positive(self):
        assert sum(r.oxygen_kg for r in run_simulation(365)) > 0

    def test_electrode_degrades_year(self):
        assert run_simulation(365)[-1].electrode_health < 1.0

    def test_higher_current(self):
        low = sum(r.iron_kg for r in run_simulation(10, cell_current_a=200.0))
        high = sum(r.iron_kg for r in run_simulation(10, cell_current_a=800.0))
        assert high > low

    def test_short(self):
        assert len(run_simulation(1)) == 1

    def test_zero(self):
        assert len(run_simulation(0)) == 0


class TestFailureModes:
    def test_cold_furnace(self):
        r = tick(OreSmelter(furnace_temp_k=300.0))
        assert not r.operational and r.iron_kg == 0.0
        assert "FURNACE COLD" in " ".join(r.events)

    def test_dead_electrode(self):
        r = tick(OreSmelter(electrode_health=0.05))
        assert not r.operational and r.iron_kg == 0.0
        assert "ELECTRODE SPENT" in " ".join(r.events)

    def test_undervoltage(self):
        r = tick(OreSmelter(cell_voltage_v=0.1))
        assert not r.operational
        assert "UNDERVOLTAGE" in " ".join(r.events)

    def test_zero_feed(self):
        r = tick(OreSmelter(regolith_feed_kg_per_sol=0.0))
        assert r.iron_kg == 0.0 and r.oxygen_kg == 0.0

    def test_zero_current(self):
        assert tick(OreSmelter(cell_current_a=0.0)).iron_kg == 0.0


class TestParametrized:
    @pytest.mark.parametrize("current", [100, 200, 500, 800, 1000])
    def test_faraday_proportional(self, current):
        m = faraday_mass_kg(float(current), 1000.0)
        m_ref = faraday_mass_kg(500.0, 1000.0)
        assert abs(m / m_ref - current / 500.0) < 0.01

    @pytest.mark.parametrize("feed", [10, 25, 50, 100, 200])
    def test_iron_from_feed(self, feed):
        fe = iron_from_regolith_kg(float(feed))
        assert 0 < fe < feed

    @pytest.mark.parametrize("iron", [0.1, 1.0, 5.0, 10.0, 50.0])
    def test_o2_proportional(self, iron):
        assert abs(oxygen_byproduct_kg(float(iron)) - iron * O2_PER_KG_FE) < 0.001

    @pytest.mark.parametrize("voltage", [0.5, 1.0, 1.5, 2.0, 3.0])
    def test_power_proportional(self, voltage):
        assert abs(electrical_power_kw(voltage, 500.0) / electrical_power_kw(2.0, 500.0) - voltage / 2.0) < 0.01

    @pytest.mark.parametrize("temp", [500, 1000, 1500, 1873, 2500])
    def test_radiation(self, temp):
        assert radiation_loss_kw(float(temp)) >= 0.0

    @pytest.mark.parametrize("health", [0.0, 0.1, 0.5, 0.75, 1.0])
    def test_electrode_bounded(self, health):
        new_h = apply_electrode_wear(health, 1.0)
        assert 0.0 <= new_h <= 1.0 and new_h <= health

    @pytest.mark.parametrize("sols", [1, 10, 50, 100, 365])
    def test_sim_length(self, sols):
        assert len(run_simulation(sols)) == sols


class TestEdgeCases:
    def test_max_efficiency(self):
        assert tick(OreSmelter(current_efficiency=1.0)).iron_kg > 0

    def test_min_efficiency(self):
        assert tick(OreSmelter(current_efficiency=0.0)).iron_kg == 0.0

    def test_high_current_capped(self):
        s = OreSmelter(cell_current_a=10000.0)
        assert tick(s).iron_kg <= iron_from_regolith_kg(s.regolith_feed_kg_per_sol) + 1e-9

    def test_low_feed(self):
        assert tick(OreSmelter(regolith_feed_kg_per_sol=0.1)).iron_kg <= iron_from_regolith_kg(0.1) + 1e-9

    def test_electrode_at_threshold(self):
        assert not tick(OreSmelter(electrode_health=ELECTRODE_REPLACEMENT_THRESHOLD)).operational

    def test_voltage_at_min(self):
        assert tick(OreSmelter(cell_voltage_v=MIN_CELL_VOLTAGE_V)).operational

    def test_round_trip_after_tick(self):
        s = OreSmelter()
        tick(s)
        s2 = OreSmelter.from_dict(s.to_dict())
        assert s2.sol == 1 and s2.cumulative_iron_kg > 0


class TestSmokeTest:
    def test_10_sol(self):
        results = run_simulation(10)
        assert len(results) == 10
        for r in results:
            assert r.sol > 0 and r.iron_kg >= 0 and 0.0 <= r.electrode_health <= 1.0

    def test_mars_year(self):
        results = run_simulation(668)
        assert len(results) == 668
        assert sum(r.iron_kg for r in results) > 0

    def test_electrode_survives_year(self):
        results = run_simulation(668)
        assert results[-1].electrode_health > 0 and results[-1].operational

    def test_realistic_daily(self):
        r = tick(OreSmelter())
        assert 1.0 < r.iron_kg < 30.0

    def test_o2_meaningful(self):
        assert tick(OreSmelter()).oxygen_kg / 0.84 > 0.1

    def test_mass_balance_sim(self):
        results = run_simulation(100)
        total_out = sum(r.iron_kg + r.oxygen_kg + r.slag_kg for r in results)
        total_in = sum(r.regolith_consumed_kg for r in results)
        assert total_out <= total_in + 1e-6
