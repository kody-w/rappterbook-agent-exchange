"""Tests for plasma_forge.py -- Mars Plasma Arc Metal Refinery.

130 tests covering arc physics, thermodynamics, carbothermic reduction,
electrode wear, mass conservation, energy conservation, metal yields,
tapping, fault handling, full-batch simulation, physical bounds, and
smoke tests.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import plasma_forge as pf


@pytest.fixture
def default_forge():
    return pf.create_forge()

@pytest.fixture
def small_forge():
    return pf.create_forge(charge_kg=50.0, arc_current_a=300.0)

@pytest.fixture
def no_charge_forge():
    return pf.create_forge(charge_kg=0.0)

@pytest.fixture
def no_electrode_forge():
    f = pf.create_forge()
    f.electrode_mass_kg = 0.0
    return f


class TestRegolithComposition:
    def test_fractions_sum_to_one(self):
        total = (pf.REGOLITH_FE2O3_FRAC + pf.REGOLITH_AL2O3_FRAC +
                 pf.REGOLITH_TIO2_FRAC + pf.REGOLITH_SIO2_FRAC +
                 pf.REGOLITH_OTHER_FRAC)
        assert abs(total - 1.0) < 1e-10

    def test_iron_oxide_dominant_over_alumina(self):
        assert pf.REGOLITH_FE2O3_FRAC > pf.REGOLITH_AL2O3_FRAC

    def test_silica_largest(self):
        assert pf.REGOLITH_SIO2_FRAC > pf.REGOLITH_FE2O3_FRAC

    def test_metal_yields_positive(self):
        for v in [pf.MAX_FE_PER_KG, pf.MAX_AL_PER_KG, pf.MAX_TI_PER_KG, pf.MAX_SI_PER_KG]:
            assert v > 0

    def test_metal_yields_less_than_oxide(self):
        assert pf.MAX_FE_PER_KG < pf.REGOLITH_FE2O3_FRAC
        assert pf.MAX_AL_PER_KG < pf.REGOLITH_AL2O3_FRAC
        assert pf.MAX_TI_PER_KG < pf.REGOLITH_TIO2_FRAC
        assert pf.MAX_SI_PER_KG < pf.REGOLITH_SIO2_FRAC

    def test_total_max_metal_less_than_one(self):
        total = pf.MAX_FE_PER_KG + pf.MAX_AL_PER_KG + pf.MAX_TI_PER_KG + pf.MAX_SI_PER_KG
        assert total < 1.0

    def test_fe_yield_ratio_physical(self):
        assert 0.65 < pf.FE_YIELD_FROM_OXIDE < 0.75

    def test_al_yield_ratio_physical(self):
        assert 0.50 < pf.AL_YIELD_FROM_OXIDE < 0.56

    def test_si_yield_ratio_physical(self):
        assert 0.44 < pf.SI_YIELD_FROM_OXIDE < 0.50


class TestArcVoltage:
    def test_positive_valid(self):
        assert pf.arc_voltage(800.0, 5.0) > 0.0

    def test_zero_zero_current(self):
        assert pf.arc_voltage(0.0, 5.0) == 0.0

    def test_zero_negative_current(self):
        assert pf.arc_voltage(-10.0, 5.0) == 0.0

    def test_zero_zero_length(self):
        assert pf.arc_voltage(800.0, 0.0) == 0.0

    def test_increases_with_arc_length(self):
        assert pf.arc_voltage(800.0, 10.0) > pf.arc_voltage(800.0, 5.0)

    def test_ayrton_manual(self):
        I, L = 800.0, 5.0
        expected = 200.0 + 10.0 * 5.0 + (20.0 + 10.0 * 5.0) / 800.0
        assert abs(pf.arc_voltage(I, L) - expected) < 1e-10


class TestArcPower:
    def test_positive_valid(self):
        assert pf.arc_power_kw(800.0, 5.0) > 0.0

    def test_zero_zero_current(self):
        assert pf.arc_power_kw(0.0, 5.0) == 0.0

    def test_scales_with_current(self):
        p1 = pf.arc_power_kw(500.0, 5.0)
        p2 = pf.arc_power_kw(1000.0, 5.0)
        assert p2 > p1 * 1.5

    def test_p_equals_vi(self):
        I, L = 800.0, 5.0
        v = pf.arc_voltage(I, L)
        assert abs(pf.arc_power_kw(I, L) - v * I / 1000.0) < 1e-10

    def test_power_in_reasonable_range(self):
        p = pf.arc_power_kw(800.0, 5.0)
        assert 100.0 < p < 500.0


class TestRadiationLoss:
    def test_zero_at_ambient(self):
        assert pf.radiation_loss_kw(pf.MARS_AMBIENT_TEMP_K) == 0.0

    def test_zero_below_ambient(self):
        assert pf.radiation_loss_kw(100.0) == 0.0

    def test_positive_above_ambient(self):
        assert pf.radiation_loss_kw(1500.0) > 0.0

    def test_increases_with_temperature(self):
        assert pf.radiation_loss_kw(2000.0) > pf.radiation_loss_kw(1000.0)

    def test_t4_scaling(self):
        r1 = pf.radiation_loss_kw(1000.0)
        r2 = pf.radiation_loss_kw(2000.0)
        assert r2 / r1 > 10.0


class TestConductionLoss:
    def test_zero_at_ambient(self):
        assert pf.conduction_loss_kw(pf.MARS_AMBIENT_TEMP_K) == 0.0

    def test_positive_above_ambient(self):
        assert pf.conduction_loss_kw(1500.0) > 0.0

    def test_linear(self):
        c1 = pf.conduction_loss_kw(1000.0)
        c2 = pf.conduction_loss_kw(1790.0)
        expected = (1790.0 - 210.0) / (1000.0 - 210.0)
        assert abs(c2 / c1 - expected) < 0.01


class TestTotalLoss:
    def test_sum(self):
        t = 1500.0
        assert abs(pf.total_loss_kw(t) - (pf.radiation_loss_kw(t) + pf.conduction_loss_kw(t))) < 1e-10


class TestHeatingEnergy:
    def test_positive_rise(self):
        assert pf.heating_energy_kj(100.0, 300.0, 800.0) > 0.0

    def test_zero_zero_mass(self):
        assert pf.heating_energy_kj(0.0, 300.0, 800.0) == 0.0

    def test_zero_no_rise(self):
        assert pf.heating_energy_kj(100.0, 500.0, 500.0) == 0.0

    def test_zero_negative_delta(self):
        assert pf.heating_energy_kj(100.0, 800.0, 300.0) == 0.0

    def test_includes_latent(self):
        e_below = pf.heating_energy_kj(100.0, 300.0, 1300.0)
        e_cross = pf.heating_energy_kj(100.0, 300.0, 1500.0)
        latent = 100.0 * pf.LATENT_HEAT_MELTING_KJ_KG
        assert e_cross > e_below + latent * 0.8

    def test_scales_with_mass(self):
        e1 = pf.heating_energy_kj(50.0, 300.0, 800.0)
        e2 = pf.heating_energy_kj(100.0, 300.0, 800.0)
        assert abs(e2 / e1 - 2.0) < 0.001


class TestReductionRate:
    def test_zero_below(self):
        assert pf.reduction_rate(1000.0) == 0.0

    def test_positive_above(self):
        assert pf.reduction_rate(1500.0) > 0.0

    def test_increases(self):
        assert pf.reduction_rate(2000.0) > pf.reduction_rate(1500.0)

    def test_approaches_prefactor(self):
        assert abs(pf.reduction_rate(5000.0) - pf.ARRHENIUS_PREFACTOR) < 0.005

    def test_bounded(self):
        for t in [1300, 1500, 2000, 5000]:
            assert pf.reduction_rate(float(t)) <= pf.ARRHENIUS_PREFACTOR


class TestReductionEnergy:
    def test_positive(self):
        assert pf.reduction_energy_kj_per_kg(100.0) > 0.0

    def test_zero(self):
        assert pf.reduction_energy_kj_per_kg(0.0) == 0.0

    def test_linear(self):
        e1 = pf.reduction_energy_kj_per_kg(50.0)
        e2 = pf.reduction_energy_kj_per_kg(100.0)
        assert abs(e2 / e1 - 2.0) < 0.001


class TestCarbonDemand:
    def test_positive(self):
        assert pf.carbon_demand_kg(100.0) > 0.0

    def test_zero(self):
        assert pf.carbon_demand_kg(0.0) == 0.0

    def test_reasonable(self):
        c = pf.carbon_demand_kg(100.0)
        assert 10.0 < c < 40.0

    def test_linear(self):
        c1 = pf.carbon_demand_kg(50.0)
        c2 = pf.carbon_demand_kg(100.0)
        assert abs(c2 / c1 - 2.0) < 0.001


class TestCOProduction:
    def test_zero_zero_ore(self):
        assert pf.co_produced_kg(0.0, 1.0) == 0.0

    def test_zero_zero_red(self):
        assert pf.co_produced_kg(100.0, 0.0) == 0.0

    def test_positive(self):
        assert pf.co_produced_kg(100.0, 0.5) > 0.0

    def test_scales(self):
        co_h = pf.co_produced_kg(100.0, 0.5)
        co_f = pf.co_produced_kg(100.0, 1.0)
        assert abs(co_f / co_h - 2.0) < 0.001

    def test_capped(self):
        assert abs(pf.co_produced_kg(100.0, 1.0) - pf.co_produced_kg(100.0, 2.0)) < 1e-10


class TestElectrode:
    def test_wear_positive(self):
        assert pf.electrode_wear_kg(500.0, 1.0) > 0.0

    def test_wear_zero_current(self):
        assert pf.electrode_wear_kg(0.0, 1.0) == 0.0

    def test_wear_zero_time(self):
        assert pf.electrode_wear_kg(500.0, 0.0) == 0.0

    def test_wear_scales(self):
        w1 = pf.electrode_wear_kg(500.0, 1.0)
        w2 = pf.electrode_wear_kg(1000.0, 1.0)
        assert abs(w2 / w1 - 2.0) < 0.001

    def test_mass_positive(self):
        assert pf.electrode_mass_kg(0.10, 1.0) > 0.0

    def test_mass_zero_diam(self):
        assert pf.electrode_mass_kg(0.0, 1.0) == 0.0

    def test_mass_formula(self):
        d, l = 0.10, 1.0
        expected = math.pi * (d / 2) ** 2 * l * pf.ELECTRODE_DENSITY_KG_M3
        assert abs(pf.electrode_mass_kg(d, l) - expected) < 0.01


class TestMaxYield:
    def test_all_positive(self):
        y = pf.max_metal_yield_kg(100.0)
        for v in y.values():
            assert v > 0.0

    def test_sum_less_than_ore(self):
        y = pf.max_metal_yield_kg(100.0)
        assert sum(y.values()) < 100.0

    def test_iron_gt_aluminum(self):
        y = pf.max_metal_yield_kg(100.0)
        assert y["iron_kg"] > y["aluminum_kg"]

    def test_silicon_largest(self):
        y = pf.max_metal_yield_kg(100.0)
        assert y["silicon_kg"] > y["iron_kg"]

    def test_scales(self):
        y1 = pf.max_metal_yield_kg(50.0)
        y2 = pf.max_metal_yield_kg(100.0)
        assert abs(y2["iron_kg"] / y1["iron_kg"] - 2.0) < 0.001

    def test_keys(self):
        y = pf.max_metal_yield_kg(100.0)
        assert set(y.keys()) == {"iron_kg", "aluminum_kg", "titanium_kg", "silicon_kg"}


class TestCreateForge:
    def test_defaults(self, default_forge):
        assert default_forge.ore_charge_kg == pf.DEFAULT_CHARGE_KG
        assert default_forge.temperature_k == pf.MARS_AMBIENT_TEMP_K
        assert default_forge.melt_fraction == 0.0
        assert not default_forge.is_running

    def test_electrode_mass_computed(self, default_forge):
        assert default_forge.electrode_mass_kg > 0.0

    def test_zero_charge(self, no_charge_forge):
        assert no_charge_forge.ore_charge_kg == 0.0

    def test_current_clamped(self):
        f = pf.create_forge(arc_current_a=5000.0)
        assert f.arc_current_a == pf.MAX_ARC_CURRENT_A

    def test_negative_charge_zero(self):
        f = pf.create_forge(charge_kg=-10.0)
        assert f.ore_charge_kg == 0.0


class TestIgnition:
    def test_normal(self, default_forge):
        f = pf.ignite(default_forge)
        assert f.is_running and f.fault == ""

    def test_no_electrode(self, no_electrode_forge):
        f = pf.ignite(no_electrode_forge)
        assert not f.is_running and f.fault == "no_electrode"

    def test_no_charge(self, no_charge_forge):
        f = pf.ignite(no_charge_forge)
        assert not f.is_running and f.fault == "no_charge"

    def test_already_tapped(self, default_forge):
        default_forge.is_tapped = True
        f = pf.ignite(default_forge)
        assert f.fault == "already_tapped"


class TestShutdown:
    def test_stops(self, default_forge):
        f = pf.ignite(default_forge)
        f = pf.shutdown(f)
        assert not f.is_running


class TestTickHeating:
    def test_temp_rises(self, default_forge):
        f = pf.ignite(default_forge)
        t0 = f.temperature_k
        f = pf.tick(f)
        assert f.temperature_k > t0

    def test_no_change_off(self, default_forge):
        t0 = default_forge.temperature_k
        f = pf.tick(default_forge)
        assert f.temperature_k <= t0

    def test_cools_when_off(self):
        f = pf.create_forge()
        f.temperature_k = 1000.0
        f = pf.tick(f)
        assert pf.MARS_AMBIENT_TEMP_K <= f.temperature_k < 1000.0

    def test_multi_tick_heats(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(10):
            f = pf.tick(f)
        assert f.temperature_k > 500.0

    def test_melt_fraction_increases(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(15):
            f = pf.tick(f)
        assert f.melt_fraction > 0.0

    def test_energy_bookkeeping(self, default_forge):
        f = pf.ignite(default_forge)
        f = pf.tick(f)
        assert f.total_energy_kwh > 0.0 and f.total_ticks == 1


class TestTickReduction:
    def test_reduction_starts(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        assert f.reduction_fraction > 0.0

    def test_bounded(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(100):
            f = pf.tick(f)
            assert 0.0 <= f.reduction_fraction <= 1.0

    def test_carbon_consumed(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        if f.reduction_fraction > 0.0:
            assert f.carbon_consumed_kg > 0.0

    def test_co_produced(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        if f.reduction_fraction > 0.0:
            assert f.co_produced_kg > 0.0


class TestElectrodeWearTick:
    def test_decreases(self, default_forge):
        f = pf.ignite(default_forge)
        m0 = f.electrode_mass_kg
        f = pf.tick(f)
        assert f.electrode_mass_kg < m0

    def test_exhaustion_fault(self):
        f = pf.create_forge()
        f.electrode_mass_kg = 0.001
        f = pf.ignite(f)
        f = pf.tick(f)
        assert f.fault == "electrode_exhausted" and not f.is_running

    def test_never_negative(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(300):
            f = pf.tick(f)
        assert f.electrode_mass_kg >= 0.0


class TestTap:
    def test_after_reduction(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        f = pf.tap(f)
        assert f.is_tapped and not f.is_running

    def test_iron(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        f = pf.tap(f)
        assert f.iron_produced_kg > 0.0

    def test_all_metals(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        f = pf.tap(f)
        assert f.aluminum_produced_kg > 0.0
        assert f.titanium_produced_kg > 0.0
        assert f.silicon_produced_kg > 0.0

    def test_slag(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        f = pf.tap(f)
        assert f.slag_produced_kg > 0.0

    def test_ore_zero(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        f = pf.tap(f)
        assert f.ore_remaining_kg == 0.0

    def test_double_tap(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        f = pf.tap(f)
        fe1 = f.iron_produced_kg
        f = pf.tap(f)
        assert f.iron_produced_kg == fe1

    def test_no_reduction(self, default_forge):
        f = pf.tap(default_forge)
        assert f.fault == "nothing_to_tap"

    def test_cant_reignite(self, default_forge):
        f = pf.ignite(default_forge)
        for _ in range(25):
            f = pf.tick(f)
        f = pf.tap(f)
        f = pf.ignite(f)
        assert f.fault == "already_tapped"


class TestMassConservation:
    def test_mass_balance(self):
        f = pf.run_batch(charge_kg=200.0, max_ticks=150)
        total_metal = (f.iron_produced_kg + f.aluminum_produced_kg +
                       f.titanium_produced_kg + f.silicon_produced_kg)
        total_out = total_metal + f.slag_produced_kg + f.co_produced_kg
        assert total_out <= f.ore_charge_kg * 1.01

    def test_bounded_by_max(self):
        f = pf.run_batch(charge_kg=200.0, max_ticks=200)
        y = pf.max_metal_yield_kg(200.0)
        assert f.iron_produced_kg <= y["iron_kg"] * 1.001
        assert f.aluminum_produced_kg <= y["aluminum_kg"] * 1.001
        assert f.titanium_produced_kg <= y["titanium_kg"] * 1.001
        assert f.silicon_produced_kg <= y["silicon_kg"] * 1.001

    def test_proportional(self):
        f = pf.create_forge(charge_kg=200.0)
        f = pf.ignite(f)
        for _ in range(25):
            f = pf.tick(f)
        rf = f.reduction_fraction
        f = pf.tap(f)
        y = pf.max_metal_yield_kg(200.0)
        assert abs(f.iron_produced_kg - y["iron_kg"] * rf) < 0.01

    def test_slag_positive(self):
        f = pf.run_batch(charge_kg=200.0)
        assert f.slag_produced_kg >= 0.0


class TestEnergyConservation:
    def test_monotonic(self):
        f = pf.create_forge()
        f = pf.ignite(f)
        prev = 0.0
        for _ in range(20):
            f = pf.tick(f)
            assert f.total_energy_kwh >= prev
            prev = f.total_energy_kwh

    def test_no_energy_off(self):
        f = pf.create_forge()
        f = pf.tick(f)
        assert f.total_energy_kwh == 0.0


class TestPhysicalBounds:
    def test_temp_above_ambient_off(self):
        f = pf.create_forge()
        f.temperature_k = 300.0
        for _ in range(100):
            f = pf.tick(f)
        assert f.temperature_k >= pf.MARS_AMBIENT_TEMP_K

    def test_melt_bounded(self):
        f = pf.create_forge()
        f = pf.ignite(f)
        for _ in range(100):
            f = pf.tick(f)
            assert 0.0 <= f.melt_fraction <= 1.0

    def test_reduction_bounded(self):
        f = pf.create_forge()
        f = pf.ignite(f)
        for _ in range(100):
            f = pf.tick(f)
            assert 0.0 <= f.reduction_fraction <= 1.0

    def test_electrode_nonneg(self):
        f = pf.create_forge()
        f = pf.ignite(f)
        for _ in range(200):
            f = pf.tick(f)
            assert f.electrode_mass_kg >= 0.0

    def test_carbon_nonneg(self):
        f = pf.create_forge()
        f = pf.ignite(f)
        for _ in range(100):
            f = pf.tick(f)
            assert f.carbon_consumed_kg >= 0.0

    def test_co_nonneg(self):
        f = pf.create_forge()
        f = pf.ignite(f)
        for _ in range(100):
            f = pf.tick(f)
            assert f.co_produced_kg >= 0.0


class TestRunBatch:
    def test_completes(self):
        f = pf.run_batch(charge_kg=200.0, max_ticks=200)
        assert f.is_tapped

    def test_iron(self):
        f = pf.run_batch()
        assert f.iron_produced_kg > 0.0

    def test_silicon(self):
        f = pf.run_batch()
        assert f.silicon_produced_kg > 0.0

    def test_efficiency(self):
        f = pf.run_batch(charge_kg=200.0, max_ticks=200)
        assert 0.0 < pf.efficiency(f) < 1.0

    def test_energy_per_kg(self):
        f = pf.run_batch(charge_kg=200.0, max_ticks=200)
        assert 5.0 < pf.energy_per_kg_metal(f) < 500.0

    def test_small(self):
        f = pf.run_batch(charge_kg=10.0, max_ticks=100)
        assert f.is_tapped

    def test_large(self):
        f = pf.run_batch(charge_kg=1000.0, max_ticks=300)
        assert f.is_tapped

    def test_high_power(self):
        f = pf.run_batch(charge_kg=200.0, power_kw=500.0, max_ticks=100)
        assert f.is_tapped and f.reduction_fraction > 0.5


class TestSummary:
    def test_keys(self):
        f = pf.run_batch()
        s = pf.summary(f)
        expected = {"temperature_k", "melt_fraction", "reduction_fraction",
                    "iron_kg", "aluminum_kg", "titanium_kg", "silicon_kg",
                    "total_metal_kg", "slag_kg", "co_produced_kg",
                    "electrode_remaining_kg", "carbon_consumed_kg",
                    "energy_kwh", "ticks", "efficiency",
                    "is_running", "is_tapped", "fault"}
        assert set(s.keys()) == expected

    def test_total_metal(self):
        f = pf.run_batch()
        s = pf.summary(f)
        expected = round(f.iron_produced_kg + f.aluminum_produced_kg +
                         f.titanium_produced_kg + f.silicon_produced_kg, 3)
        assert abs(s["total_metal_kg"] - expected) < 0.01


class TestEfficiency:
    def test_zero_no_energy(self):
        assert pf.efficiency(pf.create_forge()) == 0.0

    def test_zero_no_metal(self):
        f = pf.create_forge()
        f.total_energy_kwh = 100.0
        assert pf.efficiency(f) == 0.0

    def test_bounded(self):
        f = pf.run_batch()
        assert 0.0 <= pf.efficiency(f) <= 1.0


class TestEnergyPerKg:
    def test_inf_no_metal(self):
        assert pf.energy_per_kg_metal(pf.create_forge()) == float("inf")

    def test_positive(self):
        f = pf.run_batch()
        assert 0.0 < pf.energy_per_kg_metal(f) < float("inf")


class TestEdgeCases:
    def test_zero_power(self, default_forge):
        f = pf.ignite(default_forge)
        t0 = f.temperature_k
        f = pf.tick(f, power_limit_kw=0.0)
        assert f.temperature_k <= t0 + 1.0

    def test_tiny_charge(self):
        f = pf.create_forge(charge_kg=0.01)
        f = pf.ignite(f)
        for _ in range(10):
            f = pf.tick(f)
        assert f.temperature_k > pf.MARS_AMBIENT_TEMP_K

    def test_max_current(self):
        f = pf.create_forge(arc_current_a=pf.MAX_ARC_CURRENT_A)
        f = pf.ignite(f)
        f = pf.tick(f)
        assert f.total_energy_kwh > 0.0

    def test_negative_power(self, default_forge):
        f = pf.ignite(default_forge)
        f = pf.tick(f, power_limit_kw=-10.0)
        assert f.total_ticks == 1


class TestSmoke:
    def test_10_step(self):
        f = pf.create_forge()
        f = pf.ignite(f)
        for i in range(10):
            f = pf.tick(f)
            assert f.total_ticks == i + 1

    def test_full_lifecycle(self):
        f = pf.create_forge(charge_kg=100.0)
        f = pf.ignite(f)
        assert f.is_running
        for _ in range(30):
            f = pf.tick(f)
        assert f.temperature_k >= pf.REGOLITH_MELTING_TEMP_K
        f = pf.tap(f)
        assert f.is_tapped
        total = (f.iron_produced_kg + f.aluminum_produced_kg +
                 f.titanium_produced_kg + f.silicon_produced_kg)
        assert total > 0.0

    def test_100_ticks(self):
        f = pf.create_forge()
        f = pf.ignite(f)
        for _ in range(100):
            f = pf.tick(f)
        assert f.total_ticks == 100
