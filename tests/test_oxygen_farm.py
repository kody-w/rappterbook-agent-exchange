"""Tests for oxygen_farm.py — Mars Colony Algae Photobioreactor."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.oxygen_farm import (
    ReactorState, SolRecord,
    monod_growth_rate, temperature_factor, ph_factor,
    density_inhibition, biomass_growth_g, o2_produced_g,
    co2_consumed_g, water_consumed_g, should_harvest,
    harvest_biomass, energy_per_sol_kwh, culture_health,
    tick_oxygen_farm, create_oxygen_farm, run_oxygen_farm,
    MU_MAX_PER_SOL, K_LIGHT, OPTIMAL_TEMP_C, OPTIMAL_PH,
    MAX_DENSITY_G_L, HARVEST_THRESHOLD_G_L, HARVEST_FRACTION,
    RESPIRATION_LOSS, O2_PER_BIOMASS_G, CO2_PER_BIOMASS_G,
    WATER_PER_BIOMASS_G, DEFAULT_VOLUME_L, DEFAULT_INOCULUM_G_L,
    MIN_GROWTH_TEMP_C, MAX_GROWTH_TEMP_C,
)


# ── ReactorState clamping ──────────────────────────────────────────

class TestReactorState:
    def test_defaults(self):
        s = ReactorState()
        assert s.volume_l == DEFAULT_VOLUME_L
        assert s.density_g_l == DEFAULT_INOCULUM_G_L
        assert s.sol == 0
        assert s.total_o2_produced_g == 0.0

    def test_negative_volume_clamped(self):
        s = ReactorState(volume_l=-10.0)
        assert s.volume_l == 0.0

    def test_density_clamped_high(self):
        s = ReactorState(density_g_l=100.0)
        assert s.density_g_l == MAX_DENSITY_G_L

    def test_density_clamped_low(self):
        s = ReactorState(density_g_l=-1.0)
        assert s.density_g_l == 0.0

    def test_irradiance_clamped(self):
        s = ReactorState(irradiance=-50.0)
        assert s.irradiance == 0.0

    def test_temperature_clamped_low(self):
        s = ReactorState(temperature_c=-100.0)
        assert s.temperature_c == -40.0

    def test_temperature_clamped_high(self):
        s = ReactorState(temperature_c=200.0)
        assert s.temperature_c == 60.0

    def test_ph_clamped_low(self):
        s = ReactorState(ph=-1.0)
        assert s.ph == 0.0

    def test_ph_clamped_high(self):
        s = ReactorState(ph=20.0)
        assert s.ph == 14.0

    def test_negative_sol_clamped(self):
        s = ReactorState(sol=-5)
        assert s.sol == 0

    def test_negative_totals_clamped(self):
        s = ReactorState(total_o2_produced_g=-1.0, total_co2_consumed_g=-1.0,
                         total_water_consumed_g=-1.0, total_biomass_harvested_g=-1.0,
                         total_energy_kwh=-1.0, harvests_performed=-1)
        assert s.total_o2_produced_g == 0.0
        assert s.total_co2_consumed_g == 0.0
        assert s.total_water_consumed_g == 0.0
        assert s.total_biomass_harvested_g == 0.0
        assert s.total_energy_kwh == 0.0
        assert s.harvests_performed == 0


# ── SolRecord ──────────────────────────────────────────────────────

class TestSolRecord:
    def test_defaults(self):
        r = SolRecord()
        assert r.sol == 0
        assert r.biomass_grown_g == 0.0
        assert r.culture_status == "thriving"

    def test_fields_accessible(self):
        r = SolRecord(sol=5, o2_produced_g=100.0, harvested_g=50.0)
        assert r.sol == 5
        assert r.o2_produced_g == 100.0
        assert r.harvested_g == 50.0


# ── Monod growth rate ──────────────────────────────────────────────

class TestMonodGrowthRate:
    def test_zero_irradiance(self):
        assert monod_growth_rate(0.0) == 0.0

    def test_negative_irradiance(self):
        assert monod_growth_rate(-100.0) == 0.0

    def test_at_half_saturation(self):
        """At I = K_I, rate should be µ_max / 2."""
        rate = monod_growth_rate(K_LIGHT)
        assert abs(rate - MU_MAX_PER_SOL / 2.0) < 1e-10

    def test_high_irradiance_approaches_max(self):
        rate = monod_growth_rate(10000.0)
        assert rate > 0.99 * MU_MAX_PER_SOL

    def test_monotonically_increasing(self):
        rates = [monod_growth_rate(i * 50.0) for i in range(1, 20)]
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i + 1]

    def test_zero_mu_max(self):
        assert monod_growth_rate(300.0, mu_max=0.0) == 0.0

    def test_custom_parameters(self):
        rate = monod_growth_rate(200.0, mu_max=1.0, k_light=200.0)
        assert abs(rate - 0.5) < 1e-10


# ── Temperature factor ─────────────────────────────────────────────

class TestTemperatureFactor:
    def test_optimal_is_one(self):
        f = temperature_factor(OPTIMAL_TEMP_C)
        assert abs(f - 1.0) < 1e-10

    def test_below_min_is_zero(self):
        assert temperature_factor(MIN_GROWTH_TEMP_C - 1.0) == 0.0

    def test_above_max_is_zero(self):
        assert temperature_factor(MAX_GROWTH_TEMP_C + 1.0) == 0.0

    def test_at_min_boundary(self):
        f = temperature_factor(MIN_GROWTH_TEMP_C)
        assert 0.0 < f < 1.0

    def test_symmetric_around_optimal(self):
        f_low = temperature_factor(OPTIMAL_TEMP_C - 5.0)
        f_high = temperature_factor(OPTIMAL_TEMP_C + 5.0)
        assert abs(f_low - f_high) < 1e-10

    def test_decreases_away_from_optimal(self):
        f_close = temperature_factor(OPTIMAL_TEMP_C - 3.0)
        f_far = temperature_factor(OPTIMAL_TEMP_C - 8.0)
        assert f_close > f_far


# ── pH factor ──────────────────────────────────────────────────────

class TestPhFactor:
    def test_optimal_is_one(self):
        f = ph_factor(OPTIMAL_PH)
        assert abs(f - 1.0) < 1e-10

    def test_extreme_low_is_near_zero(self):
        f = ph_factor(0.0)
        assert f < 0.01

    def test_extreme_high_is_near_zero(self):
        f = ph_factor(14.0)
        assert f < 0.01

    def test_clamped_negative(self):
        f = ph_factor(-5.0)
        assert f == ph_factor(0.0)

    def test_clamped_above_14(self):
        f = ph_factor(20.0)
        assert f == ph_factor(14.0)

    def test_symmetric(self):
        f_low = ph_factor(OPTIMAL_PH - 0.5)
        f_high = ph_factor(OPTIMAL_PH + 0.5)
        assert abs(f_low - f_high) < 1e-10


# ── Density inhibition ─────────────────────────────────────────────

class TestDensityInhibition:
    def test_zero_density_is_one(self):
        assert density_inhibition(0.0) == 1.0

    def test_max_density_is_zero(self):
        assert density_inhibition(MAX_DENSITY_G_L) == 0.0

    def test_half_density(self):
        f = density_inhibition(MAX_DENSITY_G_L / 2.0)
        assert 0.0 < f < 1.0

    def test_monotonically_decreasing(self):
        values = [density_inhibition(i * 0.5) for i in range(int(MAX_DENSITY_G_L * 2))]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_negative_density(self):
        assert density_inhibition(-5.0) == 1.0

    def test_zero_max_density(self):
        assert density_inhibition(1.0, max_density=0.0) == 0.0

    def test_over_max_clamped(self):
        assert density_inhibition(100.0) == 0.0


# ── Biomass growth ─────────────────────────────────────────────────

class TestBiomassGrowth:
    def test_zero_volume(self):
        assert biomass_growth_g(1.0, 0.0, 0.5, 1.0, 1.0, 1.0) == 0.0

    def test_zero_growth_rate(self):
        assert biomass_growth_g(1.0, 500.0, 0.0, 1.0, 1.0, 1.0) == 0.0

    def test_positive_growth(self):
        g = biomass_growth_g(1.0, 500.0, 0.5, 1.0, 1.0, 1.0)
        assert g == 1.0 * 500.0 * 0.5 * 1.0 * 1.0 * 1.0
        assert g == 250.0

    def test_factors_multiply(self):
        g = biomass_growth_g(2.0, 100.0, 0.5, 0.5, 0.5, 0.5)
        expected = 2.0 * 100.0 * 0.5 * 0.5 * 0.5 * 0.5
        assert abs(g - expected) < 1e-10

    def test_never_negative(self):
        g = biomass_growth_g(0.0, 500.0, 0.5, 1.0, 1.0, 1.0)
        assert g >= 0.0


# ── O₂ / CO₂ / H₂O exchange ──────────────────────────────────────

class TestGasExchange:
    def test_o2_positive_for_growth(self):
        assert o2_produced_g(100.0) > 0.0

    def test_o2_zero_for_zero_growth(self):
        assert o2_produced_g(0.0) == 0.0

    def test_o2_includes_respiration_loss(self):
        gross = 100.0 * O2_PER_BIOMASS_G
        net = o2_produced_g(100.0)
        assert abs(net - gross * (1.0 - RESPIRATION_LOSS)) < 1e-10

    def test_co2_proportional(self):
        c = co2_consumed_g(100.0)
        assert abs(c - 100.0 * CO2_PER_BIOMASS_G) < 1e-10

    def test_co2_zero(self):
        assert co2_consumed_g(0.0) == 0.0

    def test_water_proportional(self):
        w = water_consumed_g(100.0)
        assert abs(w - 100.0 * WATER_PER_BIOMASS_G) < 1e-10

    def test_water_zero(self):
        assert water_consumed_g(0.0) == 0.0

    def test_o2_never_negative(self):
        assert o2_produced_g(-10.0) >= 0.0

    def test_co2_never_negative(self):
        assert co2_consumed_g(-10.0) >= 0.0


# ── Harvest logic ──────────────────────────────────────────────────

class TestHarvest:
    def test_below_threshold(self):
        assert should_harvest(HARVEST_THRESHOLD_G_L - 0.1) is False

    def test_at_threshold(self):
        assert should_harvest(HARVEST_THRESHOLD_G_L) is True

    def test_above_threshold(self):
        assert should_harvest(HARVEST_THRESHOLD_G_L + 1.0) is True

    def test_harvest_reduces_density(self):
        new_d, harvested = harvest_biomass(6.0, 500.0)
        assert new_d < 6.0
        assert harvested > 0.0

    def test_harvest_mass_balance(self):
        density = 6.0
        volume = 500.0
        original_mass = density * volume
        new_d, harvested = harvest_biomass(density, volume)
        remaining_mass = new_d * volume
        assert abs(remaining_mass + harvested - original_mass) < 1e-10

    def test_harvest_fraction_default(self):
        new_d, _ = harvest_biomass(10.0, 100.0)
        expected = 10.0 * (1.0 - HARVEST_FRACTION)
        assert abs(new_d - expected) < 1e-10

    def test_harvest_fraction_clamped(self):
        new_d, harvested = harvest_biomass(5.0, 100.0, fraction=2.0)
        assert new_d == 0.0
        assert abs(harvested - 5.0 * 100.0) < 1e-10


# ── Energy ─────────────────────────────────────────────────────────

class TestEnergy:
    def test_positive_for_active_reactor(self):
        e = energy_per_sol_kwh(500.0, 100.0, 0.0)
        assert e > 0.0

    def test_harvest_adds_energy(self):
        e_no_harvest = energy_per_sol_kwh(500.0, 100.0, 0.0)
        e_harvest = energy_per_sol_kwh(500.0, 100.0, 500.0)
        assert e_harvest > e_no_harvest

    def test_larger_volume_more_energy(self):
        e_small = energy_per_sol_kwh(100.0, 50.0, 0.0)
        e_large = energy_per_sol_kwh(1000.0, 50.0, 0.0)
        assert e_large > e_small

    def test_never_negative(self):
        assert energy_per_sol_kwh(0.0, 0.0, 0.0) >= 0.0


# ── Culture health ─────────────────────────────────────────────────

class TestCultureHealth:
    def test_thriving(self):
        assert culture_health(1.0, 1.0, 2.0) == "thriving"

    def test_stressed(self):
        assert culture_health(0.5, 0.8, 2.0) == "stressed"

    def test_critical(self):
        assert culture_health(0.2, 0.2, 2.0) == "critical"

    def test_dead_zero_density(self):
        assert culture_health(1.0, 1.0, 0.0) == "dead"

    def test_dead_trace_density(self):
        assert culture_health(1.0, 1.0, 0.005) == "dead"


# ── tick_oxygen_farm ───────────────────────────────────────────────

class TestTickOxygenFarm:
    def test_sol_increments(self):
        s = create_oxygen_farm()
        s2, _ = tick_oxygen_farm(s)
        assert s2.sol == 1

    def test_o2_produced(self):
        s = create_oxygen_farm()
        s2, rec = tick_oxygen_farm(s)
        assert s2.total_o2_produced_g > 0.0
        assert rec.o2_produced_g > 0.0

    def test_co2_consumed(self):
        s = create_oxygen_farm()
        _, rec = tick_oxygen_farm(s)
        assert rec.co2_consumed_g > 0.0

    def test_water_consumed(self):
        s = create_oxygen_farm()
        _, rec = tick_oxygen_farm(s)
        assert rec.water_consumed_g > 0.0

    def test_density_increases_from_growth(self):
        s = create_oxygen_farm(inoculum_g_l=0.5)
        s2, _ = tick_oxygen_farm(s)
        assert s2.density_g_l > 0.5

    def test_energy_consumed(self):
        s = create_oxygen_farm()
        _, rec = tick_oxygen_farm(s)
        assert rec.energy_kwh > 0.0

    def test_power_limited(self):
        """When power is scarce, growth is reduced."""
        s = create_oxygen_farm()
        _, rec_full = tick_oxygen_farm(s, available_power_kwh=1000.0)
        _, rec_low = tick_oxygen_farm(s, available_power_kwh=0.1)
        assert rec_low.biomass_grown_g <= rec_full.biomass_grown_g

    def test_harvest_triggers_at_high_density(self):
        s = create_oxygen_farm(inoculum_g_l=4.9)
        # Run a few sols to push past threshold
        for _ in range(5):
            s, _ = tick_oxygen_farm(s)
        assert s.harvests_performed >= 1
        assert s.total_biomass_harvested_g > 0.0

    def test_zero_irradiance_no_growth(self):
        s = create_oxygen_farm(irradiance=0.0)
        s2, rec = tick_oxygen_farm(s)
        assert rec.biomass_grown_g == 0.0
        assert rec.o2_produced_g == 0.0

    def test_extreme_cold_no_growth(self):
        s = create_oxygen_farm(temperature_c=0.0)
        s2, rec = tick_oxygen_farm(s)
        assert rec.biomass_grown_g == 0.0

    def test_extreme_ph_minimal_growth(self):
        s = create_oxygen_farm(ph=2.0)
        _, rec = tick_oxygen_farm(s)
        # Growth should be near zero at pH 2
        assert rec.biomass_grown_g < 1.0


# ── create_oxygen_farm ─────────────────────────────────────────────

class TestCreateOxygenFarm:
    def test_defaults(self):
        s = create_oxygen_farm()
        assert s.volume_l == DEFAULT_VOLUME_L
        assert s.density_g_l == DEFAULT_INOCULUM_G_L
        assert s.sol == 0

    def test_custom_volume(self):
        s = create_oxygen_farm(volume_l=1000.0)
        assert s.volume_l == 1000.0

    def test_custom_inoculum(self):
        s = create_oxygen_farm(inoculum_g_l=3.0)
        assert s.density_g_l == 3.0

    def test_custom_irradiance(self):
        s = create_oxygen_farm(irradiance=500.0)
        assert s.irradiance == 500.0

    def test_custom_temp(self):
        s = create_oxygen_farm(temperature_c=30.0)
        assert s.temperature_c == 30.0

    def test_custom_ph(self):
        s = create_oxygen_farm(ph=6.8)
        assert s.ph == 6.8


# ── run_oxygen_farm ────────────────────────────────────────────────

class TestRunOxygenFarm:
    def test_smoke_10_sols(self):
        s = create_oxygen_farm()
        final, records = run_oxygen_farm(s, 10)
        assert final.sol == 10
        assert len(records) == 10

    def test_smoke_30_sols(self):
        s = create_oxygen_farm()
        final, records = run_oxygen_farm(s, 30)
        assert final.sol == 30
        assert final.total_o2_produced_g > 0.0

    def test_zero_sols(self):
        s = create_oxygen_farm()
        final, records = run_oxygen_farm(s, 0)
        assert final.sol == 0
        assert len(records) == 0

    def test_negative_sols(self):
        s = create_oxygen_farm()
        final, records = run_oxygen_farm(s, -5)
        assert len(records) == 0

    def test_100_sol_smoke(self):
        """Extended run — should not crash and should produce O₂."""
        s = create_oxygen_farm()
        final, records = run_oxygen_farm(s, 100)
        assert final.sol == 100
        assert final.total_o2_produced_g > 1000.0


# ── Invariants (property-based) ────────────────────────────────────

class TestInvariants:
    def test_o2_monotonically_increasing(self):
        s = create_oxygen_farm()
        prev = 0.0
        for _ in range(20):
            s, _ = tick_oxygen_farm(s)
            assert s.total_o2_produced_g >= prev
            prev = s.total_o2_produced_g

    def test_co2_monotonically_increasing(self):
        s = create_oxygen_farm()
        prev = 0.0
        for _ in range(20):
            s, _ = tick_oxygen_farm(s)
            assert s.total_co2_consumed_g >= prev
            prev = s.total_co2_consumed_g

    def test_water_monotonically_increasing(self):
        s = create_oxygen_farm()
        prev = 0.0
        for _ in range(20):
            s, _ = tick_oxygen_farm(s)
            assert s.total_water_consumed_g >= prev
            prev = s.total_water_consumed_g

    def test_energy_monotonically_increasing(self):
        s = create_oxygen_farm()
        prev = 0.0
        for _ in range(20):
            s, _ = tick_oxygen_farm(s)
            assert s.total_energy_kwh >= prev
            prev = s.total_energy_kwh

    def test_density_bounded(self):
        s = create_oxygen_farm()
        for _ in range(50):
            s, _ = tick_oxygen_farm(s)
            assert 0.0 <= s.density_g_l <= MAX_DENSITY_G_L

    def test_o2_co2_ratio_physical(self):
        """O₂ produced should be proportional to CO₂ consumed."""
        s = create_oxygen_farm()
        final, _ = run_oxygen_farm(s, 50)
        if final.total_co2_consumed_g > 0:
            ratio = final.total_o2_produced_g / final.total_co2_consumed_g
            # Should be close to O2_PER_BIOMASS_G / CO2_PER_BIOMASS_G * (1-resp)
            expected = O2_PER_BIOMASS_G * (1.0 - RESPIRATION_LOSS) / CO2_PER_BIOMASS_G
            assert abs(ratio - expected) < 0.01

    def test_sol_counter_sequential(self):
        s = create_oxygen_farm()
        for i in range(1, 15):
            s, rec = tick_oxygen_farm(s)
            assert s.sol == i
            assert rec.sol == i

    def test_growth_factors_bounded(self):
        """All growth factors in [0, 1] over a run."""
        s = create_oxygen_farm()
        for _ in range(20):
            s, rec = tick_oxygen_farm(s)
            assert 0.0 <= rec.temp_factor <= 1.0
            assert 0.0 <= rec.ph_factor <= 1.0
            assert 0.0 <= rec.growth_rate <= MU_MAX_PER_SOL


# ── Edge cases ─────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_volume_reactor(self):
        s = create_oxygen_farm(volume_l=0.0)
        s2, rec = tick_oxygen_farm(s)
        assert rec.biomass_grown_g == 0.0

    def test_dead_culture_no_growth(self):
        s = create_oxygen_farm(inoculum_g_l=0.0)
        s2, rec = tick_oxygen_farm(s)
        assert rec.biomass_grown_g == 0.0
        assert rec.culture_status == "dead"

    def test_max_density_no_growth(self):
        """At max density, inhibition = 0 → no growth."""
        s = create_oxygen_farm(inoculum_g_l=MAX_DENSITY_G_L)
        s2, rec = tick_oxygen_farm(s)
        assert rec.biomass_grown_g == 0.0

    def test_zero_power(self):
        s = create_oxygen_farm()
        s2, rec = tick_oxygen_farm(s, available_power_kwh=0.0)
        # Should still produce some growth (power check is >0 for scaling)
        # With 0 power, energy is capped at 0 but the available_power > 0 check
        # means scale = 0/energy = 0, so growth = 0
        assert rec.energy_kwh == 0.0

    def test_very_high_irradiance(self):
        s = create_oxygen_farm(irradiance=10000.0)
        _, rec = tick_oxygen_farm(s)
        # Should still be bounded by other factors
        assert rec.biomass_grown_g > 0.0
        assert rec.growth_rate <= MU_MAX_PER_SOL

    def test_freezing_kills_culture(self):
        s = create_oxygen_farm(temperature_c=-20.0)
        s2, rec = tick_oxygen_farm(s)
        assert rec.biomass_grown_g == 0.0
        assert rec.temp_factor == 0.0

    def test_acidic_ph_reduces_growth(self):
        s_normal = create_oxygen_farm(ph=7.0)
        s_acidic = create_oxygen_farm(ph=4.0)
        _, rec_normal = tick_oxygen_farm(s_normal)
        _, rec_acidic = tick_oxygen_farm(s_acidic)
        assert rec_normal.biomass_grown_g > rec_acidic.biomass_grown_g

    def test_harvest_cycle_stabilizes(self):
        """Over 200 sols, density should oscillate around harvest threshold."""
        s = create_oxygen_farm()
        densities = []
        for _ in range(200):
            s, _ = tick_oxygen_farm(s)
            densities.append(s.density_g_l)
        # Last 50 sols should stay in a reasonable range
        late = densities[-50:]
        assert all(0.0 < d <= MAX_DENSITY_G_L for d in late)
        # Should have harvested multiple times
        assert s.harvests_performed >= 3
