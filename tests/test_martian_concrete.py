"""Tests for martian_concrete.py — Sulfur Concrete Production for Mars Construction."""
from __future__ import annotations

import json
import math
import os
import sys
import dataclasses

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import martian_concrete as mc


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def plant():
    return mc.create_plant(initial_regolith_kg=1000.0)


@pytest.fixture
def empty_plant():
    return mc.create_plant()


# ── 1. Heating energy ──────────────────────────────────────────────────

class TestHeatingEnergy:
    def test_zero_mass(self):
        assert mc.heating_energy_kj(0.0, 210.0, 400.0, 0.71) == 0.0

    def test_no_temp_change(self):
        assert mc.heating_energy_kj(1.0, 300.0, 300.0, 0.71) == 0.0

    def test_cooling_returns_zero(self):
        assert mc.heating_energy_kj(1.0, 400.0, 300.0, 0.71) == 0.0

    def test_sensible_heat_only(self):
        energy = mc.heating_energy_kj(1.0, 200.0, 300.0, 1.0)
        assert abs(energy - 100.0) < 0.01

    def test_includes_phase_change(self):
        energy_no_phase = mc.heating_energy_kj(
            1.0, 200.0, 400.0, 0.71, enthalpy_fusion=0.0
        )
        energy_with_phase = mc.heating_energy_kj(
            1.0, 200.0, 400.0, 0.71,
            enthalpy_fusion=54.0, melting_point_k=388.4
        )
        assert energy_with_phase > energy_no_phase
        assert abs(energy_with_phase - energy_no_phase - 54.0) < 0.01

    def test_phase_change_not_crossed(self):
        energy = mc.heating_energy_kj(
            1.0, 200.0, 300.0, 0.71,
            enthalpy_fusion=54.0, melting_point_k=388.4
        )
        expected = 1.0 * 0.71 * 100.0
        assert abs(energy - expected) < 0.01

    def test_scales_with_mass(self):
        e1 = mc.heating_energy_kj(1.0, 210.0, 400.0, 0.71)
        e2 = mc.heating_energy_kj(2.0, 210.0, 400.0, 0.71)
        assert abs(e2 - 2.0 * e1) < 0.01


# ── 2. Sulfur heating ──────────────────────────────────────────────────

class TestSulfurHeating:
    def test_positive_energy(self):
        energy = mc.sulfur_heating_energy_kj(1.0)
        assert energy > 0.0

    def test_includes_fusion(self):
        energy = mc.sulfur_heating_energy_kj(1.0)
        sensible_only = 1.0 * mc.SULFUR_SPECIFIC_HEAT_KJ_KG_K * (
            mc.SULFUR_WORKING_K - mc.MARS_AMBIENT_K
        )
        assert energy > sensible_only

    def test_scales_linearly(self):
        e1 = mc.sulfur_heating_energy_kj(1.0)
        e5 = mc.sulfur_heating_energy_kj(5.0)
        assert abs(e5 - 5.0 * e1) < 0.01

    def test_warmer_ambient_less_energy(self):
        e_cold = mc.sulfur_heating_energy_kj(1.0, ambient_k=210.0)
        e_warm = mc.sulfur_heating_energy_kj(1.0, ambient_k=300.0)
        assert e_cold > e_warm


# ── 3. Regolith processing ─────────────────────────────────────────────

class TestRegolithProcessing:
    def test_sulfur_extraction(self):
        sulfur = mc.sulfur_from_regolith_kg(100.0)
        assert abs(sulfur - 5.0) < 0.01

    def test_aggregate_sieving(self):
        agg = mc.aggregate_from_regolith_kg(100.0)
        assert abs(agg - 70.0) < 0.01

    def test_zero_regolith(self):
        assert mc.sulfur_from_regolith_kg(0.0) == 0.0
        assert mc.aggregate_from_regolith_kg(0.0) == 0.0

    def test_negative_regolith(self):
        assert mc.sulfur_from_regolith_kg(-10.0) == 0.0
        assert mc.aggregate_from_regolith_kg(-10.0) == 0.0

    def test_mass_conservation(self):
        raw = 1000.0
        sulfur = mc.sulfur_from_regolith_kg(raw)
        agg = mc.aggregate_from_regolith_kg(raw)
        assert sulfur + agg <= raw


# ── 4. Concrete strength ───────────────────────────────────────────────

class TestConcreteStrength:
    def test_optimal_ratio_peak(self):
        # With zero porosity, peak is PEAK_STRENGTH_MPA exactly
        strength_pure = mc.concrete_strength_mpa(mc.OPTIMAL_SULFUR_RATIO, porosity=0.0)
        assert abs(strength_pure - mc.PEAK_STRENGTH_MPA) < 0.01
        # With default porosity (0.05), strength is reduced
        strength_default = mc.concrete_strength_mpa(mc.OPTIMAL_SULFUR_RATIO)
        assert strength_default < mc.PEAK_STRENGTH_MPA
        assert strength_default > 30.0

    def test_below_minimum_zero(self):
        assert mc.concrete_strength_mpa(0.10) == 0.0

    def test_above_maximum_zero(self):
        assert mc.concrete_strength_mpa(0.60) == 0.0

    def test_porosity_reduces_strength(self):
        s_low = mc.concrete_strength_mpa(0.35, porosity=0.05)
        s_high = mc.concrete_strength_mpa(0.35, porosity=0.30)
        assert s_low > s_high

    def test_zero_porosity_highest(self):
        s_zero = mc.concrete_strength_mpa(0.35, porosity=0.0)
        s_some = mc.concrete_strength_mpa(0.35, porosity=0.05)
        assert s_zero >= s_some

    def test_symmetric_around_optimum(self):
        delta = 0.05
        s_low = mc.concrete_strength_mpa(mc.OPTIMAL_SULFUR_RATIO - delta)
        s_high = mc.concrete_strength_mpa(mc.OPTIMAL_SULFUR_RATIO + delta)
        assert abs(s_low - s_high) < 1.0

    def test_always_non_negative(self):
        for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
            assert mc.concrete_strength_mpa(ratio) >= 0.0

    def test_strength_bounded(self):
        for ratio in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            s = mc.concrete_strength_mpa(ratio)
            assert s <= mc.PEAK_STRENGTH_MPA + 0.01

    def test_porosity_clamp(self):
        s = mc.concrete_strength_mpa(0.35, porosity=1.5)
        assert s >= 0.0
        s2 = mc.concrete_strength_mpa(0.35, porosity=-0.1)
        assert s2 >= 0.0


# ── 5. Cooling time ────────────────────────────────────────────────────

class TestCoolingTime:
    def test_positive(self):
        t = mc.cooling_time_seconds(mc.BLOCK_VOLUME_M3, mc.SULFUR_WORKING_K)
        assert t > 0.0

    def test_zero_volume(self):
        assert mc.cooling_time_seconds(0.0, mc.SULFUR_WORKING_K) == 0.0

    def test_already_cool(self):
        assert mc.cooling_time_seconds(0.01, mc.MARS_AMBIENT_K) == 0.0

    def test_larger_blocks_cool_slower(self):
        t_small = mc.cooling_time_seconds(0.001, mc.SULFUR_WORKING_K)
        t_large = mc.cooling_time_seconds(0.01, mc.SULFUR_WORKING_K)
        assert t_large > t_small

    def test_hotter_pour_longer_cool(self):
        t_warm = mc.cooling_time_seconds(0.01, 400.0)
        t_hot = mc.cooling_time_seconds(0.01, 500.0)
        assert t_hot > t_warm

    def test_mars_vs_earth_cooling(self):
        # Mars has bigger dT (400->210 vs 400->293), so log ratio is
        # larger -> longer cooling time despite stronger driving force.
        t_mars = mc.cooling_time_seconds(0.01, 400.0, ambient_k=210.0)
        t_earth = mc.cooling_time_seconds(0.01, 400.0, ambient_k=293.0)
        assert t_mars > t_earth


# ── 6. Thermal fatigue ─────────────────────────────────────────────────

class TestThermalFatigue:
    def test_initial_strength_preserved(self):
        s = mc.thermal_fatigue_strength(45.0, 0)
        assert abs(s - 45.0) < 0.01

    def test_degrades_over_time(self):
        s0 = 45.0
        s100 = mc.thermal_fatigue_strength(s0, 100)
        assert s100 < s0

    def test_monotonic_decrease(self):
        s0 = 45.0
        prev = s0
        for sol in [10, 50, 100, 500, 1000]:
            s = mc.thermal_fatigue_strength(s0, sol)
            assert s <= prev
            prev = s

    def test_never_negative(self):
        s = mc.thermal_fatigue_strength(45.0, 10000)
        assert s >= 0.0

    def test_higher_amplitude_faster_degradation(self):
        s_mild = mc.thermal_fatigue_strength(45.0, 100, cycle_amplitude_k=20.0)
        s_severe = mc.thermal_fatigue_strength(45.0, 100, cycle_amplitude_k=80.0)
        assert s_mild > s_severe

    def test_zero_initial_strength(self):
        s = mc.thermal_fatigue_strength(0.0, 100)
        assert s == 0.0

    def test_negative_initial_clamped(self):
        s = mc.thermal_fatigue_strength(-5.0, 10)
        assert s >= 0.0


# ── 7. Block calculation ───────────────────────────────────────────────

class TestBlocks:
    def test_standard_block(self):
        blocks = mc.blocks_from_concrete_kg(mc.BLOCK_STANDARD_KG)
        assert blocks == 1

    def test_multiple_blocks(self):
        blocks = mc.blocks_from_concrete_kg(200.0)
        assert blocks == 10

    def test_fractional_discarded(self):
        blocks = mc.blocks_from_concrete_kg(25.0)
        assert blocks == 1

    def test_zero(self):
        assert mc.blocks_from_concrete_kg(0.0) == 0

    def test_negative(self):
        assert mc.blocks_from_concrete_kg(-10.0) == 0

    def test_less_than_one_block(self):
        assert mc.blocks_from_concrete_kg(10.0) == 0


# ── 8. Regolith requirement ────────────────────────────────────────────

class TestRegolithNeeded:
    def test_positive(self):
        r = mc.regolith_needed_kg(100.0)
        assert r > 0.0

    def test_zero_concrete(self):
        assert mc.regolith_needed_kg(0.0) == 0.0

    def test_more_concrete_more_regolith(self):
        r1 = mc.regolith_needed_kg(100.0)
        r2 = mc.regolith_needed_kg(200.0)
        assert r2 > r1

    def test_scales_linearly(self):
        r1 = mc.regolith_needed_kg(100.0)
        r2 = mc.regolith_needed_kg(200.0)
        assert abs(r2 - 2.0 * r1) < 0.01

    def test_regolith_much_more_than_concrete(self):
        r = mc.regolith_needed_kg(100.0)
        assert r > 100.0


# ── 9. Batch energy ────────────────────────────────────────────────────

class TestBatchEnergy:
    def test_positive(self):
        e = mc.batch_energy_kj(10.0, 50.0)
        assert e > 0.0

    def test_more_sulfur_more_energy(self):
        e1 = mc.batch_energy_kj(10.0, 50.0)
        e2 = mc.batch_energy_kj(20.0, 50.0)
        assert e2 > e1

    def test_includes_mixing(self):
        e_heat_only = mc.sulfur_heating_energy_kj(10.0) / mc.KILN_EFFICIENCY
        e_total = mc.batch_energy_kj(10.0, 50.0)
        assert e_total > e_heat_only

    def test_warmer_ambient_less_energy(self):
        e_cold = mc.batch_energy_kj(10.0, 50.0, ambient_k=210.0)
        e_warm = mc.batch_energy_kj(10.0, 50.0, ambient_k=300.0)
        assert e_cold > e_warm


# ── 10. Plant creation ─────────────────────────────────────────────────

class TestPlantCreation:
    def test_empty_plant(self, empty_plant):
        assert empty_plant.sol == 0
        assert empty_plant.sulfur_stockpile_kg == 0.0
        assert empty_plant.aggregate_stockpile_kg == 0.0
        assert empty_plant.blocks_produced == 0

    def test_preloaded_plant(self, plant):
        assert plant.sulfur_stockpile_kg > 0.0
        assert plant.aggregate_stockpile_kg > 0.0

    def test_preloaded_stockpile_matches_extraction(self, plant):
        expected_sulfur = mc.sulfur_from_regolith_kg(1000.0)
        expected_agg = mc.aggregate_from_regolith_kg(1000.0)
        assert abs(plant.sulfur_stockpile_kg - expected_sulfur) < 0.01
        assert abs(plant.aggregate_stockpile_kg - expected_agg) < 0.01

    def test_plant_active_by_default(self, plant):
        assert plant.plant_active is True


# ── 11. Tick engine basics ─────────────────────────────────────────────

class TestTick:
    def test_advances_sol(self, plant):
        mc.tick(plant)
        assert plant.sol == 1

    def test_returns_tick_result(self, plant):
        result = mc.tick(plant)
        assert isinstance(result, mc.TickResult)

    def test_produces_concrete(self, plant):
        result = mc.tick(plant)
        assert result.concrete_produced_kg > 0.0

    def test_produces_blocks(self, plant):
        result = mc.tick(plant)
        assert result.blocks_produced > 0

    def test_consumes_energy(self, plant):
        result = mc.tick(plant)
        assert result.energy_consumed_kj > 0.0

    def test_consumes_sulfur(self, plant):
        initial_sulfur = plant.sulfur_stockpile_kg
        mc.tick(plant)
        assert plant.sulfur_stockpile_kg < initial_sulfur

    def test_consumes_aggregate(self, plant):
        initial_agg = plant.aggregate_stockpile_kg
        mc.tick(plant)
        assert plant.aggregate_stockpile_kg < initial_agg

    def test_inactive_plant_no_production(self, plant):
        plant.plant_active = False
        result = mc.tick(plant)
        assert result.batches_run == 0
        assert result.concrete_produced_kg == 0.0

    def test_no_regolith_delivery(self, empty_plant):
        result = mc.tick(empty_plant, regolith_delivery_kg=0.0)
        assert result.concrete_produced_kg == 0.0

    def test_regolith_intake_recorded(self, empty_plant):
        result = mc.tick(empty_plant, regolith_delivery_kg=500.0)
        assert result.regolith_intake_kg == 500.0


# ── 12. Mass conservation ─────────────────────────────────────────────

class TestMassConservation:
    def test_input_equals_output(self, plant):
        initial_sulfur = plant.sulfur_stockpile_kg
        initial_agg = plant.aggregate_stockpile_kg
        result = mc.tick(plant, regolith_delivery_kg=0.0)
        sulfur_remaining = plant.sulfur_stockpile_kg
        agg_remaining = plant.aggregate_stockpile_kg
        sulfur_used = initial_sulfur - sulfur_remaining
        agg_used = initial_agg - agg_remaining
        mass_in = sulfur_used + agg_used
        assert abs(result.concrete_produced_kg - mass_in) < 0.1

    def test_stockpiles_non_negative(self, plant):
        for _ in range(50):
            mc.tick(plant, regolith_delivery_kg=10.0)
        assert plant.sulfur_stockpile_kg >= 0.0
        assert plant.aggregate_stockpile_kg >= 0.0

    def test_total_regolith_tracked(self, plant):
        initial_total = plant.total_regolith_processed_kg
        mc.tick(plant, regolith_delivery_kg=200.0)
        assert abs(plant.total_regolith_processed_kg - initial_total - 200.0) < 0.01


# ── 13. Energy constraints ─────────────────────────────────────────────

class TestEnergyConstraints:
    def test_limited_power_fewer_batches(self, plant):
        r_unlimited = mc.tick(
            mc.create_plant(initial_regolith_kg=10000.0),
            power_available_kj=float("inf"),
        )
        r_limited = mc.tick(
            mc.create_plant(initial_regolith_kg=10000.0),
            power_available_kj=5000.0,
        )
        assert r_limited.batches_run <= r_unlimited.batches_run

    def test_zero_power_no_production(self, plant):
        result = mc.tick(plant, power_available_kj=0.0)
        assert result.batches_run == 0
        assert result.concrete_produced_kg == 0.0

    def test_energy_consumed_bounded(self, plant):
        result = mc.tick(plant, power_available_kj=10000.0)
        assert result.energy_consumed_kj <= 10000.0


# ── 14. Mix design variants ───────────────────────────────────────────

class TestMixDesign:
    def test_optimal_ratio(self, plant):
        result = mc.tick(plant, sulfur_ratio=mc.OPTIMAL_SULFUR_RATIO)
        assert result.batch_strength_mpa > 35.0

    def test_low_sulfur_weaker(self):
        p1 = mc.create_plant(initial_regolith_kg=5000.0)
        p2 = mc.create_plant(initial_regolith_kg=5000.0)
        r_opt = mc.tick(p1, sulfur_ratio=0.35)
        r_low = mc.tick(p2, sulfur_ratio=0.22)
        assert r_opt.batch_strength_mpa > r_low.batch_strength_mpa

    def test_ratio_clamped_to_valid_range(self, plant):
        result = mc.tick(plant, sulfur_ratio=0.01)
        assert result.batch_strength_mpa >= 0.0

    @pytest.mark.parametrize("ratio", [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    def test_valid_ratios_produce(self, ratio):
        p = mc.create_plant(initial_regolith_kg=5000.0)
        result = mc.tick(p, sulfur_ratio=ratio)
        assert result.batch_strength_mpa >= 0.0


# ── 15. Cooling physics ───────────────────────────────────────────────

class TestCoolingPhysics:
    def test_blocks_have_cooling_time(self, plant):
        result = mc.tick(plant)
        if result.blocks_produced > 0:
            assert result.cooling_time_s > 0.0

    def test_no_blocks_no_cooling(self, empty_plant):
        result = mc.tick(empty_plant, regolith_delivery_kg=0.0)
        assert result.cooling_time_s == 0.0


# ── 16. Multi-sol simulation ──────────────────────────────────────────

class TestSimulation:
    def test_run_10_sols(self):
        state = mc.create_plant(initial_regolith_kg=2000.0)
        results = mc.run_simulation(state, sols=10)
        assert len(results) == 10

    def test_sol_counter_increments(self):
        state = mc.create_plant(initial_regolith_kg=2000.0)
        results = mc.run_simulation(state, sols=5)
        for i, r in enumerate(results):
            assert r.sol == i + 1

    def test_no_crash_100_sols(self):
        state = mc.create_plant(initial_regolith_kg=5000.0)
        results = mc.run_simulation(state, sols=100)
        assert len(results) == 100

    def test_production_accumulates(self):
        state = mc.create_plant(initial_regolith_kg=5000.0)
        mc.run_simulation(state, sols=50)
        assert state.concrete_produced_kg > 0.0
        assert state.blocks_produced > 0

    def test_blocks_increase_monotonically(self):
        state = mc.create_plant(initial_regolith_kg=10000.0)
        results = mc.run_simulation(state, sols=20)
        blocks_cumulative = 0
        for r in results:
            blocks_cumulative += r.blocks_produced
        assert blocks_cumulative == state.blocks_produced

    def test_energy_accumulates(self):
        state = mc.create_plant(initial_regolith_kg=5000.0)
        results = mc.run_simulation(state, sols=20)
        total_energy = sum(r.energy_consumed_kj for r in results)
        assert abs(state.total_energy_consumed_kj - total_energy) < 1.0


# ── 17. Serialisation ─────────────────────────────────────────────────

class TestSerialisation:
    def test_to_dict(self, plant):
        d = plant.to_dict()
        assert isinstance(d, dict)
        assert "sol" in d
        assert "blocks_produced" in d

    def test_roundtrip(self, plant):
        mc.tick(plant)
        d = plant.to_dict()
        s = json.dumps(d)
        loaded = json.loads(s)
        p2 = mc.PlantState.from_dict(loaded)
        assert p2.sol == plant.sol
        assert p2.blocks_produced == plant.blocks_produced

    def test_json_serialisable(self, plant):
        d = plant.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)


# ── 18. Thermal cycling integration ──────────────────────────────────

class TestThermalCyclingIntegration:
    def test_block_strength_degrades_over_mission(self):
        initial = mc.concrete_strength_mpa(0.35)
        after_1yr = mc.thermal_fatigue_strength(initial, 668)
        assert after_1yr < initial
        assert after_1yr > initial * 0.5

    def test_fatigue_life_reasonable(self):
        initial = mc.concrete_strength_mpa(0.35)
        after_life = mc.thermal_fatigue_strength(initial, mc.FATIGUE_LIFE_SOLS)
        assert after_life > 0.0
        assert after_life < initial * 0.5


# ── 19. Conservation and bounds ────────────────────────────────────────

class TestConservationBounds:
    def test_stockpiles_never_negative(self):
        state = mc.create_plant(initial_regolith_kg=100.0)
        mc.run_simulation(state, sols=50, regolith_per_sol_kg=0.0)
        assert state.sulfur_stockpile_kg >= 0.0
        assert state.aggregate_stockpile_kg >= 0.0

    def test_concrete_always_non_negative(self):
        state = mc.create_plant()
        mc.run_simulation(state, sols=10)
        assert state.concrete_produced_kg >= 0.0

    def test_blocks_always_non_negative(self):
        state = mc.create_plant()
        mc.run_simulation(state, sols=10)
        assert state.blocks_produced >= 0

    def test_energy_always_non_negative(self):
        state = mc.create_plant(initial_regolith_kg=1000.0)
        results = mc.run_simulation(state, sols=10)
        for r in results:
            assert r.energy_consumed_kj >= 0.0

    def test_strength_always_bounded(self):
        state = mc.create_plant(initial_regolith_kg=5000.0)
        results = mc.run_simulation(state, sols=20)
        for r in results:
            assert 0.0 <= r.batch_strength_mpa <= mc.PEAK_STRENGTH_MPA + 0.01

    def test_avg_strength_bounded(self):
        state = mc.create_plant(initial_regolith_kg=5000.0)
        mc.run_simulation(state, sols=50)
        assert 0.0 <= state.avg_strength_mpa <= mc.PEAK_STRENGTH_MPA + 0.01


# ── 20. Parametrised invariants ──────────────────────────────────────

class TestParametrised:
    @pytest.mark.parametrize("ratio", [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    def test_strength_non_negative(self, ratio):
        assert mc.concrete_strength_mpa(ratio) >= 0.0

    @pytest.mark.parametrize("mass", [0.0, 1.0, 10.0, 100.0, 1000.0])
    def test_sulfur_extraction_scales(self, mass):
        s = mc.sulfur_from_regolith_kg(mass)
        assert abs(s - mass * mc.REGOLITH_SULFUR_FRACTION) < 0.01

    @pytest.mark.parametrize("mass", [0.0, 1.0, 10.0, 100.0, 1000.0])
    def test_aggregate_extraction_scales(self, mass):
        a = mc.aggregate_from_regolith_kg(mass)
        assert abs(a - mass * mc.REGOLITH_SIEVE_YIELD) < 0.01

    @pytest.mark.parametrize("regolith_kg", [100.0, 500.0, 1000.0, 5000.0])
    def test_plant_preload_variants(self, regolith_kg):
        p = mc.create_plant(initial_regolith_kg=regolith_kg)
        assert p.sulfur_stockpile_kg > 0.0
        assert p.aggregate_stockpile_kg > 0.0

    @pytest.mark.parametrize("sols", [1, 5, 10, 50, 100])
    def test_simulation_length_variants(self, sols):
        state = mc.create_plant(initial_regolith_kg=10000.0)
        results = mc.run_simulation(state, sols=sols)
        assert len(results) == sols

    @pytest.mark.parametrize("temp_k", [150.0, 200.0, 250.0, 293.0])
    def test_ambient_temperature_variants(self, temp_k):
        state = mc.create_plant(initial_regolith_kg=2000.0)
        results = mc.run_simulation(state, sols=5, ambient_k=temp_k)
        assert len(results) == 5

    @pytest.mark.parametrize("porosity", [0.0, 0.05, 0.10, 0.20, 0.30])
    def test_porosity_variants(self, porosity):
        s = mc.concrete_strength_mpa(0.35, porosity=porosity)
        assert s >= 0.0
        assert s <= mc.PEAK_STRENGTH_MPA + 0.01


# ── 21. Edge cases ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_huge_regolith_delivery(self):
        state = mc.create_plant()
        result = mc.tick(state, regolith_delivery_kg=1_000_000.0)
        assert result.batches_run > 0

    def test_tiny_regolith_delivery(self):
        state = mc.create_plant()
        result = mc.tick(state, regolith_delivery_kg=1.0)
        assert result.blocks_produced == 0

    def test_zero_regolith_zero_production(self):
        state = mc.create_plant()
        result = mc.tick(state, regolith_delivery_kg=0.0)
        assert result.concrete_produced_kg == 0.0

    def test_very_cold_ambient(self):
        state = mc.create_plant(initial_regolith_kg=2000.0)
        result = mc.tick(state, ambient_k=100.0)
        assert result.energy_consumed_kj > 0.0

    def test_warm_ambient_less_energy(self):
        p1 = mc.create_plant(initial_regolith_kg=5000.0)
        p2 = mc.create_plant(initial_regolith_kg=5000.0)
        r_cold = mc.tick(p1, ambient_k=150.0)
        r_warm = mc.tick(p2, ambient_k=280.0)
        assert r_cold.energy_consumed_kj > r_warm.energy_consumed_kj

    def test_plant_to_dict_after_production(self, plant):
        mc.tick(plant)
        d = plant.to_dict()
        assert d["blocks_produced"] > 0
        assert d["sol"] == 1


# ── 22. Smoke test — 10-step simulation ──────────────────────────────

class TestSmokeTest:
    def test_smoke_10_sols(self):
        """The simulation runs 10 sols without crashing."""
        state = mc.create_plant(initial_regolith_kg=5000.0)
        results = mc.run_simulation(state, sols=10, regolith_per_sol_kg=200.0)
        assert len(results) == 10
        assert state.blocks_produced > 0
        assert state.concrete_produced_kg > 0.0
        for r in results:
            assert r.batch_strength_mpa >= 0.0
            assert r.energy_consumed_kj >= 0.0

    def test_smoke_colony_year(self):
        """Full Mars year (668 sols) without crash."""
        state = mc.create_plant(initial_regolith_kg=50000.0)
        results = mc.run_simulation(state, sols=668, regolith_per_sol_kg=100.0)
        assert len(results) == 668
        assert state.blocks_produced > 100


# ── 23. Physical plausibility ────────────────────────────────────────

class TestPhysicalPlausibility:
    def test_sulfur_melting_point_correct(self):
        assert abs(mc.SULFUR_MELTING_K - 388.4) < 0.1

    def test_sulfur_enthalpy_correct(self):
        assert abs(mc.SULFUR_ENTHALPY_FUSION_KJ_KG - 54.0) < 1.0

    def test_mars_regolith_sulfur_content(self):
        assert abs(mc.REGOLITH_SULFUR_FRACTION - 0.05) < 0.01

    def test_peak_strength_realistic(self):
        assert 35.0 <= mc.PEAK_STRENGTH_MPA <= 50.0

    def test_energy_per_block_order_of_magnitude(self):
        state = mc.create_plant(initial_regolith_kg=10000.0)
        result = mc.tick(state)
        if result.blocks_produced > 0:
            energy_per_block = result.energy_consumed_kj / result.blocks_produced
            assert 500.0 < energy_per_block < 10000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
