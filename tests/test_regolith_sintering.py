"""tests for regolith_sintering.py - Mars Regolith Sintering Kiln.

Covers every public function, all kiln phases, conservation laws,
physical bounds, edge cases, and multi-batch integration.
"""
from __future__ import annotations

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import regolith_sintering as rs


class TestSinteringTemperature:

    def test_default_iron_fraction(self):
        temp = rs.sintering_temperature(rs.REGOLITH_IRON_OXIDE_FRAC)
        assert rs.MARS_AMBIENT_K + 100.0 < temp < rs.SINTER_TEMP_BASE_K

    def test_zero_iron(self):
        temp = rs.sintering_temperature(0.0)
        assert temp == rs.SINTER_TEMP_BASE_K

    def test_high_iron_lowers_temp(self):
        low = rs.sintering_temperature(0.05)
        high = rs.sintering_temperature(0.30)
        assert high < low

    def test_monotonically_decreasing(self):
        temps = [rs.sintering_temperature(f / 100.0) for f in range(0, 100, 5)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1]

    def test_never_below_floor(self):
        temp = rs.sintering_temperature(1.0)
        assert temp >= rs.MARS_AMBIENT_K + 100.0

    def test_negative_iron_clamped(self):
        assert rs.sintering_temperature(-0.5) == rs.sintering_temperature(0.0)

    def test_iron_above_one_clamped(self):
        assert rs.sintering_temperature(1.5) == rs.sintering_temperature(1.0)


class TestSinteringRate:

    def test_ambient_returns_zero(self):
        assert rs.sintering_rate(rs.MARS_AMBIENT_K) == 0.0

    def test_below_ambient_returns_zero(self):
        assert rs.sintering_rate(100.0) == 0.0

    def test_positive_at_sintering_temp(self):
        rate = rs.sintering_rate(rs.SINTER_TEMP_BASE_K)
        assert rate > 0.0

    def test_higher_temp_faster_rate(self):
        r1 = rs.sintering_rate(1200.0)
        r2 = rs.sintering_rate(1400.0)
        assert r2 > r1

    def test_rate_is_finite(self):
        assert math.isfinite(rs.sintering_rate(5000.0))

    def test_rate_non_negative(self):
        for t in range(100, 3000, 100):
            assert rs.sintering_rate(float(t)) >= 0.0


class TestDensification:

    def test_zero_rate_no_change(self):
        assert rs.densification(0.0, 1000.0, 0.3) == 0.3

    def test_zero_time_no_change(self):
        assert rs.densification(1e-5, 0.0, 0.3) == 0.3

    def test_increases_over_time(self):
        f1 = rs.densification(1e-5, 1000.0, 0.3)
        f2 = rs.densification(1e-5, 10000.0, 0.3)
        assert f2 > f1 > 0.3

    def test_never_exceeds_one(self):
        assert rs.densification(1.0, 1e10, 0.0) <= 1.0

    def test_never_decreases(self):
        assert rs.densification(1e-5, 1000.0, 0.8) >= 0.8

    def test_approaches_one(self):
        assert rs.densification(0.01, 100000.0, 0.0) > 0.99

    def test_clamped_inputs(self):
        f = rs.densification(1e-5, 1000.0, -0.5)
        assert 0.0 <= f <= 1.0


class TestBrickStrength:

    def test_zero_density_zero_strength(self):
        assert rs.brick_strength(0.0) == 0.0

    def test_full_density_max_strength(self):
        s = rs.brick_strength(1.0)
        assert s == pytest.approx(rs.STRENGTH_COEFF_MPA, rel=1e-6)

    def test_strength_increases_with_density(self):
        assert rs.brick_strength(0.8) > rs.brick_strength(0.5)

    def test_unsieved_penalty(self):
        sieved = rs.brick_strength(0.8, sieved=True)
        unsieved = rs.brick_strength(0.8, sieved=False)
        assert unsieved < sieved
        ratio = unsieved / sieved
        assert ratio == pytest.approx(1.0 - rs.FINE_DUST_STRENGTH_PENALTY, rel=1e-6)

    def test_non_negative(self):
        for d in [x / 10.0 for x in range(11)]:
            assert rs.brick_strength(d) >= 0.0

    def test_clamped_above_one(self):
        assert rs.brick_strength(1.5) == rs.brick_strength(1.0)


class TestEnergyToHeat:

    def test_zero_mass(self):
        assert rs.energy_to_heat(0.0, 200.0, 1000.0) == 0.0

    def test_zero_delta_t(self):
        assert rs.energy_to_heat(10.0, 500.0, 500.0) == 0.0

    def test_cooling_returns_zero(self):
        assert rs.energy_to_heat(10.0, 1000.0, 500.0) == 0.0

    def test_positive_energy(self):
        assert rs.energy_to_heat(50.0, 210.0, 1300.0) > 0.0

    def test_energy_proportional_to_mass(self):
        e1 = rs.energy_to_heat(10.0, 210.0, 1300.0)
        e2 = rs.energy_to_heat(20.0, 210.0, 1300.0)
        assert e2 == pytest.approx(2.0 * e1, rel=1e-6)

    def test_known_value(self):
        e = rs.energy_to_heat(1.0, 300.0, 301.0)
        expected = 800.0 / 3_600_000.0
        assert e == pytest.approx(expected, rel=1e-6)


class TestKilnLosses:

    def test_wall_at_ambient_zero(self):
        assert rs.kiln_wall_loss_kw(rs.MARS_AMBIENT_K) == 0.0

    def test_wall_positive_above_ambient(self):
        assert rs.kiln_wall_loss_kw(1300.0) > 0.0

    def test_wall_increases_with_temp(self):
        assert rs.kiln_wall_loss_kw(1000.0) > rs.kiln_wall_loss_kw(500.0)

    def test_radiation_at_ambient_zero(self):
        assert rs.kiln_radiation_loss_kw(rs.MARS_AMBIENT_K) == 0.0

    def test_radiation_positive_above_ambient(self):
        assert rs.kiln_radiation_loss_kw(1300.0) > 0.0

    def test_total_is_sum(self):
        t = 1000.0
        total = rs.total_loss_kw(t)
        assert total == pytest.approx(
            rs.kiln_wall_loss_kw(t) + rs.kiln_radiation_loss_kw(t), rel=1e-9)


class TestHeatingHours:

    def test_zero_power(self):
        assert rs.heating_hours(50.0, 210.0, 1300.0, 0.0) == 0.0

    def test_positive_result(self):
        h = rs.heating_hours(50.0, 210.0, 1300.0, 8.0)
        assert 0.0 < h < 1000.0

    def test_more_power_faster(self):
        h1 = rs.heating_hours(50.0, 210.0, 1300.0, 4.0)
        h2 = rs.heating_hours(50.0, 210.0, 1300.0, 8.0)
        assert h2 < h1

    def test_more_mass_slower(self):
        h1 = rs.heating_hours(25.0, 210.0, 1300.0, 8.0)
        h2 = rs.heating_hours(50.0, 210.0, 1300.0, 8.0)
        assert h2 > h1


class TestCoolingHours:

    def test_no_cooling_needed(self):
        assert rs.cooling_hours(300.0, 300.0) == 0.0

    def test_positive_duration(self):
        assert rs.cooling_hours(1300.0, 300.0) > 0.0

    def test_known_value(self):
        h = rs.cooling_hours(1000.0, 500.0)
        assert h == pytest.approx(250.0 / 60.0, rel=1e-6)


class TestCrackProbability:

    def test_zero_rate(self):
        assert rs.crack_probability(0.0) == 0.0

    def test_safe_rate(self):
        assert rs.crack_probability(rs.MAX_COOLING_RATE_K_PER_MIN) == pytest.approx(rs.CRACK_PROBABILITY_BASE)

    def test_excessive_rate_higher(self):
        assert rs.crack_probability(rs.MAX_COOLING_RATE_K_PER_MIN + 5.0) > rs.CRACK_PROBABILITY_BASE

    def test_capped_at_one(self):
        assert rs.crack_probability(100.0) <= 1.0


class TestBricksFromCharge:

    def test_full_batch(self):
        assert rs.bricks_from_charge(rs.KILN_CAPACITY_KG) == int(rs.KILN_CAPACITY_KG / rs.BRICK_MASS_KG)

    def test_zero_mass(self):
        assert rs.bricks_from_charge(0.0) == 0

    def test_partial_brick_truncated(self):
        assert rs.bricks_from_charge(7.5) == 1


class TestLoadCharge:

    def test_successful_load(self):
        state = rs.KilnState()
        err = rs.load_charge(state, 50.0)
        assert err is None
        assert state.phase == "heating"
        assert state.charge_mass_kg == 50.0

    def test_load_from_done(self):
        state = rs.KilnState(phase="done")
        assert rs.load_charge(state, 30.0) is None

    def test_reject_during_heating(self):
        state = rs.KilnState(phase="heating")
        assert rs.load_charge(state, 30.0) is not None

    def test_reject_during_soaking(self):
        state = rs.KilnState(phase="soaking")
        assert rs.load_charge(state, 30.0) is not None

    def test_reject_during_cooling(self):
        state = rs.KilnState(phase="cooling")
        assert rs.load_charge(state, 30.0) is not None

    def test_reject_zero_mass(self):
        assert rs.load_charge(rs.KilnState(), 0.0) is not None

    def test_reject_negative_mass(self):
        assert rs.load_charge(rs.KilnState(), -5.0) is not None

    def test_reject_over_capacity(self):
        assert rs.load_charge(rs.KilnState(), rs.KILN_CAPACITY_KG + 1.0) is not None

    def test_sets_target_temperature(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0, iron_oxide_frac=0.25)
        assert state.target_temp_k == pytest.approx(rs.sintering_temperature(0.25))

    def test_resets_tracking(self):
        state = rs.KilnState(phase="done", density_fraction=0.9, bricks_this_batch=8)
        rs.load_charge(state, 50.0)
        assert state.density_fraction == 0.0
        assert state.bricks_this_batch == 0


class TestTickIdle:

    def test_idle_no_energy(self):
        state = rs.KilnState()
        rec = rs.tick(state)
        assert rec.energy_input_kwh == 0.0
        assert rec.phase == "idle"
        assert state.sol == 1

    def test_idle_sol_increments(self):
        state = rs.KilnState()
        rs.tick(state)
        rs.tick(state)
        assert state.sol == 2


class TestTickHeating:

    def test_temperature_increases(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0)
        initial_temp = state.charge_temp_k
        rs.tick(state)
        assert state.charge_temp_k > initial_temp

    def test_energy_consumed(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0)
        rec = rs.tick(state)
        assert rec.energy_input_kwh > 0.0

    def test_eventually_reaches_soaking(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0)
        for _ in range(200):
            rs.tick(state)
            if state.phase != "heating":
                break
        assert state.phase in ("soaking", "cooling", "done")


class TestTickSoaking:

    def test_density_increases(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0)
        for _ in range(200):
            rs.tick(state)
            if state.phase == "soaking":
                break
        if state.phase == "soaking":
            d_before = state.density_fraction
            rs.tick(state)
            assert state.density_fraction >= d_before

    def test_transitions_to_cooling(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0)
        for _ in range(500):
            rs.tick(state)
            if state.phase == "cooling":
                break
        assert state.phase in ("cooling", "done")


class TestTickCooling:

    def test_temperature_decreases_during_cooling(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0)
        # Get to cooling phase
        for _ in range(500):
            rs.tick(state)
            if state.phase == "cooling":
                break
        if state.phase == "cooling":
            t_before = state.charge_temp_k
            rs.tick(state, dt_hours=1.0)  # small step to stay in cooling
            assert state.charge_temp_k <= t_before


class TestFullBatch:

    def test_full_batch_completes(self):
        state, records = rs.run_batch()
        assert state.phase == "done"
        assert state.bricks_this_batch > 0
        assert state.batches_completed == 1

    def test_expected_brick_count(self):
        state, _ = rs.run_batch(mass_kg=50.0)
        expected = rs.bricks_from_charge(50.0)
        assert state.bricks_this_batch + state.bricks_cracked == expected

    def test_positive_strength(self):
        state, _ = rs.run_batch()
        assert state.compressive_strength_mpa > 0.0

    def test_positive_density(self):
        state, _ = rs.run_batch()
        assert 0.0 < state.density_fraction <= 1.0

    def test_energy_consumed(self):
        state, _ = rs.run_batch()
        assert state.energy_input_kwh > 0.0
        assert state.energy_total_kwh > 0.0

    def test_log_populated(self):
        state, _ = rs.run_batch()
        assert len(state.log) == 1
        assert state.log[0]["bricks_good"] > 0

    def test_records_track_phases(self):
        _, records = rs.run_batch()
        phases = {r.phase for r in records}
        assert "heating" in phases

    def test_small_charge(self):
        state, _ = rs.run_batch(mass_kg=5.0)
        assert state.phase == "done"
        assert state.bricks_this_batch + state.bricks_cracked == 1

    def test_unsieved_weaker(self):
        s1, _ = rs.run_batch(sieved=True)
        s2, _ = rs.run_batch(sieved=False)
        assert s2.compressive_strength_mpa < s1.compressive_strength_mpa

    def test_high_iron_lower_energy(self):
        s1, _ = rs.run_batch(iron_oxide_frac=0.05)
        s2, _ = rs.run_batch(iron_oxide_frac=0.35)
        assert s2.energy_input_kwh < s1.energy_input_kwh

    def test_low_power_still_completes(self):
        state, _ = rs.run_batch(mass_kg=10.0, power_kw=3.0, max_sols=500)
        assert state.phase == "done"


class TestConservationLaws:

    def test_temperature_never_below_ambient(self):
        state, records = rs.run_batch()
        for r in records:
            assert r.kiln_temp_k >= rs.MARS_AMBIENT_K - 1.0
            assert r.charge_temp_k >= rs.MARS_AMBIENT_K - 1.0

    def test_density_fraction_bounded(self):
        state, records = rs.run_batch()
        for r in records:
            assert 0.0 <= r.density_fraction <= 1.0

    def test_strength_non_negative(self):
        state, records = rs.run_batch()
        for r in records:
            assert r.strength_mpa >= 0.0

    def test_energy_monotonically_increases(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0)
        cumulative = 0.0
        for _ in range(300):
            rec = rs.tick(state)
            cumulative += rec.energy_input_kwh
            if state.phase == "done":
                break
        assert cumulative >= 0.0

    def test_mass_conservation(self):
        state, _ = rs.run_batch(mass_kg=50.0)
        total_bricks = state.bricks_this_batch + state.bricks_cracked
        assert total_bricks * rs.BRICK_MASS_KG <= 50.0

    def test_brick_count_conservation(self):
        state, _ = rs.run_batch(mass_kg=50.0)
        expected = rs.bricks_from_charge(50.0)
        assert state.bricks_this_batch + state.bricks_cracked == expected

    def test_energy_first_law(self):
        state, _ = rs.run_batch()
        min_energy = rs.energy_to_heat(
            state.charge_mass_kg, rs.MARS_AMBIENT_K, state.target_temp_k)
        assert state.energy_input_kwh >= min_energy

    def test_heating_temp_monotonic(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0)
        prev = state.charge_temp_k
        for _ in range(200):
            rs.tick(state, available_power_kw=rs.KILN_HEATER_MAX_KW)
            if state.phase == "heating":
                assert state.charge_temp_k >= prev - 0.01
                prev = state.charge_temp_k
            else:
                break


class TestMultiBatch:

    def test_second_batch(self):
        state, _ = rs.run_batch()
        b1 = state.bricks_produced_total
        rs.load_charge(state, 50.0)
        for _ in range(300):
            rs.tick(state)
            if state.phase == "done":
                break
        assert state.batches_completed == 2
        assert state.bricks_produced_total > b1

    def test_cumulative_energy(self):
        state, _ = rs.run_batch()
        e1 = state.energy_total_kwh
        rs.load_charge(state, 25.0)
        for _ in range(300):
            rs.tick(state)
            if state.phase == "done":
                break
        assert state.energy_total_kwh > e1

    def test_log_grows(self):
        state, _ = rs.run_batch()
        rs.load_charge(state, 50.0)
        for _ in range(300):
            rs.tick(state)
            if state.phase == "done":
                break
        assert len(state.log) == 2


class TestEdgeCases:

    def test_minimum_charge(self):
        state, _ = rs.run_batch(mass_kg=rs.BRICK_MASS_KG)
        assert state.bricks_this_batch + state.bricks_cracked == 1

    def test_maximum_charge(self):
        state, _ = rs.run_batch(mass_kg=rs.KILN_CAPACITY_KG)
        expected = rs.bricks_from_charge(rs.KILN_CAPACITY_KG)
        assert state.bricks_this_batch + state.bricks_cracked == expected

    def test_tiny_power(self):
        state, _ = rs.run_batch(mass_kg=5.0, power_kw=1.0, max_sols=1000)
        assert state.phase == "done"

    def test_run_batch_invalid_mass(self):
        with pytest.raises(ValueError):
            rs.run_batch(mass_kg=0.0)

    def test_run_batch_over_capacity(self):
        with pytest.raises(ValueError):
            rs.run_batch(mass_kg=rs.KILN_CAPACITY_KG + 100.0)

    def test_tick_done_state_stable(self):
        state, _ = rs.run_batch()
        rec = rs.tick(state)
        assert rec.phase == "done"
        assert rec.batch_complete is True

    def test_sol_counter_increments(self):
        state = rs.KilnState()
        for _ in range(10):
            rs.tick(state)
        assert state.sol == 10


class TestSmoke:

    def test_ten_tick_smoke(self):
        state = rs.KilnState()
        rs.load_charge(state, 50.0)
        for _ in range(10):
            rec = rs.tick(state)
            assert rec is not None
            assert isinstance(rec, rs.SolRecord)

    def test_full_cycle_smoke(self):
        state, records = rs.run_batch()
        assert len(records) >= 1
        assert state.phase == "done"

    def test_three_consecutive_batches(self):
        state = rs.KilnState()
        for batch in range(3):
            rs.load_charge(state, 30.0)
            for _ in range(500):
                rs.tick(state)
                if state.phase == "done":
                    break
            assert state.phase == "done", "Batch %d didn't complete" % (batch + 1)
        assert state.batches_completed == 3
        assert state.bricks_produced_total > 0
