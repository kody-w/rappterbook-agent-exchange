"""Tests for regolith_brick.py — Mars Regolith Microwave Sintering."""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import regolith_brick as rb
from regolith_brick import (
    BrickKiln,
    TickResult,
    absorbed_power_kw,
    apply_mold_wear,
    brick_mass_kg,
    bricks_per_sol,
    compressive_strength_mpa,
    convective_loss_kw,
    cooling_time_minutes,
    cycle_time_minutes,
    densify_porosity,
    energy_per_brick_kwh,
    heating_time_minutes,
    is_thermally_shocked,
    microwave_absorption_fraction,
    radiation_loss_kw,
    regolith_needed_kg,
    run_simulation,
    sintering_rate,
    temperature_rise_k,
    tick,
)


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def kiln():
    return BrickKiln()


@pytest.fixture
def low_power_kiln():
    return BrickKiln(microwave_power_kw=1.0)


@pytest.fixture
def worn_kiln():
    return BrickKiln(mold_health=0.05)


# ── 1. Microwave absorption ─────────────────────────────────────────────

class TestMicrowaveAbsorption:
    def test_zero_thickness(self):
        assert microwave_absorption_fraction(0.0) == 0.0

    def test_positive_thickness(self):
        frac = microwave_absorption_fraction(0.10)
        expected = 1.0 - math.exp(-20.0 * 0.10)
        assert abs(frac - expected) < 1e-9

    def test_thick_material_near_one(self):
        assert microwave_absorption_fraction(1.0) > 0.99

    def test_monotonic(self):
        assert (microwave_absorption_fraction(0.05)
                < microwave_absorption_fraction(0.10)
                < microwave_absorption_fraction(0.20))

    def test_bounded_zero_one(self):
        for t in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
            f = microwave_absorption_fraction(t)
            assert 0.0 <= f <= 1.0

    def test_negative_thickness_clamped(self):
        assert microwave_absorption_fraction(-1.0) == 0.0

    def test_negative_alpha_clamped(self):
        assert microwave_absorption_fraction(0.1, alpha=-5.0) == 0.0


class TestAbsorbedPower:
    def test_basic(self):
        p = absorbed_power_kw(5.0, 0.10)
        assert p > 0

    def test_zero_power(self):
        assert absorbed_power_kw(0.0, 0.10) == 0.0

    def test_zero_thickness(self):
        assert absorbed_power_kw(5.0, 0.0) == 0.0

    def test_efficiency_bounds(self):
        p_high = absorbed_power_kw(5.0, 0.10, 1.0)
        p_low = absorbed_power_kw(5.0, 0.10, 0.5)
        assert p_high > p_low

    def test_never_exceeds_input(self):
        for pw in [1.0, 5.0, 10.0, 50.0]:
            assert absorbed_power_kw(pw, 0.10) <= pw

    def test_negative_power_clamped(self):
        assert absorbed_power_kw(-5.0, 0.10) == 0.0


# ── 2. Temperature rise ─────────────────────────────────────────────────

class TestTemperatureRise:
    def test_basic(self):
        dt = temperature_rise_k(1.0, 100.0, 1.0)
        expected = (1.0 * 1000.0 * 100.0) / (1.0 * 800.0)
        assert abs(dt - expected) < 0.01

    def test_zero_power(self):
        assert temperature_rise_k(0.0, 100.0, 1.0) == 0.0

    def test_zero_time(self):
        assert temperature_rise_k(1.0, 0.0, 1.0) == 0.0

    def test_zero_mass(self):
        assert temperature_rise_k(1.0, 100.0, 0.0) == 0.0

    def test_proportional_to_power(self):
        dt1 = temperature_rise_k(1.0, 100.0, 1.0)
        dt2 = temperature_rise_k(2.0, 100.0, 1.0)
        assert abs(dt2 / dt1 - 2.0) < 0.01

    def test_inversely_proportional_to_mass(self):
        dt1 = temperature_rise_k(1.0, 100.0, 1.0)
        dt2 = temperature_rise_k(1.0, 100.0, 2.0)
        assert abs(dt1 / dt2 - 2.0) < 0.01


# ── 3. Thermal losses ───────────────────────────────────────────────────

class TestRadiationLoss:
    def test_at_ambient(self):
        assert radiation_loss_kw(rb.MARS_AMBIENT_TEMP_K) == pytest.approx(0.0, abs=0.001)

    def test_above_ambient(self):
        assert radiation_loss_kw(1373.0) > 0.0

    def test_monotonic(self):
        assert radiation_loss_kw(1373.0) > radiation_loss_kw(1000.0)

    def test_non_negative(self):
        for t in [0.0, 100.0, 210.0, 500.0, 1000.0, 2000.0]:
            assert radiation_loss_kw(t) >= 0.0

    def test_t4_scaling(self):
        r1 = radiation_loss_kw(1000.0, ambient_temp_k=0.0)
        r2 = radiation_loss_kw(2000.0, ambient_temp_k=0.0)
        assert abs(r2 / r1 - 16.0) < 0.1  # (2000/1000)^4 = 16


class TestConvectiveLoss:
    def test_at_ambient(self):
        assert convective_loss_kw(rb.MARS_AMBIENT_TEMP_K) == pytest.approx(0.0, abs=1e-9)

    def test_above_ambient(self):
        assert convective_loss_kw(500.0) > 0.0

    def test_linear_with_delta_t(self):
        q1 = convective_loss_kw(310.0)
        q2 = convective_loss_kw(410.0)
        assert abs((q2 / q1) - 2.0) < 0.01

    def test_non_negative(self):
        for t in [0.0, 100.0, 210.0, 500.0, 1500.0]:
            assert convective_loss_kw(t) >= 0.0


# ── 4. Sintering kinetics ───────────────────────────────────────────────

class TestSinteringRate:
    def test_cold_negligible(self):
        assert sintering_rate(300.0) < 1e-20

    def test_hot_significant(self):
        assert sintering_rate(1373.0) > 0.0

    def test_monotonic(self):
        assert sintering_rate(1373.0) > sintering_rate(1000.0)

    def test_zero_temp(self):
        assert sintering_rate(0.0) == 0.0

    def test_negative_temp(self):
        assert sintering_rate(-100.0) == 0.0

    def test_arrhenius_positive(self):
        for t in [500.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0]:
            assert sintering_rate(t) >= 0.0


class TestDensifyPorosity:
    def test_no_change_at_cold(self):
        p = densify_porosity(0.40, 300.0, 3600.0)
        assert abs(p - 0.40) < 0.01

    def test_reduces_at_sintering_temp(self):
        p = densify_porosity(0.40, 1373.0, 7200.0)
        assert p < 0.40

    def test_bounded_above(self):
        assert densify_porosity(0.40, 1373.0, 7200.0) <= 0.40

    def test_bounded_below(self):
        assert densify_porosity(0.40, 1373.0, 7200.0) >= rb.MIN_POROSITY

    def test_zero_time(self):
        assert densify_porosity(0.40, 1373.0, 0.0) == 0.40

    def test_longer_time_more_dense(self):
        p1 = densify_porosity(0.40, 1373.0, 3600.0)
        p2 = densify_porosity(0.40, 1373.0, 7200.0)
        assert p2 <= p1

    def test_already_at_minimum(self):
        p = densify_porosity(rb.MIN_POROSITY, 1373.0, 7200.0)
        assert p == pytest.approx(rb.MIN_POROSITY, abs=1e-9)

    def test_porosity_clamped_above_one(self):
        p = densify_porosity(1.5, 1373.0, 100.0)
        assert p <= 1.0


# ── 5. Compressive strength (Ryshkewitch) ───────────────────────────────

class TestCompressiveStrength:
    def test_fully_dense(self):
        s = compressive_strength_mpa(0.0)
        assert abs(s - rb.RYSHKEWITCH_SIGMA_0_MPA) < 0.01

    def test_initial_porosity(self):
        s = compressive_strength_mpa(0.40)
        expected = 80.0 * math.exp(-4.0 * 0.40)
        assert abs(s - expected) < 0.01

    def test_monotonic_decreasing(self):
        strengths = [compressive_strength_mpa(p) for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]
        for i in range(len(strengths) - 1):
            assert strengths[i] > strengths[i + 1]

    def test_non_negative(self):
        for p in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            assert compressive_strength_mpa(p) >= 0.0

    def test_structural_grade_achievable(self):
        # At the sintering porosity level, strength should exceed 10 MPa
        p = densify_porosity(0.40, 1373.0, 7200.0)
        assert compressive_strength_mpa(p) > rb.MIN_STRENGTH_MPA_STRUCTURAL


# ── 6. Brick mass and geometry ───────────────────────────────────────────

class TestBrickMass:
    def test_fully_dense(self):
        m = brick_mass_kg(0.0)
        expected = 2800.0 * 0.002
        assert abs(m - expected) < 0.01

    def test_initial_porosity(self):
        m = brick_mass_kg(0.40)
        expected = 2800.0 * 0.60 * 0.002
        assert abs(m - expected) < 0.01

    def test_mass_decreases_with_porosity(self):
        assert brick_mass_kg(0.0) > brick_mass_kg(0.20) > brick_mass_kg(0.40)

    def test_non_negative(self):
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert brick_mass_kg(p) >= 0.0

    def test_regolith_needed_equals_mass(self):
        assert abs(regolith_needed_kg(0.40) - brick_mass_kg(0.40)) < 1e-9


# ── 7. Timing ────────────────────────────────────────────────────────────

class TestTiming:
    def test_heating_time_positive(self):
        assert heating_time_minutes(1373.0) > 0.0

    def test_heating_time_proportional(self):
        t1 = heating_time_minutes(600.0)
        t2 = heating_time_minutes(1200.0)
        # ratio of delta-T
        r1 = (600.0 - 210.0) / (1200.0 - 210.0)
        assert abs(t1 / t2 - r1) < 0.01

    def test_cooling_time_positive(self):
        assert cooling_time_minutes(1373.0) > 0.0

    def test_cycle_time_positive(self):
        assert cycle_time_minutes() > 0.0

    def test_cycle_time_components(self):
        ct = cycle_time_minutes()
        heat = heating_time_minutes(rb.SINTERING_TARGET_TEMP_K)
        soak = rb.SOAK_TIME_HOURS * 60.0
        cool = cooling_time_minutes(rb.SINTERING_TARGET_TEMP_K)
        assert abs(ct - (heat + soak + cool)) < 0.01

    def test_bricks_per_sol_positive(self):
        assert bricks_per_sol() >= 1

    def test_bricks_per_sol_bounded(self):
        # Cannot exceed minutes_per_sol / cycle_time
        ct = cycle_time_minutes()
        theoretical_max = rb.MINUTES_PER_SOL / ct
        assert bricks_per_sol() <= theoretical_max + 1


# ── 8. Thermal shock ─────────────────────────────────────────────────────

class TestThermalShock:
    def test_safe_rate(self):
        assert not is_thermally_shocked(30.0)

    def test_dangerous_rate(self):
        assert is_thermally_shocked(100.0)

    def test_threshold_exact(self):
        assert not is_thermally_shocked(50.0)
        assert is_thermally_shocked(50.1)


# ── 9. Energy accounting ─────────────────────────────────────────────────

class TestEnergy:
    def test_energy_positive(self):
        assert energy_per_brick_kwh(5.0) > 0.0

    def test_energy_proportional_to_power(self):
        e1 = energy_per_brick_kwh(5.0)
        e2 = energy_per_brick_kwh(10.0)
        assert abs(e2 / e1 - 2.0) < 0.01

    def test_energy_zero_power(self):
        assert energy_per_brick_kwh(0.0) == 0.0


# ── 10. Mold wear ────────────────────────────────────────────────────────

class TestMoldWear:
    def test_single_cycle(self):
        h = apply_mold_wear(1.0, 1)
        assert abs(h - 0.999) < 1e-9

    def test_multiple_cycles(self):
        h = apply_mold_wear(1.0, 100)
        assert abs(h - 0.90) < 1e-9

    def test_never_negative(self):
        assert apply_mold_wear(0.01, 100) == 0.0

    def test_zero_cycles(self):
        assert apply_mold_wear(0.5, 0) == 0.5


# ── 11. State serialisation ──────────────────────────────────────────────

class TestSerialisation:
    def test_round_trip(self):
        s = BrickKiln(microwave_power_kw=8.0, mold_health=0.75)
        d = s.to_dict()
        s2 = BrickKiln.from_dict(d)
        assert s2.microwave_power_kw == 8.0
        assert s2.mold_health == 0.75

    def test_round_trip_after_tick(self):
        s = BrickKiln()
        tick(s)
        s2 = BrickKiln.from_dict(s.to_dict())
        assert s2.sol == 1
        assert s2.total_bricks > 0

    def test_extra_keys_ignored(self):
        d = BrickKiln().to_dict()
        d["unknown_field"] = 42
        s = BrickKiln.from_dict(d)
        assert s.sol == 0


# ── 12. Tick integration ─────────────────────────────────────────────────

class TestTick:
    def test_produces_bricks(self, kiln):
        r = tick(kiln)
        assert r.bricks_produced > 0
        assert r.operational

    def test_sol_advances(self, kiln):
        tick(kiln)
        assert kiln.sol == 1

    def test_regolith_consumed(self, kiln):
        r = tick(kiln)
        assert r.regolith_consumed_kg > 0

    def test_energy_consumed(self, kiln):
        r = tick(kiln)
        assert r.energy_consumed_kwh > 0

    def test_mold_wears(self, kiln):
        r = tick(kiln)
        assert r.mold_health < 1.0

    def test_cumulative_bricks(self, kiln):
        tick(kiln)
        tick(kiln)
        assert kiln.total_bricks > 0
        assert kiln.sol == 2

    def test_structural_grade(self, kiln):
        r = tick(kiln)
        assert r.structural_bricks > 0

    def test_avg_porosity_reduced(self, kiln):
        r = tick(kiln)
        assert r.avg_porosity < rb.INITIAL_POROSITY

    def test_strength_positive(self, kiln):
        r = tick(kiln)
        assert r.avg_strength_mpa > 0.0

    def test_worn_mold_stops(self, worn_kiln):
        r = tick(worn_kiln)
        assert not r.operational
        assert "MOLD WORN" in " ".join(r.events)

    def test_no_power_stops(self):
        r = tick(BrickKiln(microwave_power_kw=0.0))
        assert not r.operational
        assert "NO POWER" in " ".join(r.events)

    def test_no_feed_stops(self):
        r = tick(BrickKiln(regolith_feed_kg_per_sol=0.0))
        assert not r.operational
        assert "NO FEEDSTOCK" in " ".join(r.events)


# ── 13. Conservation laws ────────────────────────────────────────────────

class TestConservation:
    def test_mass_balance(self, kiln):
        """Regolith consumed matches bricks produced by mass."""
        r = tick(kiln)
        expected_mass = r.bricks_produced * brick_mass_kg(rb.INITIAL_POROSITY)
        assert abs(r.regolith_consumed_kg - expected_mass) < 1e-6

    def test_energy_non_negative(self, kiln):
        r = tick(kiln)
        assert r.energy_consumed_kwh >= 0.0

    def test_porosity_in_bounds(self, kiln):
        r = tick(kiln)
        assert rb.MIN_POROSITY <= r.avg_porosity <= rb.INITIAL_POROSITY

    def test_strength_non_negative(self, kiln):
        r = tick(kiln)
        assert r.avg_strength_mpa >= 0.0

    def test_mold_health_bounded(self, kiln):
        for _ in range(100):
            tick(kiln)
        assert 0.0 <= kiln.mold_health <= 1.0

    def test_bricks_add_up(self, kiln):
        results = [tick(kiln) for _ in range(10)]
        total = sum(r.bricks_produced for r in results)
        assert kiln.total_bricks == total

    def test_structural_plus_cracked_equals_total(self, kiln):
        r = tick(kiln)
        # structural + non-structural = total (cracked is subset of non-structural)
        assert r.structural_bricks + r.cracked_bricks <= r.bricks_produced

    def test_radiation_loss_bounded(self, kiln):
        r = tick(kiln)
        assert r.radiation_loss_kw >= 0.0


# ── 14. Parametrized tests ───────────────────────────────────────────────

class TestParametrized:
    @pytest.mark.parametrize("power", [1.0, 2.5, 5.0, 10.0, 20.0])
    def test_energy_scales_with_power(self, power):
        e = energy_per_brick_kwh(power)
        e_ref = energy_per_brick_kwh(5.0)
        assert abs(e / e_ref - power / 5.0) < 0.01

    @pytest.mark.parametrize("porosity", [0.0, 0.05, 0.10, 0.20, 0.30, 0.40])
    def test_strength_positive(self, porosity):
        assert compressive_strength_mpa(porosity) > 0.0

    @pytest.mark.parametrize("thickness", [0.01, 0.05, 0.10, 0.15, 0.20, 0.50])
    def test_absorption_bounded(self, thickness):
        f = microwave_absorption_fraction(thickness)
        assert 0.0 <= f <= 1.0

    @pytest.mark.parametrize("temp", [500.0, 800.0, 1000.0, 1200.0, 1373.0, 1500.0])
    def test_radiation_positive(self, temp):
        assert radiation_loss_kw(temp) >= 0.0

    @pytest.mark.parametrize("feed", [10.0, 50.0, 100.0, 200.0, 500.0])
    def test_feed_rate(self, feed):
        r = tick(BrickKiln(regolith_feed_kg_per_sol=feed))
        assert r.bricks_produced >= 0
        assert r.regolith_consumed_kg <= feed + 1e-6

    @pytest.mark.parametrize("health", [0.05, 0.10, 0.50, 0.75, 1.0])
    def test_mold_health_bounded(self, health):
        r = tick(BrickKiln(mold_health=health))
        if health <= rb.MOLD_REPLACEMENT_THRESHOLD:
            assert not r.operational
        else:
            assert r.mold_health <= health

    @pytest.mark.parametrize("sols", [1, 10, 50, 100, 365])
    def test_sim_length(self, sols):
        assert len(run_simulation(sols)) == sols


# ── 15. Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_tiny_feed(self):
        """Feed less than one brick's worth."""
        mass = brick_mass_kg(rb.INITIAL_POROSITY)
        r = tick(BrickKiln(regolith_feed_kg_per_sol=mass * 0.5))
        assert r.bricks_produced == 0

    def test_exact_one_brick_feed(self):
        mass = brick_mass_kg(rb.INITIAL_POROSITY)
        r = tick(BrickKiln(regolith_feed_kg_per_sol=mass))
        assert r.bricks_produced >= 1

    def test_very_high_power(self):
        r = tick(BrickKiln(microwave_power_kw=100.0))
        assert r.operational and r.bricks_produced > 0

    def test_low_efficiency(self):
        r = tick(BrickKiln(magnetron_efficiency=0.1))
        assert r.operational

    def test_full_efficiency(self):
        r = tick(BrickKiln(magnetron_efficiency=1.0))
        assert r.operational

    def test_very_high_target_temp(self):
        r = tick(BrickKiln(sintering_target_k=2000.0))
        assert r.operational  # fewer bricks due to longer cycle but still works

    def test_low_target_temp(self):
        """Low sintering temp: bricks weak but produced."""
        r = tick(BrickKiln(sintering_target_k=800.0))
        assert r.bricks_produced > 0

    def test_mold_degradation_over_year(self):
        """Mold doesn't last a full Mars year without replacement."""
        results = run_simulation(668)
        # Eventually the mold gives out
        last_operational = [r for r in results if r.operational]
        assert len(last_operational) < 668


# ── 16. Smoke tests ──────────────────────────────────────────────────────

class TestSmokeTest:
    def test_10_sol(self):
        results = run_simulation(10)
        assert len(results) == 10
        for r in results:
            assert r.sol > 0
            assert r.bricks_produced >= 0
            assert 0.0 <= r.mold_health <= 1.0

    def test_mars_year(self):
        results = run_simulation(668)
        assert len(results) == 668
        total = sum(r.bricks_produced for r in results)
        assert total > 0

    def test_structural_yield(self):
        results = run_simulation(100)
        structural = sum(r.structural_bricks for r in results)
        total = sum(r.bricks_produced for r in results)
        assert structural > 0
        assert structural <= total

    def test_realistic_daily_output(self):
        r = tick(BrickKiln())
        # At 5 kW and ~6h cycle, should get a few bricks per sol
        assert 1 <= r.bricks_produced <= 20

    def test_realistic_strength(self):
        r = tick(BrickKiln())
        # Sintered regolith should be 15-60 MPa range
        assert 5.0 < r.avg_strength_mpa < 80.0

    def test_energy_reasonable(self):
        r = tick(BrickKiln())
        # ~2-30 kWh per sol for a small kiln
        assert 0.1 < r.energy_consumed_kwh < 500.0

    def test_regolith_consumption_reasonable(self):
        r = tick(BrickKiln())
        # A few kg per brick, a few bricks per sol
        assert 1.0 < r.regolith_consumed_kg < 100.0

    def test_mass_balance_multi_sol(self):
        results = run_simulation(100)
        total_regolith = sum(r.regolith_consumed_kg for r in results)
        total_bricks = sum(r.bricks_produced for r in results)
        mass_per = brick_mass_kg(rb.INITIAL_POROSITY)
        expected = total_bricks * mass_per
        assert abs(total_regolith - expected) < 1e-3

    def test_no_crash_extreme_params(self):
        """Kiln doesn't crash with extreme parameters."""
        configs = [
            {"microwave_power_kw": 0.001},
            {"microwave_power_kw": 1000.0},
            {"regolith_feed_kg_per_sol": 0.001},
            {"regolith_feed_kg_per_sol": 10000.0},
            {"sintering_target_k": 500.0},
            {"sintering_target_k": 3000.0},
            {"magnetron_efficiency": 0.01},
        ]
        for cfg in configs:
            r = tick(BrickKiln(**cfg))
            assert isinstance(r, TickResult)

    def test_main_entry_point(self):
        """Verify run_simulation works as __main__ would call it."""
        results = run_simulation(sols=5)
        total_bricks = sum(r.bricks_produced for r in results)
        assert total_bricks > 0
        assert results[-1].mold_health > 0
