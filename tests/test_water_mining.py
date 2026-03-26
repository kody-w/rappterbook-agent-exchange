"""
Tests for water_mining.py — Mars ice extraction model.

65 tests. Every function, every edge, every conservation law.
Physics-first: if it violates thermodynamics, it fails.

Run: python -m pytest tests/test_water_mining.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.water_mining import (
    IceDeposit,
    mine_water_sol,
    temperature_efficiency,
    create_colony_deposit,
    ICE_SUBLIMATION_KJ_KG,
    KJ_PER_KWH,
    REGOLITH_DENSITY_KG_M3,
    DRILL_POWER_KWH_PER_M3,
    COLD_EFFICIENCY_FLOOR,
    WARM_EFFICIENCY_CEIL,
)


class TestTemperatureEfficiency:

    def test_cold_floor(self) -> None:
        assert temperature_efficiency(-120.0) == COLD_EFFICIENCY_FLOOR

    def test_warm_ceiling(self) -> None:
        assert temperature_efficiency(0.0) == WARM_EFFICIENCY_CEIL

    def test_midpoint_exact(self) -> None:
        expected = (COLD_EFFICIENCY_FLOOR + WARM_EFFICIENCY_CEIL) / 2.0
        assert abs(temperature_efficiency(-60.0) - expected) < 1e-10

    def test_monotonic_1degree_steps(self) -> None:
        prev = 0.0
        for t in range(-120, 1):
            eff = temperature_efficiency(float(t))
            assert eff >= prev
            prev = eff

    def test_strictly_increasing_in_range(self) -> None:
        for t in range(-119, 1):
            assert temperature_efficiency(float(t)) > temperature_efficiency(float(t - 1))

    def test_bounded_extreme_cold(self) -> None:
        assert temperature_efficiency(-200.0) == COLD_EFFICIENCY_FLOOR
        assert temperature_efficiency(-1000.0) == COLD_EFFICIENCY_FLOOR

    def test_bounded_extreme_warm(self) -> None:
        assert temperature_efficiency(50.0) == WARM_EFFICIENCY_CEIL
        assert temperature_efficiency(500.0) == WARM_EFFICIENCY_CEIL

    def test_linear_quarter(self) -> None:
        quarter = COLD_EFFICIENCY_FLOOR + 0.25 * (WARM_EFFICIENCY_CEIL - COLD_EFFICIENCY_FLOOR)
        assert abs(temperature_efficiency(-90.0) - quarter) < 1e-10

    def test_linear_three_quarter(self) -> None:
        three_q = COLD_EFFICIENCY_FLOOR + 0.75 * (WARM_EFFICIENCY_CEIL - COLD_EFFICIENCY_FLOOR)
        assert abs(temperature_efficiency(-30.0) - three_q) < 1e-10

    def test_mars_mean_temp(self) -> None:
        eff = temperature_efficiency(-60.0)
        assert COLD_EFFICIENCY_FLOOR < eff < WARM_EFFICIENCY_CEIL

    def test_always_positive(self) -> None:
        for t in range(-300, 100, 7):
            assert temperature_efficiency(float(t)) > 0

    def test_return_type_float(self) -> None:
        assert isinstance(temperature_efficiency(-60.0), float)


class TestIceDeposit:

    def test_valid_creation(self) -> None:
        d = IceDeposit(concentration=0.2, depth_m=3.0, reserve_kg=10000.0)
        assert d.concentration == 0.2
        assert d.depth_m == 3.0
        assert d.reserve_kg == 10000.0

    def test_concentration_clamped_high(self) -> None:
        d = IceDeposit(concentration=1.5, depth_m=1.0, reserve_kg=1000)
        assert d.concentration == 1.0

    def test_concentration_clamped_low(self) -> None:
        d = IceDeposit(concentration=-0.5, depth_m=1.0, reserve_kg=1000)
        assert d.concentration == 0.0

    def test_concentration_boundary_zero(self) -> None:
        d = IceDeposit(concentration=0.0, depth_m=1.0, reserve_kg=1000)
        assert d.concentration == 0.0

    def test_concentration_boundary_one(self) -> None:
        d = IceDeposit(concentration=1.0, depth_m=1.0, reserve_kg=1000)
        assert d.concentration == 1.0

    def test_reserve_nonnegative(self) -> None:
        d = IceDeposit(concentration=0.1, depth_m=1.0, reserve_kg=-50)
        assert d.reserve_kg == 0.0

    def test_reserve_zero(self) -> None:
        d = IceDeposit(concentration=0.1, depth_m=1.0, reserve_kg=0.0)
        assert d.reserve_kg == 0.0

    def test_depth_minimum(self) -> None:
        d = IceDeposit(concentration=0.1, depth_m=-1.0, reserve_kg=1000)
        assert d.depth_m >= 0.1

    def test_depth_zero_clamped(self) -> None:
        d = IceDeposit(concentration=0.1, depth_m=0.0, reserve_kg=1000)
        assert d.depth_m == 0.1

    def test_phoenix_regolith(self) -> None:
        d = IceDeposit(concentration=0.04, depth_m=0.3, reserve_kg=5000)
        assert 0.03 <= d.concentration <= 0.05

    def test_sharad_glacier(self) -> None:
        d = IceDeposit(concentration=0.65, depth_m=10.0, reserve_kg=5_000_000)
        assert 0.50 <= d.concentration <= 0.80


class TestMineWaterSolZeroCases:

    def test_zero_power(self) -> None:
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=10000)
        water, power = mine_water_sol(0.0, -60.0, d)
        assert water == 0.0 and power == 0.0

    def test_negative_power(self) -> None:
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=10000)
        water, power = mine_water_sol(-10.0, -60.0, d)
        assert water == 0.0 and power == 0.0

    def test_empty_deposit(self) -> None:
        d = IceDeposit(concentration=0.3, depth_m=1.0, reserve_kg=0.0)
        water, power = mine_water_sol(100.0, -40.0, d)
        assert water == 0.0 and power == 0.0

    def test_broken_drill(self) -> None:
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=10000)
        water, power = mine_water_sol(50.0, -60.0, d, drill_condition=0.0)
        assert water == 0.0

    def test_negative_drill_condition(self) -> None:
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=10000)
        water, power = mine_water_sol(50.0, -60.0, d, drill_condition=-0.5)
        assert water == 0.0

    def test_zero_concentration_deposit(self) -> None:
        d = IceDeposit(concentration=0.0, depth_m=1.0, reserve_kg=10000)
        water, power = mine_water_sol(50.0, -60.0, d)
        assert water == 0.0


class TestMineWaterSolPositive:

    def test_positive_extraction(self) -> None:
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        water, power = mine_water_sol(50.0, -60.0, d)
        assert water > 0 and power > 0

    def test_warmer_more_water(self) -> None:
        d1 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        d2 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        cold_water, _ = mine_water_sol(50.0, -100.0, d1)
        warm_water, _ = mine_water_sol(50.0, -20.0, d2)
        assert warm_water > cold_water

    def test_more_power_more_water(self) -> None:
        d1 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        d2 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        small, _ = mine_water_sol(10.0, -60.0, d1)
        large, _ = mine_water_sol(100.0, -60.0, d2)
        assert large > small

    def test_higher_concentration_more_water(self) -> None:
        d_lean = IceDeposit(concentration=0.05, depth_m=1.0, reserve_kg=100000)
        d_rich = IceDeposit(concentration=0.50, depth_m=1.0, reserve_kg=100000)
        lean_water, _ = mine_water_sol(50.0, -60.0, d_lean)
        rich_water, _ = mine_water_sol(50.0, -60.0, d_rich)
        assert rich_water > lean_water

    def test_degraded_drill_less(self) -> None:
        d1 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        d2 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        fresh, _ = mine_water_sol(50.0, -60.0, d1, drill_condition=1.0)
        worn, _ = mine_water_sol(50.0, -60.0, d2, drill_condition=0.3)
        assert worn < fresh

    def test_drill_condition_above_one_clamped(self) -> None:
        d1 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        d2 = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        normal, _ = mine_water_sol(50.0, -60.0, d1, drill_condition=1.0)
        over, _ = mine_water_sol(50.0, -60.0, d2, drill_condition=5.0)
        assert abs(normal - over) < 1e-6


class TestMineWaterSolConservation:

    def test_never_exceed_reserve(self) -> None:
        d = IceDeposit(concentration=0.5, depth_m=1.0, reserve_kg=1.0)
        water, _ = mine_water_sol(1000.0, 0.0, d)
        assert water <= 1.0

    def test_never_exceed_power(self) -> None:
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=100000)
        _, power = mine_water_sol(10.0, -60.0, d)
        assert power <= 10.0

    def test_reserve_decreases_by_extraction(self) -> None:
        d = IceDeposit(concentration=0.3, depth_m=1.0, reserve_kg=50000)
        initial = d.reserve_kg
        water, _ = mine_water_sol(100.0, -40.0, d)
        assert abs((initial - d.reserve_kg) - water) < 1e-4

    def test_reserve_never_negative(self) -> None:
        d = IceDeposit(concentration=0.8, depth_m=1.0, reserve_kg=10.0)
        mine_water_sol(100000.0, 0.0, d)
        assert d.reserve_kg >= 0.0

    def test_power_limited_regime(self) -> None:
        d = IceDeposit(concentration=0.3, depth_m=1.0, reserve_kg=1e9)
        water, power = mine_water_sol(50.0, -60.0, d)
        assert abs(power - 50.0) < 1e-4

    def test_reserve_limited_uses_less_power(self) -> None:
        d = IceDeposit(concentration=0.5, depth_m=1.0, reserve_kg=0.5)
        _, power = mine_water_sol(1000.0, 0.0, d)
        assert power < 1000.0

    def test_energy_conservation_sweep(self) -> None:
        for power in [1, 10, 50, 100, 500]:
            for conc in [0.01, 0.1, 0.3, 0.6, 1.0]:
                for temp in [-120, -80, -40, 0]:
                    d = IceDeposit(concentration=conc, depth_m=1.0, reserve_kg=1e9)
                    water, consumed = mine_water_sol(float(power), float(temp), d)
                    theoretical_max = power * KJ_PER_KWH / ICE_SUBLIMATION_KJ_KG
                    assert water <= theoretical_max * 1.01
                    assert water >= 0
                    assert consumed >= 0
                    assert consumed <= power + 0.001


class TestMultiSolDepletion:

    def test_ten_sol_smoke(self) -> None:
        d = IceDeposit(concentration=0.15, depth_m=2.0, reserve_kg=50000)
        total_water = 0.0
        for _ in range(10):
            water, _ = mine_water_sol(30.0, -60.0, d)
            total_water += water
            assert d.reserve_kg >= 0
        assert total_water > 0
        assert d.reserve_kg < 50000

    def test_mine_to_depletion(self) -> None:
        initial_reserve = 100.0
        d = IceDeposit(concentration=0.3, depth_m=1.0, reserve_kg=initial_reserve)
        total_water = 0.0
        for _ in range(10000):
            water, _ = mine_water_sol(50.0, -40.0, d)
            total_water += water
            if d.reserve_kg <= 0:
                break
        assert d.reserve_kg == 0.0
        assert abs(total_water - initial_reserve) < 0.1

    def test_depletion_monotonic(self) -> None:
        d = IceDeposit(concentration=0.2, depth_m=1.0, reserve_kg=5000)
        prev_reserve = d.reserve_kg
        for _ in range(50):
            mine_water_sol(20.0, -60.0, d)
            assert d.reserve_kg <= prev_reserve
            prev_reserve = d.reserve_kg

    def test_water_per_sol_drops_near_empty(self) -> None:
        d = IceDeposit(concentration=0.5, depth_m=1.0, reserve_kg=50.0)
        waters = []
        for _ in range(500):
            water, _ = mine_water_sol(100.0, 0.0, d)
            waters.append(water)
            if d.reserve_kg <= 0:
                break
        non_zero = [w for w in waters if w > 0]
        if len(non_zero) >= 2:
            assert non_zero[-1] <= non_zero[0]

    def test_365_sol_year(self) -> None:
        d = IceDeposit(concentration=0.15, depth_m=2.0, reserve_kg=500_000)
        total_water = 0.0
        for sol in range(365):
            t = -55.0 + 25.0 * math.sin(2 * math.pi * sol / 365)
            water, _ = mine_water_sol(40.0, t, d)
            total_water += water
        assert total_water > 0
        assert d.reserve_kg < 500_000
        assert d.reserve_kg > 0

    def test_degrading_drill_over_year(self) -> None:
        d = IceDeposit(concentration=0.2, depth_m=1.5, reserve_kg=200_000)
        total_water = 0.0
        for sol in range(365):
            condition = 1.0 - 0.5 * (sol / 365)
            water, _ = mine_water_sol(30.0, -60.0, d, drill_condition=condition)
            total_water += water
        assert total_water > 0
        assert d.reserve_kg < 200_000


class TestCreateColonyDeposit:

    def test_conservative(self) -> None:
        d = create_colony_deposit("conservative")
        assert d.concentration > 0.2
        assert d.reserve_kg > 400_000
        assert d.depth_m > 3.0

    def test_balanced(self) -> None:
        d = create_colony_deposit("balanced")
        assert 0.05 < d.concentration < 0.5
        assert d.reserve_kg > 100_000

    def test_aggressive(self) -> None:
        d = create_colony_deposit("aggressive")
        assert d.concentration < 0.15
        assert d.depth_m < 1.0

    def test_unknown_defaults_balanced(self) -> None:
        d = create_colony_deposit("unknown")
        balanced = create_colony_deposit("balanced")
        assert d.concentration == balanced.concentration
        assert d.depth_m == balanced.depth_m
        assert d.reserve_kg == balanced.reserve_kg

    def test_empty_string_defaults_balanced(self) -> None:
        d = create_colony_deposit("")
        balanced = create_colony_deposit("balanced")
        assert d.concentration == balanced.concentration

    def test_conservative_richer_than_aggressive(self) -> None:
        c = create_colony_deposit("conservative")
        a = create_colony_deposit("aggressive")
        assert c.concentration > a.concentration
        assert c.reserve_kg > a.reserve_kg

    def test_all_strategies_valid(self) -> None:
        for strat in ["conservative", "balanced", "aggressive"]:
            d = create_colony_deposit(strat)
            assert 0.0 <= d.concentration <= 1.0
            assert d.depth_m >= 0.1
            assert d.reserve_kg >= 0.0

    def test_all_strategies_mineable(self) -> None:
        for strat in ["conservative", "balanced", "aggressive"]:
            d = create_colony_deposit(strat)
            water, _ = mine_water_sol(50.0, -60.0, d)
            assert water > 0


class TestPhysicalRealism:

    def test_sublimation_enthalpy_reasonable(self) -> None:
        assert 2800 <= ICE_SUBLIMATION_KJ_KG <= 2900

    def test_regolith_density_reasonable(self) -> None:
        assert 1200 <= REGOLITH_DENSITY_KG_M3 <= 1800

    def test_phoenix_scenario(self) -> None:
        d = IceDeposit(concentration=0.04, depth_m=0.3, reserve_kg=100_000)
        water, _ = mine_water_sol(20.0, -60.0, d)
        assert 0 < water < 100

    def test_sharad_glacier_scenario(self) -> None:
        d = IceDeposit(concentration=0.65, depth_m=10.0, reserve_kg=5_000_000)
        water, _ = mine_water_sol(100.0, -50.0, d)
        assert water > 10

    def test_daily_human_water_need(self) -> None:
        d = IceDeposit(concentration=0.15, depth_m=2.0, reserve_kg=500_000)
        water, _ = mine_water_sol(100.0, -60.0, d)
        assert water > 0

    def test_output_rounded_to_4dp(self) -> None:
        d = IceDeposit(concentration=0.123, depth_m=1.0, reserve_kg=100000)
        water, power = mine_water_sol(33.33, -47.7, d)
        assert water == round(water, 4)
        assert power == round(power, 4)


class TestPropertySweep:

    def test_water_nonnegative_sweep(self) -> None:
        rng = random.Random(42)
        for _ in range(200):
            power = rng.uniform(-10, 500)
            temp = rng.uniform(-150, 50)
            conc = rng.uniform(-0.5, 1.5)
            reserve = rng.uniform(-100, 1e6)
            drill = rng.uniform(-1, 2)
            d = IceDeposit(concentration=conc, depth_m=1.0, reserve_kg=reserve)
            water, consumed = mine_water_sol(power, temp, d, drill_condition=drill)
            assert water >= 0
            assert consumed >= 0

    def test_power_consumed_bounded_sweep(self) -> None:
        rng = random.Random(99)
        for _ in range(200):
            power = rng.uniform(0, 1000)
            temp = rng.uniform(-120, 0)
            conc = rng.uniform(0.01, 1.0)
            d = IceDeposit(concentration=conc, depth_m=1.0, reserve_kg=1e9)
            _, consumed = mine_water_sol(power, temp, d)
            assert consumed <= power + 0.001

    def test_reserve_accounting_sweep(self) -> None:
        rng = random.Random(77)
        for _ in range(100):
            power = rng.uniform(1, 500)
            temp = rng.uniform(-120, 0)
            conc = rng.uniform(0.01, 1.0)
            reserve = rng.uniform(1, 1e6)
            d = IceDeposit(concentration=conc, depth_m=1.0, reserve_kg=reserve)
            before = d.reserve_kg
            water, _ = mine_water_sol(power, temp, d)
            delta = before - d.reserve_kg
            assert abs(delta - water) < 1e-3
