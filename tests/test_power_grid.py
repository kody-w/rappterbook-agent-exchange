"""
Tests for power_grid.py - Mars colony power distribution. 83 tests.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.power_grid import (
    BATTERY_ROUND_TRIP_EFF, BATTERY_SELF_DISCHARGE_PER_SOL,
    FREQ_DROP_PER_OVERLOAD_FRACTION, GRID_FREQ_MIN_HZ, GRID_FREQ_NOMINAL_HZ,
    NUCLEAR_BASELINE_KWH_SOL, PRIORITY_CONSTRUCTION, PRIORITY_FOOD,
    PRIORITY_LIFE_SUPPORT, PRIORITY_SCIENCE, PRIORITY_THERMAL, PRIORITY_WATER,
    GridBattery, GridState, PowerLoad, allocate_power, colony_loads_default,
    create_grid, grid_frequency, tick_grid, total_allocated, total_curtailed,
    total_demand,
)


class TestConstants:
    def test_nuclear_baseline_plausible(self):
        assert 900 <= NUCLEAR_BASELINE_KWH_SOL <= 1000

    def test_battery_efficiency_below_one(self):
        assert 0.0 < BATTERY_ROUND_TRIP_EFF <= 1.0

    def test_self_discharge_small(self):
        assert 0.0 < BATTERY_SELF_DISCHARGE_PER_SOL < 0.01

    def test_grid_freq_nominal(self):
        assert GRID_FREQ_NOMINAL_HZ == 400.0

    def test_priority_ordering(self):
        assert PRIORITY_LIFE_SUPPORT < PRIORITY_THERMAL < PRIORITY_WATER
        assert PRIORITY_WATER < PRIORITY_FOOD < PRIORITY_CONSTRUCTION < PRIORITY_SCIENCE


class TestPowerLoad:
    def test_basic_load(self):
        ld = PowerLoad("test", 2, 100.0)
        assert ld.name == "test" and ld.priority == 2
        assert ld.requested_kwh == 100.0 and ld.allocated_kwh == 0.0

    def test_shortfall_unallocated(self):
        assert PowerLoad("a", 0, 100.0).shortfall() == 100.0

    def test_shortfall_partial(self):
        assert PowerLoad("a", 0, 100.0, allocated_kwh=60.0).shortfall() == pytest.approx(40.0)

    def test_shortfall_full(self):
        assert PowerLoad("a", 0, 100.0, allocated_kwh=100.0).shortfall() == pytest.approx(0.0)

    def test_fulfillment_zero(self):
        assert PowerLoad("a", 0, 100.0).fulfillment_ratio() == pytest.approx(0.0)

    def test_fulfillment_partial(self):
        assert PowerLoad("a", 0, 100.0, allocated_kwh=75.0).fulfillment_ratio() == pytest.approx(0.75)

    def test_fulfillment_full(self):
        assert PowerLoad("a", 0, 100.0, allocated_kwh=100.0).fulfillment_ratio() == pytest.approx(1.0)

    def test_fulfillment_zero_request(self):
        assert PowerLoad("a", 0, 0.0).fulfillment_ratio() == 1.0

    def test_priority_clamped_low(self):
        assert PowerLoad("a", -1, 100.0).priority == 0

    def test_priority_clamped_high(self):
        assert PowerLoad("a", 99, 100.0).priority == 5

    def test_negative_requested_clamped(self):
        assert PowerLoad("a", 0, -50.0).requested_kwh == 0.0


class TestGridBattery:
    def test_fresh_battery(self):
        b = GridBattery(500.0, 250.0)
        assert b.headroom() == 250.0

    def test_full_battery(self):
        assert GridBattery(500.0, 500.0).headroom() == 0.0

    def test_empty_battery(self):
        assert GridBattery(500.0, 0.0).headroom() == 500.0

    def test_charge_within_capacity(self):
        b = GridBattery(500.0, 200.0)
        stored = b.charge(100.0)
        assert stored > 0 and b.charge_kwh <= 500.0

    def test_charge_respects_efficiency(self):
        b = GridBattery(500.0, 0.0)
        assert b.charge(100.0) == pytest.approx(100.0 * BATTERY_ROUND_TRIP_EFF)

    def test_charge_capped(self):
        b = GridBattery(100.0, 95.0)
        b.charge(100.0)
        assert b.charge_kwh <= 100.0

    def test_discharge_delivers(self):
        b = GridBattery(500.0, 300.0)
        assert b.discharge(100.0) > 0 and b.charge_kwh < 300.0

    def test_discharge_limited_by_charge(self):
        b = GridBattery(500.0, 10.0)
        delivered = b.discharge(1000.0)
        assert delivered <= 10.0 * BATTERY_ROUND_TRIP_EFF + 0.01
        assert b.charge_kwh >= 0.0

    def test_discharge_empty(self):
        b = GridBattery(500.0, 0.0)
        assert b.discharge(100.0) == pytest.approx(0.0)

    def test_self_discharge(self):
        b = GridBattery(500.0, 400.0)
        lost = b.self_discharge()
        assert lost == pytest.approx(400.0 * BATTERY_SELF_DISCHARGE_PER_SOL)

    def test_self_discharge_empty(self):
        b = GridBattery(500.0, 0.0)
        assert b.self_discharge() == pytest.approx(0.0)

    def test_overcharge_clamped(self):
        assert GridBattery(100.0, 200.0).charge_kwh == 100.0

    def test_negative_capacity_clamped(self):
        assert GridBattery(-50.0).capacity_kwh == 0.0


class TestAllocatePower:
    def test_sufficient_supply(self):
        loads = [PowerLoad("a", 0, 100.0), PowerLoad("b", 1, 50.0)]
        surplus = allocate_power(loads, 200.0)
        assert loads[0].allocated_kwh == pytest.approx(100.0)
        assert loads[1].allocated_kwh == pytest.approx(50.0)
        assert surplus == pytest.approx(50.0)

    def test_scarce_supply_high_priority_first(self):
        loads = [PowerLoad("critical", 0, 80.0), PowerLoad("optional", 5, 60.0)]
        surplus = allocate_power(loads, 100.0)
        assert loads[0].allocated_kwh == pytest.approx(80.0)
        assert loads[1].allocated_kwh == pytest.approx(20.0)
        assert surplus == pytest.approx(0.0)

    def test_zero_supply(self):
        loads = [PowerLoad("a", 0, 100.0)]
        assert allocate_power(loads, 0.0) == 0.0 and loads[0].allocated_kwh == 0.0

    def test_no_loads(self):
        assert allocate_power([], 500.0) == pytest.approx(500.0)

    def test_priority_ordering(self):
        loads = [PowerLoad("low", 5, 50.0), PowerLoad("high", 0, 50.0)]
        allocate_power(loads, 60.0)
        high = next(ld for ld in loads if ld.name == "high")
        low = next(ld for ld in loads if ld.name == "low")
        assert high.allocated_kwh == pytest.approx(50.0) and low.allocated_kwh == pytest.approx(10.0)

    def test_total_never_exceeds_supply(self):
        loads = [PowerLoad("a", 0, 200.0), PowerLoad("b", 1, 200.0), PowerLoad("c", 2, 200.0)]
        allocate_power(loads, 300.0)
        assert total_allocated(loads) <= 300.01

    def test_allocated_plus_surplus_equals_supply(self):
        loads = [PowerLoad("a", 0, 100.0), PowerLoad("b", 2, 80.0)]
        surplus = allocate_power(loads, 250.0)
        assert total_allocated(loads) + surplus == pytest.approx(250.0)

    def test_same_priority_deterministic(self):
        loads = [PowerLoad("zz", 1, 50.0), PowerLoad("aa", 1, 50.0)]
        allocate_power(loads, 60.0)
        aa = next(ld for ld in loads if ld.name == "aa")
        zz = next(ld for ld in loads if ld.name == "zz")
        assert aa.allocated_kwh == pytest.approx(50.0) and zz.allocated_kwh == pytest.approx(10.0)


class TestGridFrequency:
    def test_nominal_balanced(self):
        assert grid_frequency(100.0, 200.0) == GRID_FREQ_NOMINAL_HZ

    def test_nominal_exact(self):
        assert grid_frequency(100.0, 100.0) == GRID_FREQ_NOMINAL_HZ

    def test_drops_on_overload(self):
        assert grid_frequency(200.0, 100.0) < GRID_FREQ_NOMINAL_HZ

    def test_zero_supply(self):
        assert grid_frequency(100.0, 0.0) == 0.0

    def test_never_negative(self):
        assert grid_frequency(999999.0, 1.0) >= 0.0

    def test_overload_math(self):
        freq = grid_frequency(150.0, 100.0)
        expected = GRID_FREQ_NOMINAL_HZ - 0.5 * FREQ_DROP_PER_OVERLOAD_FRACTION
        assert freq == pytest.approx(expected, abs=0.01)


class TestUtilities:
    def test_total_demand(self):
        assert total_demand([PowerLoad("a", 0, 100.0), PowerLoad("b", 1, 50.0)]) == pytest.approx(150.0)

    def test_total_demand_empty(self):
        assert total_demand([]) == 0.0

    def test_total_allocated(self):
        loads = [PowerLoad("a", 0, 100.0, allocated_kwh=80.0), PowerLoad("b", 1, 50.0, allocated_kwh=50.0)]
        assert total_allocated(loads) == pytest.approx(130.0)

    def test_total_curtailed(self):
        loads = [PowerLoad("a", 0, 100.0, allocated_kwh=80.0), PowerLoad("b", 1, 50.0, allocated_kwh=30.0)]
        assert total_curtailed(loads) == pytest.approx(40.0)


class TestTickGrid:
    def test_basic_tick(self):
        state = GridState(battery=GridBattery(500.0, 250.0))
        result = tick_grid(state, solar_kwh=500.0, nuclear_kwh=960.0, loads=[PowerLoad("ls", 0, 200.0)])
        assert result["sol"] == 1.0 and result["consumed_kwh"] == pytest.approx(200.0)
        assert result["brownout"] is False

    def test_sol_advances(self):
        state = GridState()
        tick_grid(state, solar_kwh=100.0)
        assert state.sol == 1
        tick_grid(state, solar_kwh=100.0)
        assert state.sol == 2

    def test_brownout_detected(self):
        state = GridState(battery=GridBattery(100.0, 0.0))
        result = tick_grid(state, solar_kwh=10.0, nuclear_kwh=10.0, loads=[PowerLoad("big", 0, 9999.0)])
        assert result["brownout"] is True and state.brownout_sols == 1

    def test_blackout_detected(self):
        state = GridState(battery=GridBattery(100.0, 0.0))
        result = tick_grid(state, solar_kwh=0.0, nuclear_kwh=0.0, loads=[PowerLoad("x", 0, 100.0)])
        assert result["blackout"] is True and state.blackout_sols == 1

    def test_surplus_charges_battery(self):
        state = GridState(battery=GridBattery(500.0, 100.0))
        result = tick_grid(state, solar_kwh=500.0, nuclear_kwh=960.0, loads=[PowerLoad("s", 0, 10.0)])
        assert result["battery_charged_kwh"] > 0 and state.battery.charge_kwh > 100.0

    def test_battery_discharge_when_needed(self):
        state = GridState(battery=GridBattery(500.0, 400.0))
        result = tick_grid(state, solar_kwh=0.0, nuclear_kwh=960.0, loads=[PowerLoad("big", 0, 2000.0)], battery_discharge_kwh=500.0)
        assert result["battery_discharged_kwh"] > 0 and state.battery.charge_kwh < 400.0

    def test_no_loads(self):
        result = tick_grid(GridState(), solar_kwh=500.0, loads=[])
        assert result["consumed_kwh"] == 0.0

    def test_generation_cumulative(self):
        state = GridState()
        tick_grid(state, solar_kwh=100.0, nuclear_kwh=200.0)
        assert state.total_generated_kwh == pytest.approx(300.0)
        tick_grid(state, solar_kwh=150.0, nuclear_kwh=200.0)
        assert state.total_generated_kwh == pytest.approx(650.0)

    def test_peak_demand_tracked(self):
        state = GridState()
        tick_grid(state, solar_kwh=999.0, loads=[PowerLoad("a", 0, 100.0)])
        tick_grid(state, solar_kwh=999.0, loads=[PowerLoad("a", 0, 500.0)])
        assert state.peak_demand_kwh == pytest.approx(500.0)

    def test_grid_frequency_nominal(self):
        result = tick_grid(GridState(), solar_kwh=500.0, loads=[PowerLoad("a", 0, 100.0)])
        assert result["grid_frequency_hz"] == GRID_FREQ_NOMINAL_HZ

    def test_overloaded_freq_drops(self):
        state = GridState(battery=GridBattery(100.0, 0.0))
        result = tick_grid(state, solar_kwh=100.0, nuclear_kwh=0.0, loads=[PowerLoad("huge", 0, 5000.0)])
        assert result["grid_frequency_hz"] < GRID_FREQ_NOMINAL_HZ


class TestEnergyConservation:
    def test_consumed_le_generated_plus_battery(self):
        state = GridState(battery=GridBattery(500.0, 300.0))
        result = tick_grid(state, solar_kwh=200.0, nuclear_kwh=500.0, loads=[PowerLoad("a", 0, 800.0)], battery_discharge_kwh=200.0)
        assert result["consumed_kwh"] <= result["generation_kwh"] + result["battery_discharged_kwh"] + 0.01

    def test_curtailed_is_demand_minus_consumed(self):
        state = GridState(battery=GridBattery(100.0, 0.0))
        loads = [PowerLoad("a", 0, 500.0), PowerLoad("b", 5, 500.0)]
        result = tick_grid(state, solar_kwh=100.0, nuclear_kwh=200.0, loads=loads)
        assert result["curtailed_kwh"] == pytest.approx(result["demand_kwh"] - result["consumed_kwh"], abs=0.01)

    def test_cumulative_only_grows(self):
        state = GridState()
        prev_gen, prev_con = 0.0, 0.0
        for _ in range(20):
            tick_grid(state, solar_kwh=200.0, loads=[PowerLoad("a", 0, 100.0)])
            assert state.total_generated_kwh >= prev_gen and state.total_consumed_kwh >= prev_con
            prev_gen, prev_con = state.total_generated_kwh, state.total_consumed_kwh

    @pytest.mark.parametrize("solar", [0.0, 50.0, 200.0, 1000.0])
    def test_non_negative_outputs(self, solar):
        state = GridState(battery=GridBattery(500.0, 250.0))
        result = tick_grid(state, solar_kwh=solar, loads=[PowerLoad("a", 0, 300.0)])
        for key in ("generation_kwh", "consumed_kwh", "curtailed_kwh", "surplus_kwh", "battery_level_kwh", "grid_frequency_hz"):
            assert result[key] >= 0.0, f"{key} was negative"


class TestColonyLoads:
    def test_count(self):
        assert len(colony_loads_default()) == 9

    def test_all_positive(self):
        for ld in colony_loads_default():
            assert ld.requested_kwh > 0

    def test_life_support_highest_priority(self):
        life = [ld for ld in colony_loads_default() if ld.priority == PRIORITY_LIFE_SUPPORT]
        assert len(life) >= 2

    def test_total_demand_reasonable(self):
        assert 200 < total_demand(colony_loads_default()) < 3000


class TestCreateGrid:
    def test_conservative(self):
        assert create_grid("conservative").battery.capacity_kwh == 800.0

    def test_balanced(self):
        assert create_grid("balanced").battery.capacity_kwh == 500.0

    def test_aggressive(self):
        assert create_grid("aggressive").battery.capacity_kwh == 300.0

    def test_unknown_defaults(self):
        assert create_grid("unknown").battery.capacity_kwh == 500.0

    def test_all_strategies_valid(self):
        for s in ("conservative", "balanced", "aggressive"):
            g = create_grid(s)
            assert g.sol == 0 and g.battery.charge_kwh <= g.battery.capacity_kwh


class TestSmokeTests:
    def test_100_sols_colony(self):
        state = create_grid("balanced")
        for sol in range(100):
            result = tick_grid(state, solar_kwh=500.0, nuclear_kwh=960.0, loads=colony_loads_default(), battery_discharge_kwh=100.0)
            assert result["sol"] == sol + 1
        assert state.sol == 100 and state.total_consumed_kwh > 0

    def test_100_sols_solar_only(self):
        state = create_grid("balanced")
        for _ in range(100):
            tick_grid(state, solar_kwh=400.0, nuclear_kwh=0.0, loads=colony_loads_default())
        assert state.brownout_sols > 0

    def test_100_sols_nuclear_only(self):
        state = create_grid("balanced")
        for _ in range(100):
            tick_grid(state, solar_kwh=0.0, nuclear_kwh=960.0, loads=colony_loads_default())
        assert state.total_consumed_kwh > 0

    def test_365_sols_battery_survives(self):
        state = create_grid("conservative")
        for _ in range(365):
            tick_grid(state, solar_kwh=500.0, nuclear_kwh=960.0, loads=colony_loads_default(), battery_discharge_kwh=50.0)
        assert state.battery.charge_kwh >= 0.0 and state.sol == 365

    def test_brownout_recovery(self):
        state = create_grid("balanced")
        for _ in range(10):
            tick_grid(state, solar_kwh=0.0, nuclear_kwh=0.0, loads=colony_loads_default())
        assert state.brownout_sols > 0
        before = state.brownout_sols
        for _ in range(10):
            result = tick_grid(state, solar_kwh=1000.0, nuclear_kwh=960.0, loads=colony_loads_default())
            assert result["brownout"] is False
        assert state.brownout_sols == before

    def test_variable_solar_50_sols(self):
        state = create_grid("balanced")
        for sol in range(50):
            solar = 300.0 + 200.0 * math.sin(2 * math.pi * sol / 50)
            tick_grid(state, solar_kwh=solar, nuclear_kwh=960.0, loads=colony_loads_default(), battery_discharge_kwh=100.0)
        assert state.sol == 50


class TestPriorityCurtailment:
    def test_life_support_over_science(self):
        loads = [PowerLoad("science", PRIORITY_SCIENCE, 100.0), PowerLoad("life_support", PRIORITY_LIFE_SUPPORT, 100.0)]
        allocate_power(loads, 120.0)
        ls = next(ld for ld in loads if ld.name == "life_support")
        sci = next(ld for ld in loads if ld.name == "science")
        assert ls.allocated_kwh == pytest.approx(100.0) and sci.allocated_kwh == pytest.approx(20.0)

    def test_all_priorities(self):
        loads = [PowerLoad(f"p{i}", i, 50.0) for i in range(6)]
        allocate_power(loads, 180.0)
        for ld in loads:
            if ld.priority <= 2:
                assert ld.allocated_kwh == pytest.approx(50.0)
        p3 = next(ld for ld in loads if ld.name == "p3")
        assert p3.allocated_kwh == pytest.approx(30.0)
        assert next(ld for ld in loads if ld.name == "p4").allocated_kwh == 0.0
        assert next(ld for ld in loads if ld.name == "p5").allocated_kwh == 0.0

    def test_colony_loads_life_support_before_science(self):
        loads = colony_loads_default()
        allocate_power(loads, 250.0)
        lc_total = sum(ld.allocated_kwh for ld in loads if ld.priority == PRIORITY_LIFE_SUPPORT)
        sci_total = sum(ld.allocated_kwh for ld in loads if ld.priority == PRIORITY_SCIENCE)
        assert lc_total > sci_total
