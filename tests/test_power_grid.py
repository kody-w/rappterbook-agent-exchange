"""
Tests for power_grid.py - Mars colony power distribution and load management.

Coverage:
  - Physical constants sanity
  - PowerLoad construction and clamping
  - BatteryBank (capacity, SoC, DoD, degradation)
  - GridState construction and clamping
  - Grid efficiency (health scaling, bounded)
  - Battery tick (charge, discharge, rate limits, self-discharge)
  - Power allocation (priority ordering, proportional sharing)
  - Full grid tick (generation to consumption flow)
  - Brownout detection and load shedding
  - Battery covers deficit
  - Curtailment (excess generation)
  - System degradation + maintenance
  - Multi-sol simulation (10, 50, 365 sols)
  - Edge cases (zero gen, zero demand, massive surplus)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from power_grid import (
    BATTERY_CYCLE_DEGRADATION,
    BATTERY_DEPTH_OF_DISCHARGE,
    BATTERY_MAX_CHARGE_RATE,
    BATTERY_MAX_DISCHARGE_RATE,
    BATTERY_ROUND_TRIP_EFF,
    BATTERY_SELF_DISCHARGE_PER_SOL,
    GRID_LOSS_FRACTION,
    GRID_WEAR_PER_SOL,
    MAINTENANCE_RESTORE,
    MIN_GRID_HEALTH,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_LOWEST,
    PRIORITY_MEDIUM,
    BatteryBank,
    GridSol,
    GridState,
    PowerLoad,
    allocate_power,
    battery_tick,
    grid_efficiency,
    tick_grid,
)


# ===================================================================
# Constants sanity
# ===================================================================

class TestConstants:

    def test_battery_efficiency_bounded(self) -> None:
        assert 0 < BATTERY_ROUND_TRIP_EFF <= 1.0

    def test_battery_self_discharge_small(self) -> None:
        assert 0 < BATTERY_SELF_DISCHARGE_PER_SOL < 0.1

    def test_battery_dod_bounded(self) -> None:
        assert 0 < BATTERY_DEPTH_OF_DISCHARGE <= 1.0

    def test_grid_loss_bounded(self) -> None:
        assert 0 < GRID_LOSS_FRACTION < 0.5

    def test_charge_rate_positive(self) -> None:
        assert BATTERY_MAX_CHARGE_RATE > 0

    def test_discharge_rate_positive(self) -> None:
        assert BATTERY_MAX_DISCHARGE_RATE > 0

    def test_grid_wear_positive(self) -> None:
        assert GRID_WEAR_PER_SOL > 0

    def test_health_floor_bounded(self) -> None:
        assert 0 < MIN_GRID_HEALTH < 1.0

    def test_priorities_ordered(self) -> None:
        assert PRIORITY_CRITICAL < PRIORITY_HIGH < PRIORITY_MEDIUM < PRIORITY_LOW < PRIORITY_LOWEST


# ===================================================================
# PowerLoad
# ===================================================================

class TestPowerLoad:

    def test_basic_construction(self) -> None:
        load = PowerLoad(name="test", demand_kwh=100.0, priority=PRIORITY_MEDIUM)
        assert load.name == "test"
        assert load.demand_kwh == 100.0
        assert load.priority == PRIORITY_MEDIUM

    def test_clamps_negative_demand(self) -> None:
        load = PowerLoad(name="x", demand_kwh=-50.0)
        assert load.demand_kwh == 0.0

    def test_clamps_priority(self) -> None:
        load = PowerLoad(name="x", demand_kwh=10, priority=-5)
        assert load.priority == PRIORITY_CRITICAL
        load2 = PowerLoad(name="x", demand_kwh=10, priority=99)
        assert load2.priority == PRIORITY_LOWEST

    def test_min_fraction_clamped(self) -> None:
        load = PowerLoad(name="x", demand_kwh=10, min_fraction=1.5)
        assert load.min_fraction == 1.0
        load2 = PowerLoad(name="x", demand_kwh=10, min_fraction=-0.5)
        assert load2.min_fraction == 0.0


# ===================================================================
# BatteryBank
# ===================================================================

class TestBatteryBank:

    def test_defaults(self) -> None:
        b = BatteryBank()
        assert b.capacity_kwh == 500.0
        assert b.charge_kwh >= 0
        assert b.degradation == 0.0

    def test_usable_capacity(self) -> None:
        b = BatteryBank(capacity_kwh=1000.0, degradation=0.0)
        expected = 1000.0 * BATTERY_DEPTH_OF_DISCHARGE
        assert abs(b.usable_capacity - expected) < 0.01

    def test_usable_capacity_with_degradation(self) -> None:
        b = BatteryBank(capacity_kwh=1000.0, degradation=0.10)
        expected = 1000.0 * BATTERY_DEPTH_OF_DISCHARGE * 0.90
        assert abs(b.usable_capacity - expected) < 0.01

    def test_state_of_charge(self) -> None:
        b = BatteryBank(capacity_kwh=1000.0, charge_kwh=400.0)
        usable = 1000.0 * BATTERY_DEPTH_OF_DISCHARGE
        expected_soc = 400.0 / usable
        assert abs(b.state_of_charge - expected_soc) < 0.01

    def test_charge_clamped_to_usable(self) -> None:
        b = BatteryBank(capacity_kwh=100.0, charge_kwh=9999.0)
        assert b.charge_kwh <= b.usable_capacity

    def test_min_capacity(self) -> None:
        b = BatteryBank(capacity_kwh=-100)
        assert b.capacity_kwh >= 1.0

    def test_degradation_clamped(self) -> None:
        b = BatteryBank(degradation=5.0)
        assert b.degradation <= 0.99


# ===================================================================
# GridState
# ===================================================================

class TestGridState:

    def test_defaults(self) -> None:
        g = GridState()
        assert g.health == 1.0
        assert g.brownout_sols == 0

    def test_clamps_health(self) -> None:
        g = GridState(health=2.0)
        assert g.health == 1.0
        g2 = GridState(health=-1.0)
        assert g2.health == MIN_GRID_HEALTH

    def test_clamps_negative_totals(self) -> None:
        g = GridState(total_generated_kwh=-100)
        assert g.total_generated_kwh == 0.0


# ===================================================================
# Grid efficiency
# ===================================================================

class TestGridEfficiency:

    def test_full_health(self) -> None:
        eff = grid_efficiency(1.0)
        assert abs(eff - (1.0 - GRID_LOSS_FRACTION)) < 0.001

    def test_bounded(self) -> None:
        for h in [0.0, 0.3, 0.5, 0.8, 1.0, 1.5]:
            eff = grid_efficiency(h)
            assert 0.5 < eff <= 1.0

    def test_monotone_increasing(self) -> None:
        healths = [0.3, 0.5, 0.7, 1.0]
        effs = [grid_efficiency(h) for h in healths]
        for i in range(len(effs) - 1):
            assert effs[i] <= effs[i + 1]


# ===================================================================
# Battery tick
# ===================================================================

class TestBatteryTick:

    def test_charge(self) -> None:
        b = BatteryBank(capacity_kwh=1000.0, charge_kwh=100.0)
        stored = battery_tick(b, 50.0)
        assert stored > 0
        assert b.charge_kwh > 100.0

    def test_discharge(self) -> None:
        b = BatteryBank(capacity_kwh=1000.0, charge_kwh=500.0)
        initial = b.charge_kwh
        result = battery_tick(b, -100.0)
        assert result < 0
        assert b.charge_kwh < initial

    def test_zero_net(self) -> None:
        b = BatteryBank(capacity_kwh=1000.0, charge_kwh=500.0)
        initial = b.charge_kwh
        battery_tick(b, 0.0)
        # Only self-discharge
        assert b.charge_kwh < initial
        assert b.charge_kwh > initial * 0.99

    def test_charge_rate_limited(self) -> None:
        b = BatteryBank(capacity_kwh=100.0, charge_kwh=0.0)
        max_charge = 100.0 * BATTERY_MAX_CHARGE_RATE
        stored = battery_tick(b, 99999.0)
        # Stored should be limited by charge rate
        assert stored <= max_charge * math.sqrt(BATTERY_ROUND_TRIP_EFF) + 0.1

    def test_discharge_rate_limited(self) -> None:
        b = BatteryBank(capacity_kwh=100.0, charge_kwh=80.0)
        result = battery_tick(b, -99999.0)
        max_discharge = 100.0 * BATTERY_MAX_DISCHARGE_RATE
        assert abs(result) <= max_discharge * math.sqrt(BATTERY_ROUND_TRIP_EFF) + 0.1

    def test_never_negative_charge(self) -> None:
        b = BatteryBank(capacity_kwh=100.0, charge_kwh=1.0)
        battery_tick(b, -9999.0)
        assert b.charge_kwh >= 0.0

    def test_cycles_tracked(self) -> None:
        b = BatteryBank(capacity_kwh=100.0, charge_kwh=0.0)
        battery_tick(b, 50.0)
        assert b.cycles > 0

    def test_degradation_increases(self) -> None:
        b = BatteryBank(capacity_kwh=100.0, charge_kwh=0.0)
        battery_tick(b, 50.0)
        assert b.degradation > 0

    def test_self_discharge_occurs(self) -> None:
        b = BatteryBank(capacity_kwh=1000.0, charge_kwh=500.0)
        battery_tick(b, 0.0)
        assert b.charge_kwh < 500.0


# ===================================================================
# Power allocation
# ===================================================================

class TestAllocatePower:

    def test_sufficient_power(self) -> None:
        loads = [
            PowerLoad("a", 30.0, PRIORITY_CRITICAL),
            PowerLoad("b", 20.0, PRIORITY_LOW),
        ]
        alloc, shed = allocate_power(100.0, loads)
        assert alloc["a"] == 30.0
        assert alloc["b"] == 20.0
        assert shed == 0.0

    def test_priority_ordering(self) -> None:
        loads = [
            PowerLoad("critical", 60.0, PRIORITY_CRITICAL),
            PowerLoad("low", 60.0, PRIORITY_LOW),
        ]
        alloc, shed = allocate_power(80.0, loads)
        assert alloc["critical"] == 60.0  # critical gets full power
        assert alloc["low"] == 20.0       # low gets remainder
        assert shed > 0

    def test_proportional_within_priority(self) -> None:
        loads = [
            PowerLoad("a", 50.0, PRIORITY_MEDIUM),
            PowerLoad("b", 50.0, PRIORITY_MEDIUM),
        ]
        alloc, shed = allocate_power(60.0, loads)
        # Both should get ~30 each (proportional)
        assert abs(alloc["a"] - 30.0) < 1.0
        assert abs(alloc["b"] - 30.0) < 1.0

    def test_zero_power(self) -> None:
        loads = [PowerLoad("a", 50.0, PRIORITY_CRITICAL)]
        alloc, shed = allocate_power(0.0, loads)
        assert alloc["a"] == 0.0
        assert shed == 50.0

    def test_empty_loads(self) -> None:
        alloc, shed = allocate_power(100.0, [])
        assert len(alloc) == 0
        assert shed == 0.0

    def test_zero_demand_loads(self) -> None:
        loads = [PowerLoad("a", 0.0, PRIORITY_CRITICAL)]
        alloc, shed = allocate_power(100.0, loads)
        assert alloc["a"] == 0.0

    def test_total_allocated_bounded(self) -> None:
        loads = [
            PowerLoad("a", 100.0, PRIORITY_CRITICAL),
            PowerLoad("b", 100.0, PRIORITY_HIGH),
            PowerLoad("c", 100.0, PRIORITY_LOW),
        ]
        alloc, shed = allocate_power(150.0, loads)
        total = sum(alloc.values())
        assert total <= 150.0 + 0.001


# ===================================================================
# Full grid tick
# ===================================================================

class TestTickGrid:

    def test_basic_operation(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=500.0, charge_kwh=250.0)
        loads = [PowerLoad("life_support", 50.0, PRIORITY_CRITICAL)]
        result = tick_grid(grid, battery, 200.0, loads)
        assert result.generated_kwh > 0
        assert result.consumed_kwh >= 50.0
        assert not result.brownout

    def test_brownout_detected(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=10.0, charge_kwh=0.0)
        loads = [PowerLoad("big_load", 500.0, PRIORITY_LOW)]
        result = tick_grid(grid, battery, 50.0, loads)
        assert result.brownout
        assert result.shed_kwh > 0

    def test_battery_covers_deficit(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=1000.0, charge_kwh=800.0)
        loads = [PowerLoad("critical", 200.0, PRIORITY_CRITICAL)]
        result = tick_grid(grid, battery, 100.0, loads)
        # Battery should cover some of the deficit
        assert result.consumed_kwh > 100.0 * grid_efficiency(1.0)

    def test_surplus_charges_battery(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=1000.0, charge_kwh=100.0)
        loads = [PowerLoad("small", 10.0, PRIORITY_LOW)]
        result = tick_grid(grid, battery, 500.0, loads)
        assert result.stored_kwh > 0
        assert battery.charge_kwh > 100.0

    def test_energy_conservation(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=500.0, charge_kwh=250.0)
        loads = [
            PowerLoad("a", 50.0, PRIORITY_CRITICAL),
            PowerLoad("b", 30.0, PRIORITY_HIGH),
        ]
        result = tick_grid(grid, battery, 200.0, loads)
        # Generated should approximately equal consumed + stored + curtailed
        balance = result.consumed_kwh + abs(result.stored_kwh) + result.curtailed_kwh
        assert balance <= result.generated_kwh + 1.0

    def test_zero_generation(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=100.0, charge_kwh=50.0)
        loads = [PowerLoad("a", 30.0, PRIORITY_CRITICAL)]
        result = tick_grid(grid, battery, 0.0, loads)
        assert result.generated_kwh == 0

    def test_no_loads(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=500.0, charge_kwh=100.0)
        result = tick_grid(grid, battery, 200.0, [])
        assert result.consumed_kwh == 0
        assert not result.brownout

    def test_grid_degrades(self) -> None:
        grid = GridState(health=1.0)
        battery = BatteryBank()
        tick_grid(grid, battery, 100.0, [])
        assert grid.health < 1.0

    def test_maintenance_restores(self) -> None:
        grid = GridState(health=0.5)
        battery = BatteryBank()
        tick_grid(grid, battery, 100.0, [], crew_maintenance=True)
        assert grid.health > 0.5

    def test_health_floor(self) -> None:
        grid = GridState(health=MIN_GRID_HEALTH + 0.001)
        battery = BatteryBank()
        for _ in range(100):
            tick_grid(grid, battery, 100.0, [])
        assert grid.health >= MIN_GRID_HEALTH

    def test_allocations_returned(self) -> None:
        grid = GridState()
        battery = BatteryBank()
        loads = [
            PowerLoad("a", 30.0, PRIORITY_CRITICAL),
            PowerLoad("b", 20.0, PRIORITY_LOW),
        ]
        result = tick_grid(grid, battery, 200.0, loads)
        assert "a" in result.allocations
        assert "b" in result.allocations

    def test_lifetime_totals_accumulate(self) -> None:
        grid = GridState()
        battery = BatteryBank()
        loads = [PowerLoad("a", 30.0, PRIORITY_CRITICAL)]
        for _ in range(5):
            tick_grid(grid, battery, 100.0, loads)
        assert grid.total_generated_kwh > 0
        assert grid.total_consumed_kwh > 0


# ===================================================================
# Multi-sol smoke tests
# ===================================================================

class TestMultiSolSmoke:

    def test_10_sols(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=500.0, charge_kwh=250.0)
        loads = [
            PowerLoad("life_support", 40.0, PRIORITY_CRITICAL),
            PowerLoad("thermal", 30.0, PRIORITY_CRITICAL),
            PowerLoad("water", 20.0, PRIORITY_HIGH),
            PowerLoad("greenhouse", 15.0, PRIORITY_MEDIUM),
            PowerLoad("isru", 25.0, PRIORITY_LOW),
        ]
        for _ in range(10):
            result = tick_grid(grid, battery, 180.0, loads)
        assert grid.total_consumed_kwh > 0
        assert battery.charge_kwh >= 0

    def test_50_sols_varying_generation(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=800.0, charge_kwh=400.0)
        loads = [
            PowerLoad("critical", 50.0, PRIORITY_CRITICAL),
            PowerLoad("normal", 40.0, PRIORITY_MEDIUM),
        ]
        brownouts = 0
        for sol in range(50):
            # Simulate varying solar output (dust storms reduce it)
            gen = 150.0 if sol % 10 != 0 else 30.0  # dust storm every 10 sols
            maintain = (sol % 7 == 0)
            result = tick_grid(grid, battery, gen, loads, crew_maintenance=maintain)
            if result.brownout:
                brownouts += 1
        assert grid.health >= MIN_GRID_HEALTH

    def test_365_sols_full_year(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=1000.0, charge_kwh=500.0)
        loads = [
            PowerLoad("life_support", 50.0, PRIORITY_CRITICAL, min_fraction=0.8),
            PowerLoad("thermal", 35.0, PRIORITY_CRITICAL, min_fraction=0.7),
            PowerLoad("water", 25.0, PRIORITY_HIGH),
            PowerLoad("comms", 10.0, PRIORITY_HIGH),
            PowerLoad("greenhouse", 20.0, PRIORITY_MEDIUM),
            PowerLoad("medical", 15.0, PRIORITY_MEDIUM),
            PowerLoad("isru", 30.0, PRIORITY_LOW),
            PowerLoad("comfort", 10.0, PRIORITY_LOWEST),
        ]
        total_consumed = 0.0
        total_shed = 0.0
        for sol in range(365):
            # Seasonal variation + occasional dust storms
            seasonal = 1.0 + 0.15 * math.sin(2 * math.pi * sol / 668)
            dust_storm = 0.3 if (sol % 60 < 5) else 1.0
            gen = 250.0 * seasonal * dust_storm
            maintain = (sol % 10 == 0)
            result = tick_grid(grid, battery, gen, loads, crew_maintenance=maintain)
            total_consumed += result.consumed_kwh
            total_shed += result.shed_kwh
        assert total_consumed > 0
        assert grid.total_generated_kwh > 0
        # Over a year, should deliver substantial energy
        assert total_consumed > 50000, f"365 sols should deliver >50000 kWh, got {total_consumed:.0f}"


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_zero_everything(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=1.0, charge_kwh=0.0)
        result = tick_grid(grid, battery, 0.0, [])
        assert result.consumed_kwh == 0
        assert not result.brownout

    def test_massive_generation(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=100.0, charge_kwh=0.0)
        loads = [PowerLoad("small", 10.0, PRIORITY_LOW)]
        result = tick_grid(grid, battery, 100000.0, loads)
        assert result.curtailed_kwh > 0

    def test_massive_demand(self) -> None:
        grid = GridState()
        battery = BatteryBank(capacity_kwh=10.0, charge_kwh=0.0)
        loads = [PowerLoad("huge", 100000.0, PRIORITY_CRITICAL)]
        result = tick_grid(grid, battery, 100.0, loads)
        assert result.brownout
        assert result.shed_kwh > 0

    def test_single_load_critical(self) -> None:
        grid = GridState()
        battery = BatteryBank()
        loads = [PowerLoad("only", 50.0, PRIORITY_CRITICAL, min_fraction=1.0)]
        result = tick_grid(grid, battery, 200.0, loads)
        assert result.allocations["only"] >= 50.0

    def test_many_priorities(self) -> None:
        grid = GridState()
        battery = BatteryBank()
        loads = [
            PowerLoad("p0", 10.0, PRIORITY_CRITICAL),
            PowerLoad("p1", 10.0, PRIORITY_HIGH),
            PowerLoad("p2", 10.0, PRIORITY_MEDIUM),
            PowerLoad("p3", 10.0, PRIORITY_LOW),
            PowerLoad("p4", 10.0, PRIORITY_LOWEST),
        ]
        result = tick_grid(grid, battery, 35.0, loads)
        # Critical and high should be fully served
        assert result.allocations["p0"] == 10.0
        assert result.allocations["p1"] == 10.0
