"""Tests for battery_bank.py -- Mars Colony Energy Storage."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from battery_bank import (
    BatteryState,
    SolRecord,
    usable_capacity,
    self_discharge_rate,
    charge_energy,
    discharge_energy,
    cycle_degradation,
    heater_energy_per_sol,
    battery_temperature,
    should_shed_module,
    tick_battery,
    make_battery,
    run_battery,
    MARS_AMBIENT_K,
    OPTIMAL_TEMP_K,
    MAX_DOD,
    ROUND_TRIP_EFFICIENCY,
    EOL_CAPACITY_FRAC,
    DEFAULT_CAPACITY_KWH,
    DEFAULT_MODULES,
    HOURS_PER_SOL,
)


# -- TestUsableCapacity ---------------------------------------------------

class TestUsableCapacity:
    """Test usable capacity calculations."""

    def test_full_health_full_modules(self):
        state = make_battery()
        cap = usable_capacity(state)
        expected = DEFAULT_CAPACITY_KWH * MAX_DOD  # 500 * 0.8 = 400
        assert abs(cap - expected) < 0.1

    def test_half_health(self):
        state = make_battery()
        state.health = 0.5
        cap = usable_capacity(state)
        expected = DEFAULT_CAPACITY_KWH * 0.5 * MAX_DOD
        assert abs(cap - expected) < 0.1

    def test_half_modules(self):
        state = make_battery()
        state.modules_online = 10
        cap = usable_capacity(state)
        expected = DEFAULT_CAPACITY_KWH * 0.5 * MAX_DOD
        assert abs(cap - expected) < 0.1

    def test_cold_reduces_capacity(self):
        warm = make_battery()
        warm.temperature_k = 293.0
        cold = make_battery()
        cold.temperature_k = 200.0
        assert usable_capacity(warm) > usable_capacity(cold)

    def test_zero_modules(self):
        state = make_battery()
        state.modules_online = 0
        assert usable_capacity(state) == 0.0

    def test_capacity_always_non_negative(self):
        state = make_battery()
        state.health = 0.0
        assert usable_capacity(state) >= 0.0

    def test_cold_penalty_capped(self):
        """Even at extremely low temps, capacity doesn't go negative."""
        state = make_battery()
        state.temperature_k = 50.0  # Extremely cold
        cap = usable_capacity(state)
        assert cap >= 0.0
        assert cap <= DEFAULT_CAPACITY_KWH * MAX_DOD


# -- TestSelfDischarge ----------------------------------------------------

class TestSelfDischarge:
    """Test self-discharge rate calculations."""

    def test_warmer_discharges_faster(self):
        """Higher temperature = faster self-discharge (Arrhenius)."""
        rate_warm = self_discharge_rate(320.0)
        rate_cool = self_discharge_rate(250.0)
        assert rate_warm > rate_cool

    def test_rate_is_positive(self):
        rate = self_discharge_rate(OPTIMAL_TEMP_K)
        assert rate > 0.0

    def test_rate_bounded(self):
        """Rate should be between 0 and 50%."""
        for temp in [100.0, 200.0, 293.0, 400.0, 500.0]:
            rate = self_discharge_rate(temp)
            assert 0.0 <= rate <= 0.5

    def test_zero_temp_safe(self):
        """Should not crash at 0 K."""
        rate = self_discharge_rate(0.0)
        assert rate >= 0.0


# -- TestChargeEnergy -----------------------------------------------------

class TestChargeEnergy:
    """Test charge energy calculations."""

    def test_charge_respects_headroom(self):
        state = make_battery(initial_charge_frac=0.99)
        cap = usable_capacity(state)
        headroom = cap - state.stored_kwh
        charged = charge_energy(state, 1000.0)
        assert charged <= headroom + 0.01

    def test_charge_applies_efficiency(self):
        state = make_battery(initial_charge_frac=0.0)
        state.stored_kwh = 0.0
        charged = charge_energy(state, 100.0)
        assert charged <= 100.0 * ROUND_TRIP_EFFICIENCY + 0.01

    def test_no_charge_when_full(self):
        state = make_battery(initial_charge_frac=1.0)
        # Force stored to exactly capacity
        state.stored_kwh = usable_capacity(state)
        charged = charge_energy(state, 100.0)
        assert charged == 0.0

    def test_charge_non_negative(self):
        state = make_battery()
        charged = charge_energy(state, -10.0)
        assert charged >= 0.0

    def test_rate_limiting(self):
        """Charge limited by max rate * hours per sol."""
        state = make_battery(initial_charge_frac=0.0)
        state.stored_kwh = 0.0
        # Very small max rate
        charged = charge_energy(state, 10000.0, max_rate_kw=0.001)
        assert charged < 1.0  # Tiny rate limits total charge


# -- TestDischargeEnergy --------------------------------------------------

class TestDischargeEnergy:
    """Test discharge energy calculations."""

    def test_discharge_limited_by_stored(self):
        state = make_battery(initial_charge_frac=0.1)
        delivered = discharge_energy(state, 10000.0)
        assert delivered <= state.stored_kwh + 0.01

    def test_no_discharge_when_empty(self):
        state = make_battery(initial_charge_frac=0.0)
        state.stored_kwh = 0.0
        delivered = discharge_energy(state, 100.0)
        assert delivered == 0.0

    def test_discharge_non_negative(self):
        state = make_battery()
        delivered = discharge_energy(state, -5.0)
        assert delivered >= 0.0

    def test_rate_limiting(self):
        state = make_battery(initial_charge_frac=0.9)
        delivered = discharge_energy(state, 10000.0, max_rate_kw=0.001)
        assert delivered < 1.0


# -- TestCycleDegradation ------------------------------------------------

class TestCycleDegradation:
    """Test cycle degradation calculations."""

    def test_no_cycling_no_degradation(self):
        loss = cycle_degradation(0.0, 0.0, 500.0)
        assert loss == 0.0

    def test_degradation_proportional(self):
        loss_small = cycle_degradation(50.0, 50.0, 500.0)
        loss_big = cycle_degradation(100.0, 100.0, 500.0)
        assert loss_big > loss_small

    def test_degradation_non_negative(self):
        loss = cycle_degradation(100.0, 100.0, 500.0)
        assert loss >= 0.0

    def test_zero_nameplate_safe(self):
        loss = cycle_degradation(100.0, 100.0, 0.0)
        assert loss == 0.0


# -- TestHeater -----------------------------------------------------------

class TestHeater:
    """Test heater energy calculations."""

    def test_heater_off_zero_energy(self):
        energy = heater_energy_per_sol(MARS_AMBIENT_K, heater_on=False)
        assert energy == 0.0

    def test_heater_energy_positive_on_mars(self):
        energy = heater_energy_per_sol(MARS_AMBIENT_K, heater_on=True)
        assert energy > 0.0

    def test_warmer_ambient_less_heating(self):
        cold = heater_energy_per_sol(150.0, True)
        warm = heater_energy_per_sol(280.0, True)
        assert cold > warm

    def test_heater_at_optimal_temp(self):
        """No heating needed if ambient is already optimal."""
        energy = heater_energy_per_sol(OPTIMAL_TEMP_K, True)
        assert energy == 0.0


# -- TestBatteryTemperature -----------------------------------------------

class TestBatteryTemperature:
    """Test temperature model."""

    def test_heater_on_optimal(self):
        temp = battery_temperature(MARS_AMBIENT_K, heater_on=True)
        assert temp == OPTIMAL_TEMP_K

    def test_heater_off_cold(self):
        temp = battery_temperature(MARS_AMBIENT_K, heater_on=False)
        assert temp < OPTIMAL_TEMP_K
        assert temp > MARS_AMBIENT_K  # Insulation adds some warmth


# -- TestModuleFailure ----------------------------------------------------

class TestModuleFailure:
    """Test module failure logic."""

    def test_no_fail_normal(self):
        state = make_battery()
        assert not should_shed_module(state, 0.5)

    def test_fail_with_extreme_rng(self):
        state = make_battery()
        state.cumulative_cycles = 5000.0  # Very aged
        # rng_val = 0.0 should always trigger
        assert should_shed_module(state, 0.0)

    def test_no_modules_no_fail(self):
        state = make_battery()
        state.modules_online = 0
        assert not should_shed_module(state, 0.0)


# -- TestTickBattery ------------------------------------------------------

class TestTickBattery:
    """Test the main tick function."""

    def test_sol_increments(self):
        state = make_battery()
        tick_battery(state, charge_available_kwh=50.0)
        assert state.sol == 1

    def test_charge_increases_stored(self):
        state = make_battery(initial_charge_frac=0.1)
        before = state.stored_kwh
        tick_battery(state, charge_available_kwh=200.0, discharge_demand_kwh=0.0)
        assert state.stored_kwh > before

    def test_discharge_decreases_stored(self):
        state = make_battery(initial_charge_frac=0.9)
        before = state.stored_kwh
        tick_battery(state, charge_available_kwh=0.0, discharge_demand_kwh=200.0)
        assert state.stored_kwh < before

    def test_stored_never_negative(self):
        state = make_battery(initial_charge_frac=0.01)
        tick_battery(state, charge_available_kwh=0.0, discharge_demand_kwh=99999.0)
        assert state.stored_kwh >= 0.0

    def test_stored_never_exceeds_capacity(self):
        state = make_battery(initial_charge_frac=0.5)
        tick_battery(state, charge_available_kwh=99999.0, discharge_demand_kwh=0.0)
        cap = usable_capacity(state)
        assert state.stored_kwh <= cap + 0.01

    def test_health_decreases_with_cycling(self):
        state = make_battery()
        initial_health = state.health
        for _ in range(100):
            tick_battery(state, charge_available_kwh=100.0, discharge_demand_kwh=80.0)
        assert state.health < initial_health

    def test_health_never_below_eol(self):
        state = make_battery()
        for _ in range(10000):
            tick_battery(state, charge_available_kwh=200.0, discharge_demand_kwh=200.0)
        assert state.health >= EOL_CAPACITY_FRAC

    def test_returns_sol_record(self):
        state = make_battery()
        record = tick_battery(state, charge_available_kwh=50.0)
        assert isinstance(record, SolRecord)
        assert record.sol == 1

    def test_self_discharge_occurs(self):
        state = make_battery(initial_charge_frac=0.8)
        record = tick_battery(state, charge_available_kwh=0.0, discharge_demand_kwh=0.0)
        assert record.self_discharge_kwh > 0.0

    def test_heater_draws_energy(self):
        state = make_battery(initial_charge_frac=0.5)
        record = tick_battery(state, ambient_k=MARS_AMBIENT_K)
        assert record.heater_energy_kwh > 0.0

    def test_cold_battery_loses_more_capacity(self):
        """Battery without heater has less usable capacity."""
        warm = make_battery()
        warm.heater_on = True
        cold = make_battery()
        cold.heater_on = False
        tick_battery(warm, charge_available_kwh=50.0)
        tick_battery(cold, charge_available_kwh=50.0)
        assert usable_capacity(warm) > usable_capacity(cold)


# -- TestMakeBattery ------------------------------------------------------

class TestMakeBattery:
    """Test the factory function."""

    def test_default_factory(self):
        state = make_battery()
        assert state.nameplate_kwh == DEFAULT_CAPACITY_KWH
        assert state.modules_online == DEFAULT_MODULES
        assert state.health == 1.0
        assert state.sol == 0

    def test_custom_capacity(self):
        state = make_battery(capacity_kwh=1000.0)
        assert state.nameplate_kwh == 1000.0

    def test_initial_charge(self):
        state = make_battery(initial_charge_frac=0.0)
        assert state.stored_kwh == 0.0

    def test_full_charge(self):
        state = make_battery(initial_charge_frac=1.0)
        cap = usable_capacity(state)
        assert abs(state.stored_kwh - cap) < 0.01


# -- TestRunBattery -------------------------------------------------------

class TestRunBattery:
    """Test multi-sol runner."""

    def test_returns_correct_count(self):
        state = make_battery()
        records = run_battery(state, sols=50)
        assert len(records) == 50

    def test_sols_sequential(self):
        state = make_battery()
        records = run_battery(state, sols=10)
        for i, r in enumerate(records):
            assert r.sol == i + 1

    def test_charge_surplus_fills_bank(self):
        """More charge than demand should fill the bank over time."""
        state = make_battery(initial_charge_frac=0.1)
        initial = state.stored_kwh
        run_battery(state, sols=50, charge_per_sol=100.0, demand_per_sol=10.0)
        assert state.stored_kwh > initial

    def test_demand_surplus_drains_bank(self):
        """More demand than charge should drain the bank."""
        state = make_battery(initial_charge_frac=0.9)
        initial = state.stored_kwh
        run_battery(state, sols=50, charge_per_sol=10.0, demand_per_sol=200.0)
        assert state.stored_kwh < initial


# -- TestSmokeTest --------------------------------------------------------

class TestSmokeTest:
    """Smoke tests: run without crash, physics invariants hold."""

    def test_100_sol_no_crash(self):
        state = make_battery()
        records = run_battery(state, sols=100)
        assert len(records) == 100
        assert state.sol == 100

    def test_1000_sol_endurance(self):
        state = make_battery()
        records = run_battery(state, sols=1000)
        assert len(records) == 1000
        # Health should degrade but stay above EOL
        assert state.health >= EOL_CAPACITY_FRAC
        # Energy is conserved (stored >= 0)
        assert state.stored_kwh >= 0.0

    def test_all_records_valid(self):
        """Every SolRecord has physically valid values."""
        state = make_battery()
        records = run_battery(state, sols=200)
        for r in records:
            assert r.stored_kwh >= 0.0
            assert r.health >= EOL_CAPACITY_FRAC
            assert r.temperature_k > 0.0
            assert r.charged_kwh >= 0.0
            assert r.discharged_kwh >= 0.0
            assert r.self_discharge_kwh >= 0.0
            assert r.usable_kwh >= 0.0
            assert r.modules_online >= 0

    def test_energy_conservation_per_sol(self):
        """Stored energy change should equal net energy flow."""
        state = make_battery(initial_charge_frac=0.5)
        stored_before = state.stored_kwh
        record = tick_battery(
            state,
            charge_available_kwh=60.0,
            discharge_demand_kwh=30.0,
            ambient_k=MARS_AMBIENT_K,
        )
        stored_after = state.stored_kwh
        delta = stored_after - stored_before
        net_flow = record.charged_kwh - record.discharged_kwh - record.self_discharge_kwh
        # Allow small floating-point tolerance and capacity clamping
        assert abs(delta - net_flow) < 1.0, (
            f"Energy not conserved: delta={delta:.2f}, net_flow={net_flow:.2f}"
        )
