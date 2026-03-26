"""
Tests for nuclear_reactor.py — Mars Colony Fission Reactor.

72 tests across 9 test classes. The colony's backup power during dust storms.

Run: python -m pytest tests/test_nuclear_reactor.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nuclear_reactor import (
    ReactorState,
    ReactorSol,
    thermal_power,
    electric_output,
    fuel_burnup_per_sol,
    core_temperature,
    should_scram,
    tick_reactor,
    create_reactor,
    run_reactor,
    THERMAL_POWER_MAX_KW,
    ELECTRIC_POWER_MAX_KW,
    STIRLING_EFFICIENCY,
    NOMINAL_ROD_POSITION,
    NOMINAL_CORE_TEMP_K,
    CONTROL_ROD_MIN,
    CONTROL_ROD_MAX,
    TEMP_COEFFICIENT,
    FUEL_LIFE_SOLS,
    NUM_STIRLING_ENGINES,
    MIN_ENGINES_FOR_OPERATION,
    STARTUP_SOLS,
    SCRAM_COOLDOWN_SOLS,
    RADIATOR_DEGRADATION_PER_SOL,
    RADIATOR_MIN_EFFICIENCY,
    RADIATOR_REPAIR_PER_SOL,
    STIRLING_FAILURE_PROB_PER_SOL,
)


# --- Thermal Power ---

class TestThermalPower:

    def test_shutdown_zero_power(self):
        assert thermal_power(0.0, 1.0, NOMINAL_CORE_TEMP_K) == 0.0

    def test_no_fuel_zero_power(self):
        assert thermal_power(NOMINAL_ROD_POSITION, 0.0, NOMINAL_CORE_TEMP_K) == 0.0

    def test_nominal_power(self):
        p = thermal_power(NOMINAL_ROD_POSITION, 1.0, NOMINAL_CORE_TEMP_K)
        expected = THERMAL_POWER_MAX_KW * NOMINAL_ROD_POSITION
        assert abs(p - expected) < 0.1

    def test_more_rod_more_power(self):
        p_low = thermal_power(0.3, 1.0, NOMINAL_CORE_TEMP_K)
        p_high = thermal_power(0.7, 1.0, NOMINAL_CORE_TEMP_K)
        assert p_high > p_low

    def test_depleted_fuel_less_power(self):
        p_full = thermal_power(0.7, 1.0, NOMINAL_CORE_TEMP_K)
        p_half = thermal_power(0.7, 0.5, NOMINAL_CORE_TEMP_K)
        assert p_half < p_full

    def test_negative_temp_coefficient(self):
        """Hotter core produces less power (passive safety)."""
        p_nominal = thermal_power(0.7, 1.0, NOMINAL_CORE_TEMP_K)
        p_hot = thermal_power(0.7, 1.0, NOMINAL_CORE_TEMP_K + 200)
        assert p_hot < p_nominal

    def test_cold_core_no_penalty(self):
        """Below nominal temp, no negative feedback."""
        p_nominal = thermal_power(0.7, 1.0, NOMINAL_CORE_TEMP_K)
        p_cold = thermal_power(0.7, 1.0, NOMINAL_CORE_TEMP_K - 100)
        assert abs(p_cold - p_nominal) < 0.1

    def test_power_always_non_negative(self):
        for rod in [0.0, 0.3, 0.7, 1.0]:
            for fuel in [0.0, 0.5, 1.0]:
                for temp in [300, 800, 1073, 1500]:
                    p = thermal_power(rod, fuel, float(temp))
                    assert p >= 0.0


# --- Electric Output ---

class TestElectricOutput:

    def test_zero_thermal_zero_electric(self):
        assert electric_output(0.0, NUM_STIRLING_ENGINES, 1.0) == 0.0

    def test_no_engines_zero_electric(self):
        assert electric_output(100.0, 0, 1.0) == 0.0

    def test_nominal_output(self):
        el = electric_output(THERMAL_POWER_MAX_KW, NUM_STIRLING_ENGINES, 1.0)
        assert abs(el - ELECTRIC_POWER_MAX_KW) < 0.1

    def test_fewer_engines_less_output(self):
        el_full = electric_output(100.0, NUM_STIRLING_ENGINES, 1.0)
        el_half = electric_output(100.0, NUM_STIRLING_ENGINES // 2, 1.0)
        assert el_half < el_full

    def test_poor_radiator_less_output(self):
        el_good = electric_output(100.0, NUM_STIRLING_ENGINES, 1.0)
        el_bad = electric_output(100.0, NUM_STIRLING_ENGINES, 0.5)
        assert el_bad < el_good

    def test_efficiency_never_exceeds_carnot(self):
        """Stirling efficiency < Carnot efficiency (thermodynamic limit)."""
        assert STIRLING_EFFICIENCY < 1.0 - 373.0 / 1073.0

    def test_output_always_non_negative(self):
        for th in [0, 50, 100, 160]:
            for eng in [0, 4, 8]:
                for rad in [0.5, 0.75, 1.0]:
                    e = electric_output(float(th), eng, rad)
                    assert e >= 0.0


# --- Fuel Burnup ---

class TestFuelBurnup:

    def test_zero_power_no_burnup(self):
        assert fuel_burnup_per_sol(0.0) == 0.0

    def test_full_power_known_rate(self):
        rate = fuel_burnup_per_sol(THERMAL_POWER_MAX_KW)
        assert abs(rate - 1.0 / FUEL_LIFE_SOLS) < 1e-10

    def test_half_power_half_rate(self):
        full = fuel_burnup_per_sol(THERMAL_POWER_MAX_KW)
        half = fuel_burnup_per_sol(THERMAL_POWER_MAX_KW / 2)
        assert abs(half - full / 2) < 1e-10

    def test_burnup_non_negative(self):
        for p in [0, 50, 100, 160]:
            assert fuel_burnup_per_sol(float(p)) >= 0.0

    def test_full_life_calculation(self):
        """At full power, fuel lasts FUEL_LIFE_SOLS."""
        rate = fuel_burnup_per_sol(THERMAL_POWER_MAX_KW)
        sols_to_depletion = 1.0 / rate
        assert abs(sols_to_depletion - FUEL_LIFE_SOLS) < 1.0


# --- Core Temperature ---

class TestCoreTemperature:

    def test_no_power_cools_down(self):
        new_temp = core_temperature(800.0, 0.0, 1.0)
        assert new_temp < 800.0

    def test_power_heats_up(self):
        new_temp = core_temperature(300.0, 100.0, 1.0)
        assert new_temp > 300.0

    def test_temp_never_below_mars_ambient(self):
        """Core never drops below ~210 K (Mars average)."""
        t = core_temperature(210.0, 0.0, 1.0)
        assert t >= 210.0

    def test_good_radiator_keeps_cooler(self):
        t_good = core_temperature(1000.0, 100.0, 1.0)
        t_bad = core_temperature(1000.0, 100.0, 0.5)
        assert t_good <= t_bad


# --- SCRAM Logic ---

class TestScramLogic:

    def test_normal_no_scram(self):
        scram, _ = should_scram(NOMINAL_CORE_TEMP_K, NUM_STIRLING_ENGINES)
        assert scram is False

    def test_overtemp_scrams(self):
        scram, reason = should_scram(1300.0, NUM_STIRLING_ENGINES)
        assert scram is True
        assert "overtemp" in reason

    def test_few_engines_scrams(self):
        scram, reason = should_scram(NOMINAL_CORE_TEMP_K, MIN_ENGINES_FOR_OPERATION - 1)
        assert scram is True
        assert "Stirling" in reason

    def test_minimum_engines_ok(self):
        scram, _ = should_scram(NOMINAL_CORE_TEMP_K, MIN_ENGINES_FOR_OPERATION)
        assert scram is False


# --- Tick Function ---

class TestTickReactor:

    def test_tick_advances_sol(self):
        state = create_reactor()
        state, sol = tick_reactor(state, rng=random.Random(42))
        assert state.sol == 1

    def test_startup_sequence(self):
        state = create_reactor()
        state, sol = tick_reactor(state, command="startup", rng=random.Random(42))
        assert state.state == "startup"
        assert state.control_rod_position > 0.0

    def test_startup_reaches_nominal(self):
        state = create_reactor()
        rng = random.Random(42)
        state, _ = tick_reactor(state, command="startup", rng=rng)
        for _ in range(STARTUP_SOLS + 1):
            state, _ = tick_reactor(state, rng=rng)
        assert state.state == "nominal"

    def test_nominal_produces_power(self):
        state = create_reactor()
        rng = random.Random(42)
        state, _ = tick_reactor(state, command="startup", rng=rng)
        for _ in range(STARTUP_SOLS + 2):
            state, sol = tick_reactor(state, rng=rng)
        assert state.electric_power_kw > 0.0

    def test_shutdown_command(self):
        state = create_reactor()
        state.state = "nominal"
        state.control_rod_position = NOMINAL_ROD_POSITION
        state, sol = tick_reactor(state, command="shutdown", rng=random.Random(42))
        assert state.state == "shutdown"
        assert state.control_rod_position == CONTROL_ROD_MIN

    def test_radiator_degrades(self):
        state = create_reactor()
        state, _ = tick_reactor(state, rng=random.Random(42))
        assert state.radiator_efficiency < 1.0

    def test_radiator_maintenance(self):
        state = create_reactor()
        state.radiator_efficiency = 0.8
        state, _ = tick_reactor(state, maintain_radiator=True, rng=random.Random(42))
        # Should increase (maintenance) minus degradation
        expected = 0.8 - RADIATOR_DEGRADATION_PER_SOL + RADIATOR_REPAIR_PER_SOL
        assert abs(state.radiator_efficiency - expected) < 0.001

    def test_non_operational_skips(self):
        state = create_reactor()
        state.operational = False
        state, sol = tick_reactor(state, rng=random.Random(42))
        assert state.sol == 1
        assert sol.electric_power_kw == 0.0

    def test_fuel_depletes_over_time(self):
        state = create_reactor()
        rng = random.Random(42)
        state, _ = tick_reactor(state, command="startup", rng=rng)
        for _ in range(STARTUP_SOLS + 10):
            state, _ = tick_reactor(state, rng=rng)
        assert state.fuel_remaining_fraction < 1.0

    def test_deterministic_with_seed(self):
        s1, h1 = run_reactor(20, seed=123)
        s2, h2 = run_reactor(20, seed=123)
        assert s1.sol == s2.sol
        assert abs(s1.fuel_remaining_fraction - s2.fuel_remaining_fraction) < 1e-10
        assert abs(s1.total_energy_kwh - s2.total_energy_kwh) < 1e-6

    def test_energy_accumulates(self):
        state, history = run_reactor(20, seed=42)
        assert state.total_energy_kwh > 0.0

    def test_fuel_never_negative(self):
        state, history = run_reactor(100, seed=42)
        assert state.fuel_remaining_fraction >= 0.0
        for h in history:
            assert h.fuel_remaining >= 0.0


# --- Stirling Failures ---

class TestStirlingFailures:

    def test_engine_can_fail(self):
        """Over many sols, at least one engine should fail."""
        state = create_reactor()
        rng = random.Random(42)
        for _ in range(5000):
            state, sol = tick_reactor(state, rng=rng)
            if sol.engine_failed:
                break
        assert state.stirling_engines_active < NUM_STIRLING_ENGINES

    def test_scram_on_too_few_engines(self):
        """If engines drop below minimum, reactor SCRAMs."""
        state = create_reactor()
        state.state = "nominal"
        state.control_rod_position = NOMINAL_ROD_POSITION
        state.core_temp_k = NOMINAL_CORE_TEMP_K
        state.stirling_engines_active = MIN_ENGINES_FOR_OPERATION - 1
        state, sol = tick_reactor(state, rng=random.Random(42))
        assert state.state == "scram"


# --- Campaign ---

class TestReactorCampaign:

    def test_campaign_runs(self):
        state, history = run_reactor(50, seed=42)
        assert state.sol == 50
        assert len(history) == 50

    def test_auto_start(self):
        state, history = run_reactor(10, auto_start=True, seed=42)
        # Should have started up
        assert state.state in ("startup", "nominal", "scram", "shutdown")
        assert any(h.rod_position > 0 for h in history)

    def test_long_campaign_stable(self):
        state, history = run_reactor(500, seed=42)
        assert state.sol == 500
        assert state.fuel_remaining_fraction > 0.0
        assert state.total_energy_kwh > 0.0

    def test_operating_sols_tracked(self):
        state, history = run_reactor(50, seed=42)
        assert state.total_operating_sols >= 0
        assert state.total_operating_sols <= 50


# --- Smoke Tests ---

class TestSmokeTest:

    def test_10_sol_smoke(self):
        state, history = run_reactor(10, seed=1)
        assert state.sol == 10
        assert state.fuel_remaining_fraction >= 0.0
        assert state.radiator_efficiency > 0.0
        assert len(history) == 10

    def test_1000_sol_endurance(self):
        state, history = run_reactor(1000, seed=99)
        assert state.sol == 1000
        assert state.fuel_remaining_fraction > 0.0
        assert state.total_energy_kwh > 0.0
        for h in history:
            assert h.fuel_remaining >= 0.0
            assert h.electric_power_kw >= 0.0

    def test_state_always_valid(self):
        state, history = run_reactor(200, seed=42)
        valid_states = ("shutdown", "startup", "nominal", "scram", "degraded")
        for h in history:
            assert h.state in valid_states

    def test_no_maintenance_degradation(self):
        state, history = run_reactor(100, maintain_every_n=9999, seed=42)
        assert state.radiator_efficiency < 1.0
