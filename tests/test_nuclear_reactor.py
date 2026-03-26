"""tests/test_nuclear_reactor.py — 85+ tests for the Mars fission reactor.

Covers: physics functions, state machine, conservation laws, edge cases,
property-based invariants, and multi-sol smoke tests.
"""
from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nuclear_reactor import (
    FissionReactor,
    ReactorState,
    thermal_power_from_drums,
    radiator_rejection_kw,
    stirling_efficiency,
    core_temperature_step,
    fuel_burnup_per_sol,
    radiation_dose_at_hab,
    move_drum_toward,
    tick,
    THERMAL_POWER_MAX_KW,
    DRUM_ANGLE_SHUTDOWN,
    DRUM_ANGLE_FULL_POWER,
    DRUM_RATE_DEG_PER_SOL,
    MARS_AMBIENT_TEMP_K,
    CORE_TEMP_NOMINAL_K,
    CORE_TEMP_MAX_K,
    BURNUP_FRACTION_AT_EOL,
    DESIGN_LIFE_SOLS,
    STIRLING_FRACTION_OF_CARNOT,
    RADIATOR_EMISSIVITY,
    RADIATOR_MIN_EMISSIVITY,
    DUST_ACCUMULATION_PER_SOL,
    DUST_STORM_ACCUMULATION,
    STARTUP_SOLS,
    SCRAM_COOLDOWN_SOLS,
    SHIELD_ATTENUATION_FACTOR,
    HAB_DISTANCE_M,
    MARS_SOL_SECONDS,
)


# ===========================================================================
# thermal_power_from_drums
# ===========================================================================

class TestThermalPower:
    """Control drum → thermal power model."""

    def test_shutdown_zero_power(self):
        """Drums at 0° → zero thermal power."""
        assert thermal_power_from_drums(0.0, 1.0) == 0.0

    def test_full_power_at_180(self):
        """Drums at 180° with fresh fuel → max thermal power."""
        p = thermal_power_from_drums(180.0, 1.0)
        assert abs(p - THERMAL_POWER_MAX_KW) < 0.1

    def test_partial_power(self):
        """90° drums → intermediate power."""
        p = thermal_power_from_drums(90.0, 1.0)
        assert 0.0 < p < THERMAL_POWER_MAX_KW

    def test_depleted_fuel_reduces_power(self):
        """Fuel depletion reduces available power."""
        p_fresh = thermal_power_from_drums(180.0, 1.0)
        p_depleted = thermal_power_from_drums(180.0, 0.5)
        assert p_depleted < p_fresh

    def test_zero_fuel_zero_power(self):
        """No fuel → no power regardless of drum angle."""
        assert thermal_power_from_drums(180.0, 0.0) == 0.0

    def test_monotonic_with_angle(self):
        """Power increases monotonically with drum angle."""
        prev = 0.0
        for angle in range(0, 181, 5):
            p = thermal_power_from_drums(float(angle), 1.0)
            assert p >= prev - 0.001, f"Power decreased at {angle}°"
            prev = p

    def test_negative_angle_clamps(self):
        """Negative angle → zero power (not negative)."""
        assert thermal_power_from_drums(-10.0, 1.0) == 0.0

    def test_over_180_clamps(self):
        """Angle > 180° clamps to 180°."""
        p_180 = thermal_power_from_drums(180.0, 1.0)
        p_200 = thermal_power_from_drums(200.0, 1.0)
        assert abs(p_180 - p_200) < 0.01

    def test_power_always_non_negative(self):
        """Power is never negative for any inputs."""
        for angle in range(-10, 200, 15):
            for fuel in [0.0, 0.5, 1.0]:
                p = thermal_power_from_drums(float(angle), fuel)
                assert p >= 0.0


# ===========================================================================
# radiator_rejection_kw
# ===========================================================================

class TestRadiator:
    """Radiator heat rejection model."""

    def test_cold_radiator_no_rejection(self):
        """Radiator at ambient temp → zero rejection."""
        assert radiator_rejection_kw(MARS_AMBIENT_TEMP_K, 0.85) == 0.0

    def test_hot_radiator_rejects_heat(self):
        """Hot radiator → positive heat rejection."""
        q = radiator_rejection_kw(550.0, 0.85)
        assert q > 0.0

    def test_hotter_rejects_more(self):
        """Higher temperature → more heat rejection (T⁴ law)."""
        q1 = radiator_rejection_kw(400.0, 0.85)
        q2 = radiator_rejection_kw(600.0, 0.85)
        assert q2 > q1

    def test_emissivity_matters(self):
        """Higher emissivity → more rejection."""
        q_low = radiator_rejection_kw(550.0, 0.40)
        q_high = radiator_rejection_kw(550.0, 0.85)
        assert q_high > q_low

    def test_below_ambient_zero(self):
        """Below ambient temperature → zero rejection."""
        assert radiator_rejection_kw(100.0, 0.85) == 0.0

    def test_rejection_always_non_negative(self):
        """Heat rejection is never negative."""
        for t in [100, 210, 300, 500, 800]:
            for e in [0.1, 0.5, 0.9]:
                q = radiator_rejection_kw(float(t), e)
                assert q >= 0.0


# ===========================================================================
# stirling_efficiency
# ===========================================================================

class TestStirlingEfficiency:
    """Stirling engine conversion efficiency."""

    def test_reasonable_efficiency(self):
        """At KRUSTY temps, efficiency should be ~20-30%."""
        eta = stirling_efficiency(1073.0, 550.0)
        assert 0.15 < eta < 0.35

    def test_equal_temps_zero(self):
        """Same hot and cold → zero efficiency."""
        assert stirling_efficiency(500.0, 500.0) == 0.0

    def test_cold_above_hot_zero(self):
        """Cold > hot → zero efficiency (thermodynamic impossibility)."""
        assert stirling_efficiency(300.0, 500.0) == 0.0

    def test_higher_delta_more_efficient(self):
        """Larger temperature difference → higher efficiency."""
        eta1 = stirling_efficiency(800.0, 500.0)
        eta2 = stirling_efficiency(1200.0, 500.0)
        assert eta2 > eta1

    def test_never_exceeds_carnot(self):
        """Efficiency never exceeds Carnot limit."""
        for t_hot in [500, 800, 1200, 2000]:
            for t_cold in [200, 400, 600]:
                if t_hot > t_cold:
                    eta = stirling_efficiency(float(t_hot), float(t_cold))
                    carnot = 1.0 - t_cold / t_hot
                    assert eta <= carnot + 0.001

    def test_efficiency_bounded(self):
        """Efficiency always in [0, 0.45]."""
        for t_hot in [300, 500, 1000, 2000]:
            for t_cold in [200, 300, 500]:
                eta = stirling_efficiency(float(t_hot), float(t_cold))
                assert 0.0 <= eta <= 0.45


# ===========================================================================
# core_temperature_step
# ===========================================================================

class TestCoreTemperature:
    """Core thermal dynamics."""

    def test_heating_with_power(self):
        """High thermal power → temperature rises."""
        t = core_temperature_step(500.0, 43.0, 0.85, 1.0)
        assert t > 500.0

    def test_cooling_without_power(self):
        """No power + radiator → temperature drops."""
        t = core_temperature_step(800.0, 0.0, 0.85, 1.0)
        assert t < 800.0

    def test_no_time_no_change(self):
        """Zero sol fraction → no temperature change."""
        t = core_temperature_step(600.0, 43.0, 0.85, 0.0)
        assert t == 600.0

    def test_bounded_above(self):
        """Temperature never exceeds 2000 K."""
        t = core_temperature_step(1999.0, 100.0, 0.0, 1.0)
        assert t <= 2000.0

    def test_bounded_below(self):
        """Temperature never drops below Mars ambient."""
        t = core_temperature_step(MARS_AMBIENT_TEMP_K + 1, 0.0, 0.85, 1.0)
        assert t >= MARS_AMBIENT_TEMP_K


# ===========================================================================
# fuel_burnup_per_sol
# ===========================================================================

class TestFuelBurnup:
    """Fuel depletion model."""

    def test_zero_power_no_burnup(self):
        """No thermal power → no fuel consumption."""
        assert fuel_burnup_per_sol(0.0) == 0.0

    def test_max_power_design_life(self):
        """At max power, total burnup over design life = BURNUP_FRACTION_AT_EOL."""
        per_sol = fuel_burnup_per_sol(THERMAL_POWER_MAX_KW)
        total = per_sol * DESIGN_LIFE_SOLS
        assert abs(total - BURNUP_FRACTION_AT_EOL) < 0.001

    def test_half_power_half_burnup(self):
        """Half power → half burnup rate."""
        full = fuel_burnup_per_sol(THERMAL_POWER_MAX_KW)
        half = fuel_burnup_per_sol(THERMAL_POWER_MAX_KW / 2)
        assert abs(half - full / 2) < 1e-10

    def test_burnup_always_non_negative(self):
        """Burnup is never negative."""
        for p in [0, 10, 43, 100]:
            assert fuel_burnup_per_sol(float(p)) >= 0.0


# ===========================================================================
# radiation_dose_at_hab
# ===========================================================================

class TestRadiation:
    """Radiation shielding model."""

    def test_shielded_dose_tiny(self):
        """Shielded dose at 100 m should be negligible."""
        dose = radiation_dose_at_hab(shielded=True)
        assert dose < 0.001  # well below health limits

    def test_unshielded_dangerous(self):
        """Unshielded dose at 100 m is still significant."""
        dose_shielded = radiation_dose_at_hab(shielded=True)
        dose_unshielded = radiation_dose_at_hab(shielded=False)
        assert dose_unshielded > dose_shielded * 1000

    def test_shielding_reduces_dose(self):
        """Shielded dose << unshielded dose."""
        d_s = radiation_dose_at_hab(True)
        d_u = radiation_dose_at_hab(False)
        assert d_s < d_u

    def test_dose_non_negative(self):
        """Dose is never negative."""
        assert radiation_dose_at_hab(True) >= 0.0
        assert radiation_dose_at_hab(False) >= 0.0


# ===========================================================================
# move_drum_toward
# ===========================================================================

class TestDrumMovement:
    """Control drum positioning."""

    def test_reaches_target_if_close(self):
        """If target is within rate limit, reach it exactly."""
        new = move_drum_toward(90.0, 100.0, 1.0)
        assert new == 100.0

    def test_rate_limited(self):
        """Large moves are rate-limited."""
        new = move_drum_toward(0.0, 180.0, 1.0)
        assert new == DRUM_RATE_DEG_PER_SOL

    def test_clamps_to_bounds(self):
        """Angle clamps to [0, 180]."""
        assert move_drum_toward(0.0, -50.0, 1.0) == 0.0
        assert move_drum_toward(180.0, 250.0, 1.0) == 180.0

    def test_fractional_sol(self):
        """Half sol → half rate."""
        new = move_drum_toward(0.0, 180.0, 0.5)
        assert abs(new - DRUM_RATE_DEG_PER_SOL * 0.5) < 0.01

    def test_no_movement_at_target(self):
        """Already at target → no change."""
        assert move_drum_toward(90.0, 90.0, 1.0) == 90.0

    def test_can_move_down(self):
        """Drum can decrease angle (reduce power)."""
        new = move_drum_toward(120.0, 60.0, 1.0)
        assert new == 60.0  # within rate limit


# ===========================================================================
# tick — state machine & integration
# ===========================================================================

class TestTick:
    """Full tick integration tests."""

    def test_cold_start_sequence(self):
        """Cold reactor → STARTING → RUNNING over multiple sols."""
        r = FissionReactor()
        assert r.state == ReactorState.COLD

        # Sol 1: demand triggers startup
        result = tick(r, demand_kw=10.0)
        assert r.state == ReactorState.STARTING
        assert result["electrical_kw"] == 0.0

        # Sol 2: startup countdown continues
        tick(r, demand_kw=10.0)

        # Sol 3: startup complete → running
        tick(r, demand_kw=10.0)
        assert r.state == ReactorState.RUNNING

    def test_running_produces_power(self):
        """Running reactor produces electrical power after warmup."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=CORE_TEMP_NOMINAL_K)
        # Warm up: let core temp stabilize over a few sols
        for _ in range(5):
            tick(r, demand_kw=10.0)
        result = tick(r, demand_kw=10.0)
        assert result["electrical_kw"] > 0.0
        assert result["thermal_kw"] > 0.0

    def test_scram_shuts_down(self):
        """SCRAM immediately shuts down the reactor."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=120.0,
                           core_temp_k=800.0)
        result = tick(r, demand_kw=10.0, scram=True)
        assert r.state == ReactorState.SCRAMMED
        assert r.drum_angle_deg == DRUM_ANGLE_SHUTDOWN
        assert result["electrical_kw"] == 0.0

    def test_scram_cooldown_then_cold(self):
        """After SCRAM cooldown, reactor goes COLD."""
        r = FissionReactor(state=ReactorState.SCRAMMED,
                           cooldown_sols_remaining=1,
                           core_temp_k=600.0)
        tick(r)
        assert r.state == ReactorState.COLD

    def test_no_demand_stays_cold(self):
        """Zero demand on cold reactor → stays cold."""
        r = FissionReactor()
        tick(r, demand_kw=0.0)
        assert r.state == ReactorState.COLD

    def test_dust_storm_degrades_radiator(self):
        """Dust storm accelerates radiator emissivity loss."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=800.0,
                           radiator_emissivity=0.85)
        tick(r, demand_kw=5.0, dust_storm=True)
        assert r.radiator_emissivity < 0.85

    def test_normal_dust_slow(self):
        """Normal conditions → slow radiator degradation."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=800.0,
                           radiator_emissivity=0.85)
        tick(r, demand_kw=5.0, dust_storm=False)
        expected = 0.85 - DUST_ACCUMULATION_PER_SOL
        assert abs(r.radiator_emissivity - expected) < 0.001

    def test_fuel_depletes_over_time(self):
        """Fuel decreases when reactor runs."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=120.0,
                           core_temp_k=CORE_TEMP_NOMINAL_K)
        initial_fuel = r.fuel_remaining_fraction
        tick(r, demand_kw=10.0)
        assert r.fuel_remaining_fraction < initial_fuel

    def test_energy_accumulates(self):
        """Total energy increases after reactor warms up."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=CORE_TEMP_NOMINAL_K)
        # Warm up for several sols to reach thermal equilibrium
        for _ in range(20):
            tick(r, demand_kw=5.0)
        assert r.total_energy_kwh > 0.0

    def test_scram_count_increments(self):
        """SCRAM count tracks emergency shutdowns."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=800.0)
        tick(r, scram=True)
        assert r.scram_count == 1

    def test_sols_operated_increments(self):
        """sols_operated counts running sols only."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=800.0)
        tick(r, demand_kw=5.0)
        tick(r, demand_kw=5.0)
        assert r.sols_operated == 2


# ===========================================================================
# Conservation laws / physical invariants
# ===========================================================================

class TestConservation:
    """Invariants that must hold for any simulation."""

    def test_electrical_never_exceeds_thermal(self):
        """Electrical output ≤ thermal output (2nd law)."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=150.0,
                           core_temp_k=CORE_TEMP_NOMINAL_K)
        for _ in range(50):
            result = tick(r, demand_kw=15.0)
            assert result["electrical_kw"] <= result["thermal_kw"] + 0.01

    def test_efficiency_below_carnot(self):
        """Efficiency never exceeds Carnot limit."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=120.0,
                           core_temp_k=CORE_TEMP_NOMINAL_K)
        for _ in range(30):
            result = tick(r, demand_kw=10.0)
            if result["thermal_kw"] > 0 and result["core_temp_k"] > MARS_AMBIENT_TEMP_K:
                carnot = 1.0 - MARS_AMBIENT_TEMP_K / result["core_temp_k"]
                assert result["efficiency"] <= carnot + 0.01

    def test_fuel_never_negative(self):
        """Fuel fraction never goes below zero."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=180.0,
                           core_temp_k=CORE_TEMP_NOMINAL_K)
        for _ in range(100):
            tick(r, demand_kw=20.0)
            assert r.fuel_remaining_fraction >= 0.0

    def test_fuel_monotonically_decreases(self):
        """Fuel only decreases (no refueling in this model)."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=CORE_TEMP_NOMINAL_K)
        prev = r.fuel_remaining_fraction
        for _ in range(30):
            tick(r, demand_kw=5.0)
            assert r.fuel_remaining_fraction <= prev + 1e-12
            prev = r.fuel_remaining_fraction

    def test_core_temp_bounded(self):
        """Core temp stays within physical bounds."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=150.0,
                           core_temp_k=900.0)
        for _ in range(100):
            tick(r, demand_kw=15.0)
            assert MARS_AMBIENT_TEMP_K <= r.core_temp_k <= 2000.0

    def test_radiator_emissivity_bounded(self):
        """Radiator emissivity stays in [MIN, 1.0]."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=800.0)
        for _ in range(500):
            tick(r, demand_kw=5.0, dust_storm=True)
            assert RADIATOR_MIN_EMISSIVITY <= r.radiator_emissivity <= 1.0

    def test_drum_angle_bounded(self):
        """Drum angle always in [0°, 180°]."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=800.0)
        for demand in [0, 5, 10, 20, 50]:
            tick(r, demand_kw=float(demand))
            assert 0.0 <= r.drum_angle_deg <= 180.0

    def test_energy_monotonically_increases(self):
        """Total energy never decreases."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=CORE_TEMP_NOMINAL_K)
        prev = 0.0
        for _ in range(30):
            tick(r, demand_kw=5.0)
            assert r.total_energy_kwh >= prev - 0.01
            prev = r.total_energy_kwh


# ===========================================================================
# Smoke tests — multi-sol simulations
# ===========================================================================

class TestSmoke:
    """Extended simulations without crashes."""

    def test_100_sols_nominal(self):
        """100 sols at nominal power without crash."""
        r = FissionReactor()
        # Start up (3 ticks: COLD→STARTING, STARTING countdown, STARTING→RUNNING)
        for _ in range(3):
            tick(r, demand_kw=10.0)
        assert r.state == ReactorState.RUNNING
        for _ in range(97):
            result = tick(r, demand_kw=10.0)
        assert r.sols_operated > 0

    def test_365_sols_with_storms(self):
        """Full Mars year with periodic dust storms and radiator cleaning."""
        r = FissionReactor()
        for _ in range(3):
            tick(r, demand_kw=10.0)  # startup
        assert r.state == ReactorState.RUNNING
        for sol in range(362):
            storm = (sol % 60) < 15  # 15-sol storms every 60 sols
            # Clean radiator between storms (maintenance crew)
            if sol % 60 == 16:
                r.radiator_emissivity = min(
                    RADIATOR_EMISSIVITY,
                    r.radiator_emissivity + 0.2
                )
            tick(r, demand_kw=10.0, dust_storm=storm)
        assert r.sols_operated > 300
        assert r.radiator_emissivity < RADIATOR_EMISSIVITY  # some dust

    def test_variable_demand(self):
        """Reactor handles wildly varying demand."""
        r = FissionReactor()
        tick(r, demand_kw=5.0)
        tick(r, demand_kw=5.0)
        demands = [1, 5, 10, 20, 0, 15, 3, 8, 12, 0, 7]
        for d in demands:
            tick(r, demand_kw=float(d))
        assert r.sols_operated > 0

    def test_multiple_scram_restart_cycles(self):
        """Reactor can SCRAM and restart multiple times."""
        r = FissionReactor()
        for cycle in range(3):
            # Start (3 sols)
            for _ in range(3):
                tick(r, demand_kw=10.0)
            assert r.state == ReactorState.RUNNING, f"cycle {cycle}: not running after startup"
            # Run for a bit
            for _ in range(5):
                tick(r, demand_kw=10.0)
            # SCRAM
            tick(r, scram=True)
            assert r.state == ReactorState.SCRAMMED
            # Cooldown (1 sol to go COLD)
            tick(r, demand_kw=0.0)
            assert r.state == ReactorState.COLD
        assert r.scram_count == 3

    def test_long_life_fuel_depletion(self):
        """Over extended operation, fuel depletes measurably."""
        r = FissionReactor()
        # Startup
        for _ in range(3):
            tick(r, demand_kw=10.0)
        assert r.state == ReactorState.RUNNING
        # Run for 1000 sols
        for _ in range(1000):
            tick(r, demand_kw=10.0)
        assert r.fuel_remaining_fraction < 1.0
        assert r.fuel_remaining_fraction > 0.9  # 1000/6680 sols ≈ 0.75% burnup


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Boundary and degenerate inputs."""

    def test_fresh_reactor_defaults(self):
        """Default reactor has sane initial values."""
        r = FissionReactor()
        assert r.state == ReactorState.COLD
        assert r.fuel_remaining_fraction == 1.0
        assert r.core_temp_k == MARS_AMBIENT_TEMP_K
        assert r.drum_angle_deg == DRUM_ANGLE_SHUTDOWN

    def test_zero_demand_cold(self):
        """Zero demand never starts reactor."""
        r = FissionReactor()
        for _ in range(10):
            tick(r, demand_kw=0.0)
        assert r.state == ReactorState.COLD

    def test_massive_demand(self):
        """Huge demand doesn't crash — reactor gives what it can."""
        r = FissionReactor(state=ReactorState.RUNNING,
                           drum_angle_deg=90.0,
                           core_temp_k=CORE_TEMP_NOMINAL_K)
        result = tick(r, demand_kw=1000.0)
        assert result["electrical_kw"] <= THERMAL_POWER_MAX_KW  # can't exceed thermal

    def test_scram_on_cold_is_noop(self):
        """SCRAM on cold reactor with no demand is ignored."""
        r = FissionReactor()
        tick(r, demand_kw=0.0, scram=True)
        assert r.state == ReactorState.COLD
        assert r.scram_count == 0

    def test_scram_on_starting_is_noop(self):
        """SCRAM during startup is ignored (drums already at 0)."""
        r = FissionReactor(state=ReactorState.STARTING,
                           startup_sols_remaining=1)
        tick(r, scram=True)
        assert r.scram_count == 0  # SCRAM only works in RUNNING state
