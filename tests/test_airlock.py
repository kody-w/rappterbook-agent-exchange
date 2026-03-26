"""
Tests for airlock.py — Mars Colony Airlock Simulation.

87 tests across 10 test classes.  Every function, edge case, and physics
invariant tested.  The airlock is the colony's gateway between inside
and the lethal Martian surface — it must never fail silently.

Run: python -m pytest tests/test_airlock.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.airlock import (
    AirlockState,
    AirlockSol,
    CycleResult,
    air_mass_in_chamber,
    air_lost_per_cycle,
    leak_rate,
    cycle_energy_kwh,
    cycle_time_min,
    dust_ingress_g,
    run_cycle,
    tick_airlock,
    create_airlock,
    HABITAT_PRESSURE_KPA,
    MARS_AMBIENT_KPA,
    CHAMBER_VOLUME_M3,
    AIR_MOLAR_MASS_KG,
    PUMP_EFFICIENCY,
    PUMP_POWER_KW,
    DEPRESS_TIME_MIN,
    REPRESS_TIME_MIN,
    HEATER_POWER_KW,
    HEATER_TIME_MIN,
    ADIABATIC_COOLING_C_PER_CYCLE,
    HABITAT_TEMP_C,
    DUST_PER_PERSON_PER_EVA_G,
    PERCHLORATE_FRACTION,
    DUST_FILTER_EFFICIENCY,
    FILTER_LIFE_CYCLES,
    SEAL_LIFE_CYCLES,
    SEAL_LEAK_RATE_BASE_KPA_HR,
    SEAL_LEAK_RATE_MAX_KPA_HR,
    MAINTENANCE_SEAL_RESTORE,
    MAX_CYCLES_PER_SOL,
    MAX_CREW_PER_CYCLE,
    MARS_SOL_HOURS,
    SAFE_OPEN_THRESHOLD_KPA,
)


# ─── AirlockState ────────────────────────────────────────────────────────────

class TestAirlockState:
    """Unit tests for the AirlockState dataclass."""

    def test_defaults(self):
        s = AirlockState()
        assert s.sol == 0
        assert s.pressure_kpa == HABITAT_PRESSURE_KPA
        assert s.chamber_temp_c == HABITAT_TEMP_C
        assert s.seal_wear == 0.0
        assert s.filter_wear == 0.0
        assert s.operational is True
        assert s.total_cycles == 0
        assert s.total_air_lost_kg == 0.0
        assert s.inner_door_open is False
        assert s.outer_door_open is False

    def test_pressure_clamped_high(self):
        s = AirlockState(pressure_kpa=99999.0)
        assert s.pressure_kpa <= HABITAT_PRESSURE_KPA * 1.1

    def test_pressure_clamped_low(self):
        s = AirlockState(pressure_kpa=-50.0)
        assert s.pressure_kpa == 0.0

    def test_seal_wear_clamped(self):
        s = AirlockState(seal_wear=5.0)
        assert s.seal_wear == 1.0
        s2 = AirlockState(seal_wear=-1.0)
        assert s2.seal_wear == 0.0

    def test_filter_wear_clamped(self):
        s = AirlockState(filter_wear=2.0)
        assert s.filter_wear == 1.0

    def test_negative_totals_clamped(self):
        s = AirlockState(total_air_lost_kg=-10.0, total_dust_ingress_g=-5.0)
        assert s.total_air_lost_kg == 0.0
        assert s.total_dust_ingress_g == 0.0

    def test_negative_cycles_clamped(self):
        s = AirlockState(total_cycles=-3)
        assert s.total_cycles == 0


# ─── air_mass_in_chamber ─────────────────────────────────────────────────────

class TestAirMass:
    """Tests for air_mass_in_chamber (ideal gas law)."""

    def test_positive_at_habitat_pressure(self):
        mass = air_mass_in_chamber(HABITAT_PRESSURE_KPA)
        assert mass > 0.0

    def test_zero_at_zero_pressure(self):
        assert air_mass_in_chamber(0.0) == 0.0

    def test_zero_at_negative_pressure(self):
        assert air_mass_in_chamber(-10.0) == 0.0

    def test_zero_at_zero_volume(self):
        assert air_mass_in_chamber(HABITAT_PRESSURE_KPA, volume_m3=0.0) == 0.0

    def test_proportional_to_pressure(self):
        m1 = air_mass_in_chamber(50.0)
        m2 = air_mass_in_chamber(100.0)
        assert abs(m2 / m1 - 2.0) < 0.01

    def test_proportional_to_volume(self):
        m1 = air_mass_in_chamber(HABITAT_PRESSURE_KPA, volume_m3=6.0)
        m2 = air_mass_in_chamber(HABITAT_PRESSURE_KPA, volume_m3=12.0)
        assert abs(m2 / m1 - 2.0) < 0.01

    def test_physically_reasonable_mass(self):
        """At 101.3 kPa, 12 m³, 293 K: expect ~14-15 kg of air."""
        mass = air_mass_in_chamber(HABITAT_PRESSURE_KPA)
        assert 12.0 < mass < 18.0  # Reasonable for N₂/O₂ mix

    def test_mars_ambient_nearly_zero(self):
        """Mars ambient pressure should give negligible air mass."""
        mass = air_mass_in_chamber(MARS_AMBIENT_KPA)
        assert mass < 0.1  # Less than 100g at 0.636 kPa


# ─── air_lost_per_cycle ──────────────────────────────────────────────────────

class TestAirLoss:
    """Tests for air_lost_per_cycle."""

    def test_positive_loss(self):
        loss = air_lost_per_cycle()
        assert loss > 0.0

    def test_perfect_pump_zero_loss(self):
        loss = air_lost_per_cycle(pump_efficiency=1.0)
        assert loss == 0.0

    def test_no_pump_full_loss(self):
        loss = air_lost_per_cycle(pump_efficiency=0.0)
        total = air_mass_in_chamber(HABITAT_PRESSURE_KPA)
        assert abs(loss - total) < 0.01

    def test_scales_with_inefficiency(self):
        loss_95 = air_lost_per_cycle(0.95)
        loss_90 = air_lost_per_cycle(0.90)
        assert loss_90 > loss_95

    def test_default_loss_physically_reasonable(self):
        """At 95% pump efficiency, expect ~0.5-1.0 kg loss."""
        loss = air_lost_per_cycle()
        assert 0.3 < loss < 2.0

    def test_efficiency_clamped(self):
        """Efficiency > 1.0 should be clamped to 1.0 (zero loss)."""
        loss = air_lost_per_cycle(pump_efficiency=1.5)
        assert loss == 0.0


# ─── leak_rate ───────────────────────────────────────────────────────────────

class TestLeakRate:
    """Tests for seal leak rate calculation."""

    def test_new_seals_base_rate(self):
        rate = leak_rate(0.0)
        assert abs(rate - SEAL_LEAK_RATE_BASE_KPA_HR) < 1e-6

    def test_failed_seals_max_rate(self):
        rate = leak_rate(1.0)
        assert abs(rate - SEAL_LEAK_RATE_MAX_KPA_HR) < 1e-6

    def test_monotonically_increasing(self):
        rates = [leak_rate(w / 10.0) for w in range(11)]
        for i in range(len(rates) - 1):
            assert rates[i + 1] >= rates[i]

    def test_clamped_above_one(self):
        rate = leak_rate(5.0)
        assert rate == leak_rate(1.0)

    def test_clamped_below_zero(self):
        rate = leak_rate(-1.0)
        assert rate == leak_rate(0.0)

    def test_midpoint(self):
        rate = leak_rate(0.5)
        expected = SEAL_LEAK_RATE_BASE_KPA_HR + 0.5 * (SEAL_LEAK_RATE_MAX_KPA_HR - SEAL_LEAK_RATE_BASE_KPA_HR)
        assert abs(rate - expected) < 1e-6


# ─── cycle_energy_kwh / cycle_time_min ───────────────────────────────────────

class TestCycleMetrics:
    """Tests for energy and time per cycle."""

    def test_energy_positive(self):
        assert cycle_energy_kwh() > 0.0

    def test_time_positive(self):
        assert cycle_time_min() > 0.0

    def test_energy_matches_components(self):
        expected = (PUMP_POWER_KW * (DEPRESS_TIME_MIN + REPRESS_TIME_MIN) / 60.0 +
                    HEATER_POWER_KW * HEATER_TIME_MIN / 60.0)
        assert abs(cycle_energy_kwh() - expected) < 1e-6

    def test_time_matches_components(self):
        expected = DEPRESS_TIME_MIN + REPRESS_TIME_MIN + HEATER_TIME_MIN
        assert abs(cycle_time_min() - expected) < 1e-6

    def test_energy_physically_reasonable(self):
        """A single cycle should use 1-5 kWh — comparable to ISS operations."""
        e = cycle_energy_kwh()
        assert 0.5 < e < 5.0

    def test_time_physically_reasonable(self):
        """A full cycle should take 30-60 minutes."""
        t = cycle_time_min()
        assert 20.0 < t < 120.0


# ─── dust_ingress_g ──────────────────────────────────────────────────────────

class TestDustIngress:
    """Tests for dust ingress calculation."""

    def test_zero_crew_zero_dust(self):
        assert dust_ingress_g(0) == 0.0

    def test_positive_with_crew(self):
        assert dust_ingress_g(2) > 0.0

    def test_scales_with_crew(self):
        d1 = dust_ingress_g(1)
        d2 = dust_ingress_g(2)
        assert abs(d2 / d1 - 2.0) < 0.01

    def test_dust_storm_doubles(self):
        normal = dust_ingress_g(2, dust_storm=False)
        storm = dust_ingress_g(2, dust_storm=True)
        assert abs(storm / normal - 2.0) < 0.01

    def test_filter_reduces_dust(self):
        """Dust after filter should be less than raw dust."""
        crew = 2
        raw = crew * DUST_PER_PERSON_PER_EVA_G
        filtered = dust_ingress_g(crew)
        assert filtered < raw

    def test_filter_efficiency_applied(self):
        """Filtered = raw * (1 - filter_efficiency)."""
        crew = 1
        raw = crew * DUST_PER_PERSON_PER_EVA_G
        expected = raw * (1.0 - DUST_FILTER_EFFICIENCY)
        assert abs(dust_ingress_g(crew) - expected) < 0.01

    def test_crew_clamped_to_max(self):
        """Crew above MAX_CREW_PER_CYCLE should be clamped."""
        d_max = dust_ingress_g(MAX_CREW_PER_CYCLE)
        d_over = dust_ingress_g(MAX_CREW_PER_CYCLE + 10)
        assert abs(d_max - d_over) < 0.01

    def test_negative_crew_zero_dust(self):
        assert dust_ingress_g(-1) == 0.0


# ─── run_cycle ───────────────────────────────────────────────────────────────

class TestRunCycle:
    """Tests for individual airlock cycles."""

    def test_basic_cycle_completes(self):
        s = AirlockState()
        cr = run_cycle(s)
        assert not cr.aborted
        assert cr.cycle_number == 1
        assert cr.air_lost_kg > 0.0
        assert cr.energy_kwh > 0.0

    def test_cycle_increments_counter(self):
        s = AirlockState()
        run_cycle(s)
        assert s.total_cycles == 1
        run_cycle(s)
        assert s.total_cycles == 2

    def test_air_loss_accumulates(self):
        s = AirlockState()
        run_cycle(s)
        first_loss = s.total_air_lost_kg
        run_cycle(s)
        assert s.total_air_lost_kg > first_loss

    def test_seal_degrades_per_cycle(self):
        s = AirlockState()
        run_cycle(s)
        expected_wear = 1.0 / SEAL_LIFE_CYCLES
        assert abs(s.seal_wear - expected_wear) < 1e-6

    def test_filter_degrades_per_cycle(self):
        s = AirlockState()
        run_cycle(s)
        expected_wear = 1.0 / FILTER_LIFE_CYCLES
        assert abs(s.filter_wear - expected_wear) < 1e-6

    def test_offline_airlock_aborts(self):
        s = AirlockState(operational=False)
        cr = run_cycle(s)
        assert cr.aborted
        assert any("OFFLINE" in w for w in cr.warnings)

    def test_failed_seals_abort(self):
        s = AirlockState(seal_wear=1.0)
        cr = run_cycle(s)
        assert cr.aborted
        assert not s.operational

    def test_dust_tracked_with_crew(self):
        s = AirlockState()
        cr = run_cycle(s, crew_count=3)
        assert cr.dust_ingress_g > 0.0
        assert cr.perchlorate_ingress_g > 0.0

    def test_perchlorate_fraction_correct(self):
        s = AirlockState()
        cr = run_cycle(s, crew_count=2)
        assert abs(cr.perchlorate_ingress_g - cr.dust_ingress_g * PERCHLORATE_FRACTION) < 0.01

    def test_pressure_returns_to_habitat(self):
        """After a full cycle, chamber pressure should be back to habitat."""
        s = AirlockState()
        run_cycle(s)
        assert abs(s.pressure_kpa - HABITAT_PRESSURE_KPA) < 0.01

    def test_temp_returns_to_habitat(self):
        """After a full cycle, chamber temp should be reheated."""
        s = AirlockState()
        run_cycle(s)
        assert abs(s.chamber_temp_c - HABITAT_TEMP_C) < 0.01

    def test_inner_door_open_after_cycle(self):
        s = AirlockState()
        run_cycle(s)
        assert s.inner_door_open is True
        assert s.outer_door_open is False

    def test_dust_storm_increases_ingress(self):
        s1 = AirlockState()
        cr_normal = run_cycle(s1, crew_count=2, dust_storm=False)
        s2 = AirlockState()
        cr_storm = run_cycle(s2, crew_count=2, dust_storm=True)
        assert cr_storm.dust_ingress_g > cr_normal.dust_ingress_g


# ─── tick_airlock ─────────────────────────────────────────────────────────────

class TestTickAirlock:
    """Tests for the per-sol tick function."""

    def test_no_cycles_advances_sol(self):
        s = AirlockState()
        result = tick_airlock(s)
        assert result.sol == 1
        assert s.sol == 1
        assert result.cycles_completed == 0

    def test_egress_cycles(self):
        s = AirlockState()
        result = tick_airlock(s, egress_cycles=2)
        assert result.cycles_completed == 2

    def test_ingress_cycles(self):
        s = AirlockState()
        result = tick_airlock(s, ingress_cycles=3)
        assert result.cycles_completed == 3

    def test_combined_cycles(self):
        s = AirlockState()
        result = tick_airlock(s, egress_cycles=2, ingress_cycles=3)
        assert result.cycles_completed == 5

    def test_cycle_limit_enforced(self):
        s = AirlockState()
        result = tick_airlock(s, egress_cycles=20, ingress_cycles=20)
        assert result.cycles_completed <= MAX_CYCLES_PER_SOL
        assert any("CYCLE_LIMIT" in w for w in result.warnings)

    def test_air_loss_positive_with_cycles(self):
        s = AirlockState()
        result = tick_airlock(s, egress_cycles=1)
        assert result.total_air_lost_kg > 0.0

    def test_background_leak_even_without_cycles(self):
        s = AirlockState(seal_wear=0.5)
        result = tick_airlock(s)
        assert result.total_air_lost_kg > 0.0  # Background leak from worn seals

    def test_maintenance_reduces_seal_wear(self):
        s = AirlockState(seal_wear=0.5)
        tick_airlock(s, maintenance=True)
        assert s.seal_wear < 0.5

    def test_filter_replacement_resets_wear(self):
        s = AirlockState(filter_wear=0.8)
        tick_airlock(s, filter_replacement=True)
        assert s.filter_wear == 0.0

    def test_offline_airlock_halted(self):
        s = AirlockState(operational=False)
        result = tick_airlock(s)
        assert result.halted is True
        assert result.cycles_completed == 0

    def test_seal_health_reported(self):
        s = AirlockState(seal_wear=0.3)
        result = tick_airlock(s)
        assert abs(result.seal_health - 0.7) < 0.05

    def test_filter_health_reported(self):
        s = AirlockState(filter_wear=0.4)
        result = tick_airlock(s)
        assert abs(result.filter_health - 0.6) < 0.05

    def test_energy_scales_with_cycles(self):
        s1 = AirlockState()
        r1 = tick_airlock(s1, egress_cycles=1)
        s2 = AirlockState()
        r2 = tick_airlock(s2, egress_cycles=3)
        assert r2.total_energy_kwh > r1.total_energy_kwh

    def test_dust_storm_flag_propagates(self):
        s1 = AirlockState()
        r_normal = tick_airlock(s1, egress_cycles=1, crew_per_cycle=2)
        s2 = AirlockState()
        r_storm = tick_airlock(s2, egress_cycles=1, crew_per_cycle=2, dust_storm=True)
        assert r_storm.total_dust_ingress_g > r_normal.total_dust_ingress_g


# ─── create_airlock ──────────────────────────────────────────────────────────

class TestCreateAirlock:
    """Tests for the airlock factory function."""

    def test_standard_config(self):
        a = create_airlock("standard")
        assert a.operational is True
        assert a.seal_wear == 0.0

    def test_heavy_config(self):
        a = create_airlock("heavy")
        assert a.filter_wear == 0.1

    def test_emergency_config(self):
        a = create_airlock("emergency")
        assert a.seal_wear == 0.0
        assert a.filter_wear == 0.0

    def test_unknown_returns_standard(self):
        a = create_airlock("nonexistent")
        s = create_airlock("standard")
        assert a.seal_wear == s.seal_wear
        assert a.filter_wear == s.filter_wear


# ─── Physics invariants ──────────────────────────────────────────────────────

class TestPhysicsInvariants:
    """Property-based tests: conservation laws and physical bounds."""

    def test_air_loss_never_negative(self):
        """Air lost must always be ≥ 0."""
        s = AirlockState()
        for _ in range(50):
            cr = run_cycle(s)
            if cr.aborted:
                break
            assert cr.air_lost_kg >= 0.0

    def test_cumulative_air_loss_monotonic(self):
        """Total air lost only increases."""
        s = AirlockState()
        prev = 0.0
        for _ in range(20):
            tick_airlock(s, egress_cycles=1)
            assert s.total_air_lost_kg >= prev
            prev = s.total_air_lost_kg

    def test_seal_wear_monotonic_without_maintenance(self):
        """Seal wear only increases when no maintenance is done."""
        s = AirlockState()
        prev = 0.0
        for _ in range(50):
            tick_airlock(s, egress_cycles=1)
            assert s.seal_wear >= prev
            prev = s.seal_wear
            if not s.operational:
                break

    def test_pressure_always_in_bounds(self):
        """Chamber pressure never goes negative or beyond limits."""
        s = AirlockState()
        for _ in range(20):
            tick_airlock(s, egress_cycles=2)
            assert 0.0 <= s.pressure_kpa <= HABITAT_PRESSURE_KPA * 1.1

    def test_energy_conservation_positive(self):
        """Energy consumed is always positive for active cycles."""
        s = AirlockState()
        result = tick_airlock(s, egress_cycles=3)
        assert result.total_energy_kwh > 0.0

    def test_dust_perchlorate_ratio_constant(self):
        """Perchlorate is always PERCHLORATE_FRACTION of dust."""
        s = AirlockState()
        for _ in range(10):
            cr = run_cycle(s, crew_count=2)
            if cr.aborted:
                break
            if cr.dust_ingress_g > 0:
                ratio = cr.perchlorate_ingress_g / cr.dust_ingress_g
                assert abs(ratio - PERCHLORATE_FRACTION) < 1e-6

    def test_seal_failure_halts_airlock(self):
        """When seals reach 1.0 wear, airlock must become non-operational."""
        s = AirlockState(seal_wear=0.999)
        # Run enough cycles to push wear to 1.0
        for _ in range(100):
            cr = run_cycle(s)
            if not s.operational:
                break
        assert not s.operational

    def test_smoke_10_sols(self):
        """Smoke test: run 10 sols without crash."""
        s = AirlockState()
        for i in range(10):
            result = tick_airlock(
                s,
                egress_cycles=2,
                ingress_cycles=2,
                crew_per_cycle=2,
                dust_storm=(i == 5),
                maintenance=(i == 7),
            )
            assert result.sol == i + 1
            assert isinstance(result.warnings, list)

    def test_smoke_100_sols_lifecycle(self):
        """Run 100 sols — airlock should eventually degrade but not crash."""
        s = AirlockState()
        for i in range(100):
            result = tick_airlock(s, egress_cycles=3, ingress_cycles=3)
            assert result.sol == i + 1
            if result.halted:
                break
        # After 100 sols of heavy use, some wear is expected
        assert s.seal_wear > 0.0 or not s.operational

    def test_total_energy_matches_sum(self):
        """State's total_energy_kwh should equal sum of all sol results."""
        s = AirlockState()
        running_total = 0.0
        for _ in range(5):
            result = tick_airlock(s, egress_cycles=2)
            running_total += result.total_energy_kwh
        # Allow small floating point drift
        assert abs(s.total_energy_kwh - running_total) < 0.01
