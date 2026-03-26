"""Tests for oxygen_tank.py — Mars Colony Cryogenic LOX Storage."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.oxygen_tank import (
    TankState, TickResult,
    sphere_surface_area, tank_volume_m3, heat_leak_watts,
    boiloff_rate_kg_s, boiloff_mass_kg, reliquefaction_power_w,
    pressure_from_boiloff, ullage_volume, crew_consumption_kg,
    reserve_sols, tick_tank, create_oxygen_tank,
    LOX_BOILING_POINT_K, LOX_DENSITY_KG_M3, LATENT_HEAT_KJ_KG,
    MARS_AMBIENT_K, MLI_U_COEFF_W_M2_K, CRYO_COP, CRYO_MAX_POWER_W,
    RELIEF_PRESSURE_KPA, OPERATING_PRESSURE_KPA, MIN_PRESSURE_KPA,
    O2_PER_PERSON_PER_SOL_KG, O2_PER_EVA_KG, RESERVE_SOLS,
    SECONDS_PER_SOL,
    _clamp,
)


# ── Clamp ───────────────────────────────────────────────────────────

class TestClamp:
    def test_within(self):
        assert _clamp(5.0, 0.0, 10.0) == 5.0

    def test_below(self):
        assert _clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above(self):
        assert _clamp(15.0, 0.0, 10.0) == 10.0


# ── TankState clamping ──────────────────────────────────────────────

class TestTankState:
    def test_defaults(self):
        s = TankState()
        assert s.lox_kg == 500.0
        assert s.crew_count == 6

    def test_lox_clamped_to_capacity(self):
        s = TankState(lox_kg=2000.0, capacity_kg=1000.0)
        assert s.lox_kg == 1000.0

    def test_lox_clamped_nonneg(self):
        s = TankState(lox_kg=-50.0)
        assert s.lox_kg == 0.0

    def test_capacity_minimum(self):
        s = TankState(capacity_kg=0.0)
        assert s.capacity_kg == 1.0

    def test_pressure_minimum(self):
        s = TankState(ullage_pressure_kpa=10.0)
        assert s.ullage_pressure_kpa == MIN_PRESSURE_KPA

    def test_ambient_temp_minimum(self):
        s = TankState(ambient_temp_k=50.0)
        assert s.ambient_temp_k == 100.0

    def test_crew_nonneg(self):
        s = TankState(crew_count=-3)
        assert s.crew_count == 0


# ── Sphere surface area ────────────────────────────────────────────

class TestSphereSurfaceArea:
    def test_zero_volume(self):
        assert sphere_surface_area(0.0) == 0.0

    def test_negative_volume(self):
        assert sphere_surface_area(-1.0) == 0.0

    def test_unit_sphere(self):
        v = 4.0 / 3.0 * math.pi
        a = sphere_surface_area(v)
        assert abs(a - 4.0 * math.pi) < 0.01

    def test_positive(self):
        assert sphere_surface_area(1.0) > 0

    def test_scales_as_v_two_thirds(self):
        """Surface area scales as V^(2/3)."""
        a1 = sphere_surface_area(1.0)
        a8 = sphere_surface_area(8.0)
        assert abs(a8 / a1 - 4.0) < 0.01


# ── Tank volume ─────────────────────────────────────────────────────

class TestTankVolume:
    def test_basic(self):
        v = tank_volume_m3(LOX_DENSITY_KG_M3)
        assert abs(v - 1.0) < 0.01

    def test_zero_capacity(self):
        assert tank_volume_m3(0.0) == 0.0

    def test_zero_density(self):
        assert tank_volume_m3(1000.0, 0.0) == 0.0

    def test_1000kg_about_0_88_m3(self):
        v = tank_volume_m3(1000.0)
        assert 0.8 < v < 0.95


# ── Heat leak ───────────────────────────────────────────────────────

class TestHeatLeak:
    def test_zero_area(self):
        assert heat_leak_watts(0.0, MARS_AMBIENT_K) == 0.0

    def test_zero_delta_t(self):
        assert heat_leak_watts(1.0, LOX_BOILING_POINT_K) == 0.0

    def test_positive(self):
        q = heat_leak_watts(1.0, MARS_AMBIENT_K)
        assert q > 0

    def test_proportional_to_area(self):
        q1 = heat_leak_watts(1.0, MARS_AMBIENT_K)
        q2 = heat_leak_watts(2.0, MARS_AMBIENT_K)
        assert abs(q2 / q1 - 2.0) < 0.01

    def test_proportional_to_delta_t(self):
        q1 = heat_leak_watts(1.0, 150.0)
        q2 = heat_leak_watts(1.0, 210.0)
        dt1 = 150.0 - LOX_BOILING_POINT_K
        dt2 = 210.0 - LOX_BOILING_POINT_K
        assert abs(q2 / q1 - dt2 / dt1) < 0.01

    def test_cold_ambient_no_leak(self):
        q = heat_leak_watts(1.0, 80.0)
        assert q == 0.0

    def test_realistic_mars_value(self):
        q = heat_leak_watts(1.0, MARS_AMBIENT_K)
        expected = MLI_U_COEFF_W_M2_K * (MARS_AMBIENT_K - LOX_BOILING_POINT_K)
        assert abs(q - expected) < 0.1


# ── Boil-off ────────────────────────────────────────────────────────

class TestBoiloff:
    def test_rate_zero_heat(self):
        assert boiloff_rate_kg_s(0.0) == 0.0

    def test_rate_positive(self):
        assert boiloff_rate_kg_s(100.0) > 0

    def test_rate_formula(self):
        rate = boiloff_rate_kg_s(213.0)
        assert abs(rate - 0.001) < 1e-6

    def test_rate_zero_latent_heat(self):
        assert boiloff_rate_kg_s(100.0, 0.0) == 0.0

    def test_mass_basic(self):
        m = boiloff_mass_kg(100.0, 3600.0)
        expected = boiloff_rate_kg_s(100.0) * 3600.0
        assert abs(m - expected) < 1e-6

    def test_mass_zero_time(self):
        assert boiloff_mass_kg(100.0, 0.0) == 0.0

    def test_mass_negative_time(self):
        assert boiloff_mass_kg(100.0, -100.0) == 0.0


# ── Reliquefaction power ────────────────────────────────────────────

class TestReliquefaction:
    def test_basic(self):
        p = reliquefaction_power_w(100.0)
        assert abs(p - 100.0 / CRYO_COP) < 0.1

    def test_zero_load(self):
        assert reliquefaction_power_w(0.0) == 0.0

    def test_zero_cop(self):
        assert reliquefaction_power_w(100.0, 0.0) == float('inf')

    def test_higher_cop_less_power(self):
        p1 = reliquefaction_power_w(100.0, 0.05)
        p2 = reliquefaction_power_w(100.0, 0.10)
        assert p2 < p1


# ── Pressure from boil-off ──────────────────────────────────────────

class TestPressure:
    def test_no_boiloff_no_change(self):
        p = pressure_from_boiloff(150.0, 0.0, 1.0)
        assert p == 150.0

    def test_boiloff_increases_pressure(self):
        assert pressure_from_boiloff(150.0, 1.0, 0.5) > 150.0

    def test_zero_volume(self):
        assert pressure_from_boiloff(150.0, 1.0, 0.0) == 150.0

    def test_larger_volume_less_rise(self):
        p1 = pressure_from_boiloff(150.0, 1.0, 0.5)
        p2 = pressure_from_boiloff(150.0, 1.0, 2.0)
        assert p2 < p1

    def test_never_below_minimum(self):
        assert pressure_from_boiloff(50.0, 0.0, 1.0) >= MIN_PRESSURE_KPA


# ── Ullage volume ───────────────────────────────────────────────────

class TestUllageVolume:
    def test_empty_tank(self):
        vol = tank_volume_m3(1000.0)
        ull = ullage_volume(vol, 0.0)
        assert abs(ull - vol) < 0.001

    def test_full_tank_minimum_ullage(self):
        vol = tank_volume_m3(1000.0)
        ull = ullage_volume(vol, 1000.0)
        assert ull == 0.001

    def test_half_full(self):
        vol = tank_volume_m3(1000.0)
        ull = ullage_volume(vol, 500.0)
        assert 0 < ull < vol

    def test_always_positive(self):
        assert ullage_volume(0.5, 1000.0) >= 0.001


# ── Crew consumption ────────────────────────────────────────────────

class TestCrewConsumption:
    def test_basic(self):
        c = crew_consumption_kg(6)
        assert abs(c - 6 * O2_PER_PERSON_PER_SOL_KG) < 0.01

    def test_with_eva(self):
        c = crew_consumption_kg(6, 2)
        expected = 6 * O2_PER_PERSON_PER_SOL_KG + 2 * O2_PER_EVA_KG
        assert abs(c - expected) < 0.01

    def test_zero_crew(self):
        assert crew_consumption_kg(0) == 0.0

    def test_negative_crew_clamped(self):
        assert crew_consumption_kg(-3) == 0.0


# ── Reserve sols ────────────────────────────────────────────────────

class TestReserveSols:
    def test_basic(self):
        rs = reserve_sols(500.0, 6)
        expected = 500.0 / (6 * O2_PER_PERSON_PER_SOL_KG)
        assert abs(rs - expected) < 0.1

    def test_zero_crew_infinite(self):
        assert reserve_sols(500.0, 0) == float('inf')

    def test_zero_lox(self):
        assert reserve_sols(0.0, 6) == 0.0

    def test_negative_lox(self):
        assert reserve_sols(-10.0, 6) == 0.0


# ── Tick function ───────────────────────────────────────────────────

class TestTickTank:
    def test_basic_tick(self):
        state = create_oxygen_tank()
        result = tick_tank(state)
        assert result.heat_leak_w > 0
        assert result.fill_fraction > 0
        assert state.sol == 1

    def test_boiloff_without_cryo(self):
        state = create_oxygen_tank()
        state.cryo_enabled = False
        result = tick_tank(state)
        assert result.boiloff_kg > 0

    def test_cryo_reduces_boiloff(self):
        s1 = create_oxygen_tank()
        s1.cryo_enabled = False
        r1 = tick_tank(s1)
        s2 = create_oxygen_tank()
        s2.cryo_enabled = True
        r2 = tick_tank(s2)
        assert r2.boiloff_kg <= r1.boiloff_kg

    def test_cryo_uses_energy(self):
        state = create_oxygen_tank()
        result = tick_tank(state)
        assert result.energy_used_wh > 0
        assert result.cryo_power_w > 0

    def test_delivery_accepted(self):
        state = create_oxygen_tank()
        state.lox_kg = 400.0
        tick_tank(state, delivered_kg=50.0)
        assert state.delivered_total_kg == 50.0

    def test_delivery_capped_at_capacity(self):
        state = create_oxygen_tank()
        state.lox_kg = 990.0
        tick_tank(state, delivered_kg=100.0)
        assert state.delivered_total_kg == 10.0  # only 10 kg of headroom

    def test_crew_consumption(self):
        state = create_oxygen_tank()
        result = tick_tank(state)
        expected = 6 * O2_PER_PERSON_PER_SOL_KG
        assert abs(result.consumed_kg - expected) < 0.1

    def test_eva_extra_consumption(self):
        s1 = create_oxygen_tank()
        r1 = tick_tank(s1, eva_count=0)
        s2 = create_oxygen_tank()
        r2 = tick_tank(s2, eva_count=3)
        assert r2.consumed_kg > r1.consumed_kg

    def test_low_reserve_warning(self):
        state = create_oxygen_tank()
        state.lox_kg = 20.0
        result = tick_tank(state)
        assert "LOW" in result.warning or "CRITICAL" in result.warning

    def test_empty_tank_never_negative(self):
        state = create_oxygen_tank()
        state.lox_kg = 0.1
        state.crew_count = 1
        tick_tank(state)
        assert state.lox_kg >= 0.0

    def test_sol_increments(self):
        state = create_oxygen_tank()
        tick_tank(state)
        tick_tank(state)
        assert state.sol == 2

    def test_pressure_relief_triggers(self):
        state = create_oxygen_tank()
        state.ullage_pressure_kpa = RELIEF_PRESSURE_KPA + 50.0
        state.cryo_enabled = False
        tick_tank(state)
        assert state.ullage_pressure_kpa <= RELIEF_PRESSURE_KPA + 10.0


# ── Factory ─────────────────────────────────────────────────────────

class TestFactory:
    def test_standard(self):
        s = create_oxygen_tank("standard")
        assert s.capacity_kg == 1000.0
        assert s.crew_count == 6

    def test_outpost(self):
        s = create_oxygen_tank("outpost")
        assert s.capacity_kg == 200.0
        assert s.crew_count == 2

    def test_emergency(self):
        s = create_oxygen_tank("emergency")
        assert s.capacity_kg == 1500.0
        assert s.cryo_enabled is False

    def test_unknown_defaults(self):
        s = create_oxygen_tank("nonexistent")
        assert s.capacity_kg == 1000.0


# ── Invariants ──────────────────────────────────────────────────────

class TestInvariants:
    def test_10_tick_no_crash(self):
        state = create_oxygen_tank()
        for _ in range(10):
            tick_tank(state, delivered_kg=5.0)
        assert state.sol == 10

    def test_50_tick_no_crash(self):
        state = create_oxygen_tank()
        for i in range(50):
            tick_tank(state, delivered_kg=6.0, eva_count=i % 3)
        assert state.sol == 50

    def test_lox_never_negative(self):
        state = create_oxygen_tank()
        state.lox_kg = 10.0
        state.crew_count = 20
        for _ in range(20):
            tick_tank(state)
            assert state.lox_kg >= 0.0

    def test_fill_fraction_bounded(self):
        state = create_oxygen_tank()
        for i in range(20):
            result = tick_tank(state, delivered_kg=float(i * 10))
            assert 0.0 <= result.fill_fraction <= 1.0

    def test_energy_monotonic(self):
        state = create_oxygen_tank()
        prev = 0.0
        for _ in range(20):
            tick_tank(state, delivered_kg=5.0)
            assert state.total_energy_wh >= prev
            prev = state.total_energy_wh

    def test_consumed_monotonic(self):
        state = create_oxygen_tank()
        prev = 0.0
        for _ in range(20):
            tick_tank(state, delivered_kg=10.0)
            assert state.consumed_total_kg >= prev
            prev = state.consumed_total_kg

    def test_pressure_never_below_minimum(self):
        state = create_oxygen_tank()
        for _ in range(20):
            tick_tank(state, delivered_kg=5.0)
            assert state.ullage_pressure_kpa >= MIN_PRESSURE_KPA

    def test_reserve_sols_nonneg(self):
        state = create_oxygen_tank()
        for _ in range(20):
            result = tick_tank(state, delivered_kg=3.0)
            assert result.reserve_sols >= 0.0

    def test_all_scenarios_run(self):
        for scenario in ["standard", "outpost", "emergency"]:
            state = create_oxygen_tank(scenario)
            for _ in range(10):
                tick_tank(state, delivered_kg=2.0)
            assert state.sol == 10

    def test_mass_conservation(self):
        """initial + delivered = remaining + consumed + boiloff."""
        state = create_oxygen_tank()
        initial = state.lox_kg
        for _ in range(20):
            tick_tank(state, delivered_kg=8.0)
        lhs = initial + state.delivered_total_kg
        rhs = state.lox_kg + state.consumed_total_kg + state.boiloff_total_kg
        assert abs(lhs - rhs) < 0.1

    def test_depletion_without_production(self):
        """Without deliveries, tank empties."""
        state = create_oxygen_tank()
        state.lox_kg = 50.0
        for _ in range(20):
            tick_tank(state, delivered_kg=0.0)
        assert state.lox_kg == 0.0

    def test_boiloff_total_monotonic(self):
        state = create_oxygen_tank()
        state.cryo_enabled = False
        prev = 0.0
        for _ in range(10):
            tick_tank(state, delivered_kg=5.0)
            assert state.boiloff_total_kg >= prev
            prev = state.boiloff_total_kg

    def test_outpost_different_reserves(self):
        outpost = create_oxygen_tank("outpost")
        standard = create_oxygen_tank("standard")
        r_out = reserve_sols(outpost.lox_kg, outpost.crew_count)
        r_std = reserve_sols(standard.lox_kg, standard.crew_count)
        assert r_out != r_std

    def test_steady_state_with_enough_production(self):
        """Enough delivery keeps tank from emptying."""
        state = create_oxygen_tank()
        state.lox_kg = 800.0
        for _ in range(10):
            # 6 crew × 0.84 = 5.04 kg/sol + boiloff ~25 kg/sol → need ~30 kg
            tick_tank(state, delivered_kg=35.0)
        assert state.lox_kg > 0
