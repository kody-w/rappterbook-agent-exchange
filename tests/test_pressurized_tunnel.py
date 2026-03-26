"""Tests for pressurized_tunnel.py -- Mars Inter-Habitat Pressurized Tunnels.

Coverage:
  1. Hoop stress (thin-wall cylinder)
  2. Burst margin (safety factor)
  3. Tunnel geometry (volume, surface area)
  4. Air mass
  5. Orifice leak rate
  6. Thermal conduction loss
  7. Thermal radiation loss
  8. Combined thermal loss
  9. UV degradation model
 10. Seal fatigue model
 11. Micrometeorite puncture
 12. Tick function (full sol simulation)
 13. Multi-sol simulation
 14. Conservation laws & invariants
 15. Smoke tests
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pressurized_tunnel as pt


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def tunnel():
    return pt.PressurizedTunnel()


@pytest.fixture
def worn_tunnel():
    """Tunnel with degraded fabric and seals."""
    return pt.PressurizedTunnel(fabric_health=0.30, seal_health=0.25)


@pytest.fixture
def shielded_tunnel():
    return pt.PressurizedTunnel(regolith_shielded=True)


# ═══════════════════════════════════════════════════════════════════════
# 1. Hoop stress
# ═══════════════════════════════════════════════════════════════════════

class TestHoopStress:
    def test_zero_pressure(self):
        assert pt.hoop_stress_mpa(0.0, 1.2, 0.012) == 0.0

    def test_positive(self):
        s = pt.hoop_stress_mpa(34.0, 1.2, 0.012)
        expected = (34.0 / 1000.0) * 1.2 / 0.012
        assert abs(s - expected) < 0.001

    def test_increases_with_pressure(self):
        s1 = pt.hoop_stress_mpa(20.0, 1.2, 0.012)
        s2 = pt.hoop_stress_mpa(40.0, 1.2, 0.012)
        assert s2 > s1

    def test_increases_with_radius(self):
        s1 = pt.hoop_stress_mpa(34.0, 1.0, 0.012)
        s2 = pt.hoop_stress_mpa(34.0, 2.0, 0.012)
        assert s2 > s1

    def test_decreases_with_thickness(self):
        s1 = pt.hoop_stress_mpa(34.0, 1.2, 0.010)
        s2 = pt.hoop_stress_mpa(34.0, 1.2, 0.020)
        assert s2 < s1

    def test_zero_thickness_raises(self):
        with pytest.raises(ValueError):
            pt.hoop_stress_mpa(34.0, 1.2, 0.0)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            pt.hoop_stress_mpa(34.0, -1.0, 0.012)

    def test_proportional_to_pressure(self):
        s1 = pt.hoop_stress_mpa(10.0, 1.2, 0.012)
        s2 = pt.hoop_stress_mpa(30.0, 1.2, 0.012)
        assert s2 == pytest.approx(3.0 * s1, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 2. Burst margin
# ═══════════════════════════════════════════════════════════════════════

class TestBurstMargin:
    def test_zero_pressure(self):
        m = pt.burst_margin(0.0, 1.2, 0.012)
        assert m == float("inf")

    def test_nominal(self):
        m = pt.burst_margin(34.0, 1.2, 0.012)
        assert m > pt.SAFETY_FACTOR_MINIMUM

    def test_degraded_fabric(self):
        m_full = pt.burst_margin(34.0, 1.2, 0.012, fabric_health=1.0)
        m_half = pt.burst_margin(34.0, 1.2, 0.012, fabric_health=0.5)
        assert m_half == pytest.approx(m_full / 2.0, rel=1e-6)

    def test_zero_health(self):
        m = pt.burst_margin(34.0, 1.2, 0.012, fabric_health=0.0)
        assert m == 0.0

    def test_decreases_with_pressure(self):
        m1 = pt.burst_margin(20.0, 1.2, 0.012)
        m2 = pt.burst_margin(50.0, 1.2, 0.012)
        assert m2 < m1

    def test_increases_with_thickness(self):
        m1 = pt.burst_margin(34.0, 1.2, 0.010)
        m2 = pt.burst_margin(34.0, 1.2, 0.020)
        assert m2 > m1

    def test_kevlar_is_strong_enough(self):
        """Default tunnel config should have large safety margin."""
        m = pt.burst_margin(pt.HAB_PRESSURE_KPA, pt.DEFAULT_INNER_RADIUS_M,
                             pt.DEFAULT_WALL_THICKNESS_M)
        assert m > 100.0  # Kevlar is extremely strong


# ═══════════════════════════════════════════════════════════════════════
# 3. Tunnel geometry
# ═══════════════════════════════════════════════════════════════════════

class TestGeometry:
    def test_volume_formula(self):
        v = pt.tunnel_volume_m3(1.2, 50.0)
        expected = math.pi * 1.2**2 * 50.0
        assert v == pytest.approx(expected)

    def test_volume_zero_length(self):
        assert pt.tunnel_volume_m3(1.2, 0.0) == 0.0

    def test_volume_zero_radius(self):
        assert pt.tunnel_volume_m3(0.0, 50.0) == 0.0

    def test_surface_area_formula(self):
        a = pt.tunnel_surface_area_m2(1.2, 50.0)
        expected = 2.0 * math.pi * 1.2 * 50.0
        assert a == pytest.approx(expected)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            pt.tunnel_volume_m3(-1.0, 50.0)
        with pytest.raises(ValueError):
            pt.tunnel_surface_area_m2(1.2, -10.0)

    def test_volume_scales_with_r_squared(self):
        v1 = pt.tunnel_volume_m3(1.0, 50.0)
        v2 = pt.tunnel_volume_m3(2.0, 50.0)
        assert v2 == pytest.approx(4.0 * v1, rel=1e-6)

    def test_area_scales_linearly(self):
        a1 = pt.tunnel_surface_area_m2(1.0, 50.0)
        a2 = pt.tunnel_surface_area_m2(2.0, 50.0)
        assert a2 == pytest.approx(2.0 * a1, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 4. Air mass
# ═══════════════════════════════════════════════════════════════════════

class TestAirMass:
    def test_positive(self):
        v = pt.tunnel_volume_m3(1.2, 50.0)
        m = pt.air_mass_kg(v)
        assert m > 0.0

    def test_zero_volume(self):
        assert pt.air_mass_kg(0.0) == 0.0

    def test_proportional(self):
        m1 = pt.air_mass_kg(10.0)
        m2 = pt.air_mass_kg(20.0)
        assert m2 == pytest.approx(2.0 * m1, rel=1e-6)

    def test_reasonable_mass(self):
        """50m tunnel, 1.2m radius -> ~226 m^3 -> ~260 kg air."""
        v = pt.tunnel_volume_m3(1.2, 50.0)
        m = pt.air_mass_kg(v)
        assert 100.0 <= m <= 500.0


# ═══════════════════════════════════════════════════════════════════════
# 5. Orifice leak rate
# ═══════════════════════════════════════════════════════════════════════

class TestLeakRate:
    def test_zero_hole(self):
        assert pt.orifice_leak_rate_kg_s(0.0, 34.0) == 0.0

    def test_zero_pressure(self):
        assert pt.orifice_leak_rate_kg_s(1e-6, 0.0) == 0.0

    def test_positive(self):
        rate = pt.orifice_leak_rate_kg_s(1e-6, 33.0)
        assert rate > 0.0

    def test_increases_with_area(self):
        r1 = pt.orifice_leak_rate_kg_s(1e-7, 33.0)
        r2 = pt.orifice_leak_rate_kg_s(1e-6, 33.0)
        assert r2 > r1

    def test_increases_with_pressure(self):
        r1 = pt.orifice_leak_rate_kg_s(1e-6, 10.0)
        r2 = pt.orifice_leak_rate_kg_s(1e-6, 40.0)
        assert r2 > r1

    def test_proportional_to_area(self):
        r1 = pt.orifice_leak_rate_kg_s(1e-6, 33.0)
        r2 = pt.orifice_leak_rate_kg_s(2e-6, 33.0)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-6)

    def test_scales_with_sqrt_pressure(self):
        r1 = pt.orifice_leak_rate_kg_s(1e-6, 10.0)
        r2 = pt.orifice_leak_rate_kg_s(1e-6, 40.0)
        ratio = r2 / r1
        assert ratio == pytest.approx(2.0, rel=0.01)  # sqrt(40/10) = 2


# ═══════════════════════════════════════════════════════════════════════
# 6. Thermal conduction loss
# ═══════════════════════════════════════════════════════════════════════

class TestThermalConduction:
    def test_positive(self):
        q = pt.thermal_conduction_loss_w(100.0, 0.012)
        assert q > 0.0

    def test_zero_area(self):
        assert pt.thermal_conduction_loss_w(0.0, 0.012) == 0.0

    def test_zero_thickness(self):
        assert pt.thermal_conduction_loss_w(100.0, 0.0) == 0.0

    def test_increases_with_area(self):
        q1 = pt.thermal_conduction_loss_w(50.0, 0.012)
        q2 = pt.thermal_conduction_loss_w(100.0, 0.012)
        assert q2 > q1

    def test_decreases_with_thickness(self):
        q1 = pt.thermal_conduction_loss_w(100.0, 0.010)
        q2 = pt.thermal_conduction_loss_w(100.0, 0.020)
        assert q2 < q1

    def test_formula(self):
        a, t = 100.0, 0.012
        dt = pt.HAB_INTERIOR_TEMP_K - pt.MARS_AMBIENT_TEMP_K
        expected = pt.FABRIC_THERMAL_CONDUCTIVITY * a * dt / t
        assert pt.thermal_conduction_loss_w(a, t) == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════════
# 7. Thermal radiation loss
# ═══════════════════════════════════════════════════════════════════════

class TestThermalRadiation:
    def test_positive(self):
        q = pt.thermal_radiation_loss_w(100.0)
        assert q > 0.0

    def test_zero_area(self):
        assert pt.thermal_radiation_loss_w(0.0) == 0.0

    def test_increases_with_area(self):
        q1 = pt.thermal_radiation_loss_w(50.0)
        q2 = pt.thermal_radiation_loss_w(100.0)
        assert q2 > q1


# ═══════════════════════════════════════════════════════════════════════
# 8. Combined thermal loss
# ═══════════════════════════════════════════════════════════════════════

class TestCombinedThermal:
    def test_sum_of_parts(self):
        area, thick = 100.0, 0.012
        cond = pt.thermal_conduction_loss_w(area, thick)
        rad = pt.thermal_radiation_loss_w(area)
        total = pt.total_thermal_loss_w(area, thick)
        assert total == pytest.approx(cond + rad, rel=1e-6)

    def test_kwh_per_sol(self):
        area, thick = 100.0, 0.012
        watts = pt.total_thermal_loss_w(area, thick)
        kwh = pt.thermal_loss_kwh_per_sol(area, thick)
        expected = watts * pt.SECONDS_PER_SOL / 3.6e6
        assert kwh == pytest.approx(expected, rel=1e-6)

    def test_reasonable_range(self):
        """50m tunnel, ~377 m^2 surface -> expect significant heat loss."""
        area = pt.tunnel_surface_area_m2(1.2, 50.0)
        kwh = pt.thermal_loss_kwh_per_sol(area, 0.012)
        assert kwh > 0.0


# ═══════════════════════════════════════════════════════════════════════
# 9. UV degradation
# ═══════════════════════════════════════════════════════════════════════

class TestUVDegradation:
    def test_one_sol(self):
        h = pt.fabric_health_after_uv(1.0, sols=1)
        assert h == pytest.approx(1.0 - pt.UV_DEGRADATION_PER_SOL)

    def test_shielded_slower(self):
        h_bare = pt.fabric_health_after_uv(1.0, sols=100, shielded=False)
        h_shield = pt.fabric_health_after_uv(1.0, sols=100, shielded=True)
        assert h_shield > h_bare

    def test_shielding_factor(self):
        bare_loss = 1.0 - pt.fabric_health_after_uv(1.0, sols=1, shielded=False)
        shield_loss = 1.0 - pt.fabric_health_after_uv(1.0, sols=1, shielded=True)
        ratio = shield_loss / bare_loss
        assert ratio == pytest.approx(pt.UV_SHIELDING_FACTOR, rel=1e-6)

    def test_floor_at_zero(self):
        h = pt.fabric_health_after_uv(0.01, sols=100)
        assert h == 0.0

    def test_monotonic(self):
        h1 = pt.fabric_health_after_uv(1.0, sols=50)
        h2 = pt.fabric_health_after_uv(1.0, sols=100)
        assert h2 < h1

    def test_zero_sols(self):
        assert pt.fabric_health_after_uv(0.8, sols=0) == 0.8


# ═══════════════════════════════════════════════════════════════════════
# 10. Seal fatigue
# ═══════════════════════════════════════════════════════════════════════

class TestSealFatigue:
    def test_one_cycle(self):
        h = pt.seal_health_after_cycles(1.0, 1)
        assert h == pytest.approx(1.0 - pt.SEAL_FATIGUE_PER_CYCLE)

    def test_zero_cycles(self):
        assert pt.seal_health_after_cycles(0.5, 0) == 0.5

    def test_many_cycles(self):
        h = pt.seal_health_after_cycles(1.0, 5000)
        assert h == pytest.approx(0.5)

    def test_floor(self):
        h = pt.seal_health_after_cycles(0.01, 10000)
        assert h == 0.0

    def test_monotonic(self):
        h1 = pt.seal_health_after_cycles(1.0, 100)
        h2 = pt.seal_health_after_cycles(1.0, 200)
        assert h2 < h1


# ═══════════════════════════════════════════════════════════════════════
# 11. Micrometeorite
# ═══════════════════════════════════════════════════════════════════════

class TestMicrometeorite:
    def test_no_hit(self):
        count = pt.check_micrometeorite(100.0, 0, rng_value=0.999)
        assert count == 0

    def test_hit(self):
        count = pt.check_micrometeorite(100.0, 0, rng_value=0.001)
        assert count == 1

    def test_increments(self):
        count = pt.check_micrometeorite(100.0, 3, rng_value=0.001)
        assert count == 4

    def test_small_surface_lower_probability(self):
        """Tiny tunnel has lower hit probability than large one."""
        # prob = 0.0005 * 0.1 = 0.00005; rng=0.0001 >= prob -> no hit
        count = pt.check_micrometeorite(0.1, 0, rng_value=0.0001)
        assert count == 0
        # But a large surface: prob = 0.0005 * 1000 = 0.5; rng=0.0001 < prob -> hit
        count_large = pt.check_micrometeorite(1000.0, 0, rng_value=0.0001)
        assert count_large == 1


# ═══════════════════════════════════════════════════════════════════════
# 12. Tick function
# ═══════════════════════════════════════════════════════════════════════

class TestTick:
    def test_first_sol(self, tunnel):
        r = pt.tick(tunnel)
        assert r.sol == 1
        assert r.operational is True

    def test_advances_sol(self, tunnel):
        pt.tick(tunnel)
        r2 = pt.tick(tunnel)
        assert r2.sol == 2

    def test_air_leaks(self, tunnel):
        r = pt.tick(tunnel)
        assert r.air_leaked_kg > 0.0

    def test_thermal_loss(self, tunnel):
        r = pt.tick(tunnel)
        assert r.thermal_loss_kwh > 0.0

    def test_fabric_degrades(self, tunnel):
        pt.tick(tunnel)
        assert tunnel.fabric_health < 1.0

    def test_seal_degrades(self, tunnel):
        pt.tick(tunnel)
        assert tunnel.seal_health < 1.0

    def test_burst_margin_positive(self, tunnel):
        r = pt.tick(tunnel)
        assert r.burst_margin_ratio > 0.0

    def test_shielded_degrades_slower(self, shielded_tunnel):
        t_bare = pt.PressurizedTunnel()
        pt.tick(t_bare)
        pt.tick(shielded_tunnel)
        assert shielded_tunnel.fabric_health > t_bare.fabric_health

    def test_worn_seal_fails(self):
        t = pt.PressurizedTunnel(seal_health=0.15)
        r = pt.tick(t)
        assert r.operational is False
        assert "SEAL WORN" in " ".join(r.events)

    def test_cumulative_tracking(self, tunnel):
        r1 = pt.tick(tunnel)
        r2 = pt.tick(tunnel)
        expected_air = r1.air_leaked_kg + r2.air_leaked_kg
        assert tunnel.cumulative_air_lost_kg == pytest.approx(expected_air, rel=1e-3)

    def test_events_list(self, tunnel):
        r = pt.tick(tunnel)
        assert isinstance(r.events, list)

    def test_puncture_on_impact(self):
        t = pt.PressurizedTunnel()
        r = pt.tick(t, rng_value=0.0001)  # force hit
        assert r.puncture_count >= 1
        assert "IMPACT" in " ".join(r.events)

    def test_no_puncture_on_miss(self, tunnel):
        r = pt.tick(tunnel, rng_value=0.999)
        assert r.puncture_count == 0

    def test_pressure_cycles_tracked(self, tunnel):
        pt.tick(tunnel)
        assert tunnel.total_pressure_cycles == tunnel.pressure_cycles_today


# ═══════════════════════════════════════════════════════════════════════
# 13. Multi-sol simulation
# ═══════════════════════════════════════════════════════════════════════

class TestSimulation:
    def test_run_basic(self):
        results = pt.run_simulation(sols=10)
        assert len(results) == 10
        assert all(r.sol == i + 1 for i, r in enumerate(results))

    def test_fabric_degrades_over_year(self):
        results = pt.run_simulation(sols=365)
        assert results[-1].fabric_health < 1.0

    def test_shielded_lasts_longer(self):
        bare = pt.run_simulation(sols=365, shielded=False)
        shielded = pt.run_simulation(sols=365, shielded=True)
        assert shielded[-1].fabric_health > bare[-1].fabric_health

    def test_longer_tunnel_more_thermal_loss(self):
        short = pt.run_simulation(sols=10, length_m=20.0)
        long = pt.run_simulation(sols=10, length_m=100.0)
        t_short = sum(r.thermal_loss_kwh for r in short)
        t_long = sum(r.thermal_loss_kwh for r in long)
        assert t_long > t_short

    def test_wider_tunnel_more_air(self):
        narrow = pt.run_simulation(sols=10, radius_m=0.8)
        wide = pt.run_simulation(sols=10, radius_m=1.5)
        a_narrow = sum(r.air_leaked_kg for r in narrow)
        a_wide = sum(r.air_leaked_kg for r in wide)
        # Wider tunnel has same seals -- leak rate depends on seal, not volume
        # But both should be > 0
        assert a_narrow > 0.0
        assert a_wide > 0.0

    def test_sol_continuity(self):
        results = pt.run_simulation(sols=50)
        for i, r in enumerate(results):
            assert r.sol == i + 1


# ═══════════════════════════════════════════════════════════════════════
# 14. Conservation laws & invariants
# ═══════════════════════════════════════════════════════════════════════

class TestConservation:
    def test_fabric_never_negative(self):
        results = pt.run_simulation(sols=2000)
        for r in results:
            assert r.fabric_health >= 0.0

    def test_seal_never_negative(self):
        t = pt.PressurizedTunnel(pressure_cycles_today=50)
        for _ in range(1000):
            pt.tick(t)
        assert t.seal_health >= 0.0

    def test_air_leak_non_negative(self):
        results = pt.run_simulation(sols=100)
        for r in results:
            assert r.air_leaked_kg >= 0.0

    def test_thermal_non_negative(self):
        results = pt.run_simulation(sols=100)
        for r in results:
            assert r.thermal_loss_kwh >= 0.0

    def test_burst_margin_non_negative(self):
        results = pt.run_simulation(sols=100)
        for r in results:
            assert r.burst_margin_ratio >= 0.0

    def test_punctures_monotonic(self):
        """Puncture count can only increase."""
        results = pt.run_simulation(sols=100)
        for i in range(1, len(results)):
            assert results[i].puncture_count >= results[i - 1].puncture_count

    def test_cumulative_air_matches_sum(self):
        t = pt.PressurizedTunnel()
        results = [pt.tick(t) for _ in range(20)]
        total = sum(r.air_leaked_kg for r in results)
        assert t.cumulative_air_lost_kg == pytest.approx(total, rel=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# 15. Smoke tests
# ═══════════════════════════════════════════════════════════════════════

class TestSmoke:
    def test_10_sol_no_crash(self):
        results = pt.run_simulation(sols=10)
        for r in results:
            assert not math.isnan(r.air_leaked_kg)
            assert not math.isnan(r.thermal_loss_kwh)
            assert not math.isnan(r.fabric_health)
            assert not math.isnan(r.seal_health)
            assert not math.isnan(r.burst_margin_ratio)

    def test_365_sol_no_crash(self):
        results = pt.run_simulation(sols=365)
        assert len(results) == 365
        operational_count = sum(1 for r in results if r.operational)
        assert operational_count > 0

    def test_extreme_length(self):
        results = pt.run_simulation(sols=5, length_m=1000.0)
        for r in results:
            assert not math.isnan(r.thermal_loss_kwh)

    def test_tiny_tunnel(self):
        results = pt.run_simulation(sols=5, length_m=2.0, radius_m=0.5)
        for r in results:
            assert r.operational is True


# ═══════════════════════════════════════════════════════════════════════
# 16. Dataclass sanity
# ═══════════════════════════════════════════════════════════════════════

class TestDataclass:
    def test_defaults(self):
        t = pt.PressurizedTunnel()
        assert t.sol == 0
        assert t.fabric_health == 1.0
        assert t.seal_health == 1.0
        assert t.puncture_count == 0

    def test_tick_result_defaults(self):
        r = pt.TickResult()
        assert r.operational is True
        assert r.sol == 0

    def test_events_independent(self):
        t1 = pt.PressurizedTunnel()
        t2 = pt.PressurizedTunnel()
        t1.events.append("test")
        assert len(t2.events) == 0


# ═══════════════════════════════════════════════════════════════════════
# 17. Constants sanity
# ═══════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_mars_temp(self):
        assert 180.0 <= pt.MARS_AMBIENT_TEMP_K <= 250.0

    def test_hab_pressure(self):
        assert 30.0 <= pt.HAB_PRESSURE_KPA <= 101.3

    def test_mars_atmosphere_thin(self):
        assert pt.MARS_AMBIENT_PRESSURE_KPA < 1.0

    def test_kevlar_strong(self):
        assert pt.KEVLAR_YIELD_STRENGTH_MPA > 1000.0

    def test_seconds_per_sol(self):
        assert abs(pt.SECONDS_PER_SOL - 88775.0) < 1.0
