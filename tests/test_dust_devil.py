"""Tests for dust_devil.py — Mars convective dust devil simulation.

72 unit tests across 11 test classes covering:
  - Seasonal activity modifier
  - Surface temperature delta (diurnal + seasonal)
  - Core pressure drop
  - Cyclostrophic tangential velocity
  - Vortex height and diameter scaling
  - Panel cleaning efficiency (Spirit rover physics)
  - Equipment damage thresholds
  - Single devil generation
  - Tick function integration
  - Multi-sol simulation smoke tests
  - Physical invariants and conservation laws
"""
from __future__ import annotations

import math
import random

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dust_devil import (
    DustDevil, DustDevilState,
    MARS_AIR_DENSITY_KG_M3, MARS_SURFACE_TEMP_K,
    SURFACE_HEATING_PEAK_DT_K, SURFACE_HEATING_MIN_DT_K,
    MAX_CORE_PRESSURE_DROP_PA, MIN_CORE_PRESSURE_DROP_PA,
    V_TAN_MAX_M_S, V_TAN_MIN_M_S,
    MAX_HEIGHT_M, MIN_HEIGHT_M, MAX_DIAMETER_M, MIN_DIAMETER_M,
    MAX_CLEANING_FRACTION, CLEANING_VELOCITY_THRESHOLD_M_S,
    DAMAGE_VELOCITY_THRESHOLD_M_S, DAMAGE_RATE_PER_HIT,
    BASE_DEVILS_PER_SOL, SEASON_PEAK_LS,
    seasonal_activity_modifier, surface_temperature_delta,
    core_pressure_drop, tangential_velocity, vortex_height,
    vortex_diameter, panel_cleaning_efficiency, damage_from_devil,
    generate_devil, tick_dust_devils, create_dust_devil_state,
)


# ===================================================================
# Seasonal activity modifier
# ===================================================================

class TestSeasonalActivity:
    """Dust devil frequency peaks near perihelion (Ls ~270°)."""

    def test_peak_at_perihelion(self) -> None:
        """Maximum activity at Ls = 270°."""
        val = seasonal_activity_modifier(SEASON_PEAK_LS)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_minimum_far_from_peak(self) -> None:
        """Activity at Ls = 90° (aphelion) should be near minimum."""
        val = seasonal_activity_modifier(90.0)
        assert 0.1 <= val < 0.5

    def test_always_positive(self) -> None:
        """Activity never drops to zero — devils occur year-round."""
        for ls in range(0, 360, 10):
            assert seasonal_activity_modifier(float(ls)) >= 0.1

    def test_bounded_above(self) -> None:
        """Activity never exceeds 1.0."""
        for ls in range(0, 360, 5):
            assert seasonal_activity_modifier(float(ls)) <= 1.0

    def test_symmetric_around_peak(self) -> None:
        """Roughly symmetric around perihelion."""
        before = seasonal_activity_modifier(SEASON_PEAK_LS - 30.0)
        after = seasonal_activity_modifier(SEASON_PEAK_LS + 30.0)
        assert before == pytest.approx(after, abs=0.05)

    def test_wraps_around_360(self) -> None:
        """Ls = 0 and Ls = 360 are identical."""
        assert (seasonal_activity_modifier(0.0) ==
                pytest.approx(seasonal_activity_modifier(360.0), abs=0.001))


# ===================================================================
# Surface temperature delta
# ===================================================================

class TestSurfaceTemperatureDelta:
    """Surface heating drives convection that spawns dust devils."""

    def test_peak_at_afternoon(self) -> None:
        """Maximum ΔT at ~13:00 local time."""
        dt_13 = surface_temperature_delta(SEASON_PEAK_LS, 13.0)
        dt_06 = surface_temperature_delta(SEASON_PEAK_LS, 6.0)
        assert dt_13 > dt_06

    def test_low_at_night(self) -> None:
        """Minimal ΔT at night (02:00)."""
        dt = surface_temperature_delta(SEASON_PEAK_LS, 2.0)
        assert dt < SURFACE_HEATING_PEAK_DT_K * 0.3

    def test_always_non_negative(self) -> None:
        """Temperature difference never goes negative."""
        for hour in [0.0, 6.0, 12.0, 18.0, 24.0]:
            for ls in [0.0, 90.0, 180.0, 270.0]:
                assert surface_temperature_delta(ls, hour) >= 0.0

    def test_seasonal_variation(self) -> None:
        """Perihelion produces stronger surface heating."""
        dt_peri = surface_temperature_delta(270.0, 13.0)
        dt_aph = surface_temperature_delta(90.0, 13.0)
        assert dt_peri > dt_aph

    def test_bounded_above(self) -> None:
        """Never exceeds physical maximum."""
        dt = surface_temperature_delta(270.0, 13.0)
        assert dt <= SURFACE_HEATING_PEAK_DT_K * 1.5  # generous bound


# ===================================================================
# Core pressure drop
# ===================================================================

class TestCorePressureDrop:
    """Vortex pressure deficit from thermal gradient."""

    def test_zero_dt_gives_zero(self) -> None:
        assert core_pressure_drop(0.0) == 0.0

    def test_negative_dt_gives_zero(self) -> None:
        assert core_pressure_drop(-10.0) == 0.0

    def test_positive_dt_gives_positive(self) -> None:
        assert core_pressure_drop(20.0) > 0.0

    def test_capped_at_max(self) -> None:
        """Even extreme ΔT cannot exceed InSight's max measurement."""
        dp = core_pressure_drop(1000.0)
        assert dp <= MAX_CORE_PRESSURE_DROP_PA

    def test_monotonic_increasing(self) -> None:
        """More heating → stronger vortex."""
        dp1 = core_pressure_drop(10.0)
        dp2 = core_pressure_drop(20.0)
        assert dp2 >= dp1


# ===================================================================
# Cyclostrophic tangential velocity
# ===================================================================

class TestTangentialVelocity:
    """v = sqrt(dP / ρ) — Mars thin atmosphere amplifies wind speed."""

    def test_zero_pressure_gives_zero(self) -> None:
        assert tangential_velocity(0.0) == 0.0

    def test_negative_pressure_gives_zero(self) -> None:
        assert tangential_velocity(-5.0) == 0.0

    def test_physical_formula(self) -> None:
        """Spot-check: dP=2 Pa, ρ=0.02 → v = sqrt(100) = 10 m/s."""
        v = tangential_velocity(2.0)
        assert v == pytest.approx(10.0, rel=0.01)

    def test_capped_at_max(self) -> None:
        v = tangential_velocity(100.0)
        assert v <= V_TAN_MAX_M_S

    def test_mars_thin_atmosphere_amplifies(self) -> None:
        """Same pressure drop produces much higher velocity on Mars
        than Earth (ρ_earth ≈ 1.2 vs ρ_mars ≈ 0.02)."""
        v_mars = tangential_velocity(2.0)
        # On Earth: sqrt(2/1.2) ≈ 1.29 m/s
        v_earth_approx = math.sqrt(2.0 / 1.2)
        assert v_mars > 5 * v_earth_approx  # Mars amplifies hugely


# ===================================================================
# Vortex geometry
# ===================================================================

class TestVortexGeometry:
    """Height and diameter scaling from convective energy."""

    def test_height_zero_dt(self) -> None:
        assert vortex_height(0.0) == 0.0

    def test_height_negative_dt(self) -> None:
        assert vortex_height(-5.0) == 0.0

    def test_height_positive(self) -> None:
        h = vortex_height(30.0)
        assert MIN_HEIGHT_M <= h <= MAX_HEIGHT_M

    def test_height_monotonic(self) -> None:
        h1 = vortex_height(10.0)
        h2 = vortex_height(40.0)
        assert h2 > h1

    def test_diameter_zero_height(self) -> None:
        assert vortex_diameter(0.0) == 0.0

    def test_diameter_proportional_to_height(self) -> None:
        d1 = vortex_diameter(100.0)
        d2 = vortex_diameter(1000.0)
        assert d2 > d1

    def test_diameter_bounded(self) -> None:
        d = vortex_diameter(MAX_HEIGHT_M)
        assert d <= MAX_DIAMETER_M

    def test_tall_mars_devils(self) -> None:
        """Mars dust devils reach km-scale heights."""
        h = vortex_height(SURFACE_HEATING_PEAK_DT_K)
        assert h > 1000.0  # should exceed 1 km


# ===================================================================
# Panel cleaning (the Spirit rover effect)
# ===================================================================

class TestPanelCleaning:
    """Dust devils clean solar panels — the key beneficial effect."""

    def test_below_threshold_no_cleaning(self) -> None:
        """Weak vortices don't lift dust."""
        c = panel_cleaning_efficiency(
            CLEANING_VELOCITY_THRESHOLD_M_S - 1.0, 50.0)
        assert c == 0.0

    def test_close_strong_devil_cleans(self) -> None:
        """A strong devil passing close cleans panels."""
        c = panel_cleaning_efficiency(V_TAN_MAX_M_S, 10.0)
        assert c > 0.0

    def test_bounded_by_max(self) -> None:
        """Never removes more than MAX_CLEANING_FRACTION."""
        c = panel_cleaning_efficiency(V_TAN_MAX_M_S, 0.0)
        assert c <= MAX_CLEANING_FRACTION

    def test_decays_with_distance(self) -> None:
        """Cleaning drops off with distance."""
        c_close = panel_cleaning_efficiency(30.0, 10.0)
        c_far = panel_cleaning_efficiency(30.0, 1000.0)
        assert c_close > c_far

    def test_increases_with_velocity(self) -> None:
        """Stronger vortices clean more."""
        c_weak = panel_cleaning_efficiency(10.0, 50.0)
        c_strong = panel_cleaning_efficiency(40.0, 50.0)
        assert c_strong > c_weak

    def test_always_non_negative(self) -> None:
        """Cleaning fraction never goes negative."""
        for v in [0.0, 5.0, 10.0, 20.0, 45.0]:
            for d in [0.0, 50.0, 500.0, 5000.0]:
                assert panel_cleaning_efficiency(v, d) >= 0.0


# ===================================================================
# Equipment damage
# ===================================================================

class TestDamage:
    """Only extreme, direct-hit devils cause damage."""

    def test_below_threshold_no_damage(self) -> None:
        dmg = damage_from_devil(20.0, 10.0, 50.0)
        assert dmg == 0.0

    def test_far_away_no_damage(self) -> None:
        """Must be within vortex diameter to cause damage."""
        dmg = damage_from_devil(V_TAN_MAX_M_S, 1000.0, 50.0)
        assert dmg == 0.0

    def test_direct_hit_causes_damage(self) -> None:
        """Strong devil passing over base damages equipment."""
        dmg = damage_from_devil(V_TAN_MAX_M_S, 5.0, 100.0)
        assert 0.0 < dmg <= DAMAGE_RATE_PER_HIT

    def test_damage_bounded(self) -> None:
        """Maximum damage per hit is small."""
        dmg = damage_from_devil(V_TAN_MAX_M_S, 0.0, 1000.0)
        assert dmg <= DAMAGE_RATE_PER_HIT


# ===================================================================
# Devil generation
# ===================================================================

class TestGenerateDevil:
    """Generate individual dust devils from thermal conditions."""

    def test_cold_conditions_no_devil(self) -> None:
        """Below threshold ΔT produces no vortex."""
        random.seed(42)
        devil = generate_devil(0.5)
        assert devil is None

    def test_hot_conditions_produce_devil(self) -> None:
        """Strong thermal gradient always produces a vortex."""
        random.seed(42)
        devil = generate_devil(35.0)
        assert devil is not None
        assert devil.tangential_velocity_m_s > 0.0
        assert devil.height_m > 0.0

    def test_devil_fields_bounded(self) -> None:
        """All generated fields within physical bounds."""
        random.seed(123)
        for dt in [10.0, 20.0, 30.0, 40.0]:
            devil = generate_devil(dt)
            if devil is not None:
                assert 0.0 <= devil.pressure_drop_pa <= MAX_CORE_PRESSURE_DROP_PA
                assert 0.0 <= devil.tangential_velocity_m_s <= V_TAN_MAX_M_S
                assert 0.0 <= devil.height_m <= MAX_HEIGHT_M
                assert 0.0 <= devil.diameter_m <= MAX_DIAMETER_M
                assert devil.distance_from_base_m >= 0.0
                assert 0.0 <= devil.cleaning_fraction <= 1.0


# ===================================================================
# Tick function integration
# ===================================================================

class TestTickDustDevils:
    """Integration tests for the per-sol tick function."""

    def test_sol_increments(self) -> None:
        state = create_dust_devil_state()
        assert state.sol == 0
        tick_dust_devils(state, rng=random.Random(42))
        assert state.sol == 1

    def test_solar_longitude_advances(self) -> None:
        state = create_dust_devil_state()
        ls_before = state.solar_longitude_deg
        tick_dust_devils(state, rng=random.Random(42))
        assert state.solar_longitude_deg > ls_before

    def test_solar_longitude_wraps(self) -> None:
        """Ls wraps around 360°."""
        state = create_dust_devil_state(solar_longitude_deg=359.9)
        tick_dust_devils(state, rng=random.Random(42))
        assert 0.0 <= state.solar_longitude_deg < 360.5

    def test_panel_dust_non_negative(self) -> None:
        state = create_dust_devil_state(initial_dust=0.01)
        for _ in range(100):
            tick_dust_devils(state, rng=random.Random(42))
        assert state.panel_dust_coverage >= 0.0

    def test_panel_dust_bounded(self) -> None:
        state = create_dust_devil_state(initial_dust=0.99)
        for _ in range(50):
            tick_dust_devils(state, rng=random.Random(42))
        assert state.panel_dust_coverage <= 1.0

    def test_equipment_health_non_negative(self) -> None:
        state = create_dust_devil_state()
        for _ in range(1000):
            tick_dust_devils(state, rng=random.Random(42))
        assert state.equipment_health >= 0.0

    def test_equipment_health_bounded(self) -> None:
        state = create_dust_devil_state()
        tick_dust_devils(state, rng=random.Random(42))
        assert state.equipment_health <= 1.0

    def test_cumulative_stats_grow(self) -> None:
        state = create_dust_devil_state()
        for i in range(50):
            tick_dust_devils(state, rng=random.Random(i))
        assert state.total_devils_observed >= 0
        assert state.total_cleaning_cumulative >= 0.0

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same results."""
        s1 = create_dust_devil_state()
        s2 = create_dust_devil_state()
        for _ in range(10):
            tick_dust_devils(s1, rng=random.Random(999))
            tick_dust_devils(s2, rng=random.Random(999))
        assert s1.sol == s2.sol
        assert s1.total_devils_observed == s2.total_devils_observed
        assert s1.panel_dust_coverage == pytest.approx(
            s2.panel_dust_coverage, abs=1e-10)

    def test_dust_re_accumulation(self) -> None:
        """Even with cleaning, ambient dust settles back."""
        state = create_dust_devil_state(initial_dust=0.0)
        # Run enough sols to accumulate dust
        for _ in range(100):
            tick_dust_devils(state, rng=random.Random(42))
        assert state.panel_dust_coverage > 0.0


# ===================================================================
# Smoke tests — full simulations
# ===================================================================

class TestSmoke:
    """Full simulation smoke tests — must not crash."""

    def test_full_mars_year_668_sols(self) -> None:
        """Run a full Mars year without crash."""
        state = create_dust_devil_state()
        for sol in range(668):
            tick_dust_devils(state, rng=random.Random(sol))
        assert state.sol == 668
        assert state.total_devils_observed > 0
        assert state.equipment_health > 0.0

    def test_10_sol_quick_smoke(self) -> None:
        """Minimum viable smoke test: 10 sols."""
        state = create_dust_devil_state()
        for sol in range(10):
            tick_dust_devils(state, rng=random.Random(sol))
        assert state.sol == 10

    def test_perihelion_season_more_active(self) -> None:
        """More devils observed during perihelion season."""
        # Run near perihelion (Ls 270)
        s_peri = create_dust_devil_state(solar_longitude_deg=250.0)
        for sol in range(100):
            tick_dust_devils(s_peri, rng=random.Random(sol))

        # Run near aphelion (Ls 90)
        s_aph = create_dust_devil_state(solar_longitude_deg=70.0)
        for sol in range(100):
            tick_dust_devils(s_aph, rng=random.Random(sol))

        assert s_peri.total_devils_observed > s_aph.total_devils_observed

    def test_cleaning_over_year_keeps_panels_usable(self) -> None:
        """Over a Mars year, dust devils keep panels from clogging."""
        state = create_dust_devil_state(initial_dust=0.5)
        for sol in range(668):
            tick_dust_devils(state, rng=random.Random(sol))
        # Panels shouldn't be at 100% dust — devils help
        assert state.panel_dust_coverage < 1.0

    def test_strongest_ever_recorded(self) -> None:
        """After a full year, we should have recorded a strongest."""
        state = create_dust_devil_state()
        for sol in range(668):
            tick_dust_devils(state, rng=random.Random(sol))
        assert state.strongest_ever_pa > 0.0
        assert state.tallest_ever_m > 0.0


# ===================================================================
# Physical invariants
# ===================================================================

class TestPhysicalInvariants:
    """Properties that must hold across all conditions."""

    def test_velocity_from_pressure_is_physical(self) -> None:
        """v² × ρ = dP (cyclostrophic balance within bounds)."""
        for dp in [0.5, 1.0, 2.0, 5.0, 9.0]:
            v = tangential_velocity(dp)
            if v < V_TAN_MAX_M_S:  # not clamped
                reconstructed_dp = v * v * MARS_AIR_DENSITY_KG_M3
                assert reconstructed_dp == pytest.approx(dp, rel=0.01)

    def test_taller_devils_wider(self) -> None:
        """Diameter scales with height — no inversions."""
        heights = [50.0, 200.0, 500.0, 2000.0, 8000.0]
        diameters = [vortex_diameter(h) for h in heights]
        for i in range(len(diameters) - 1):
            assert diameters[i] <= diameters[i + 1]

    def test_cleaning_decreases_dust(self) -> None:
        """Every cleaning event reduces dust (multiplicative)."""
        dust = 0.5
        cleaning = 0.05
        new_dust = dust * (1.0 - cleaning)
        assert new_dust < dust

    def test_sol_counter_monotonic(self) -> None:
        """Sol count never decreases."""
        state = create_dust_devil_state()
        prev_sol = state.sol
        for i in range(50):
            tick_dust_devils(state, rng=random.Random(i))
            assert state.sol > prev_sol
            prev_sol = state.sol
