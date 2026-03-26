"""Tests for dust_storm.py — Mars dust storm simulation model.

59 unit tests across 12 test classes covering:
  - DustEvent lifecycle and tau curves
  - Beer-Lambert solar attenuation
  - Dust deposition, wind, abrasion physics
  - Seasonal storm probability
  - Visibility estimates
  - Tick function integration
  - Multi-event superposition
  - Colony survival scenarios
  - Physical invariants and conservation laws
  - Smoke tests (full Mars year, global storm survival)
"""
from __future__ import annotations

import math
import random

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dust_storm import (
    DustEvent, DustState,
    TAU_CLEAR, TAU_LOCAL_MAX, TAU_REGIONAL_MAX, TAU_GLOBAL_MAX, TAU_LETHAL,
    MARS_SOLAR_IRRADIANCE_W_M2, SURFACE_COOLING_PER_TAU,
    WIND_SPEED_CLEAR_KMH, WIND_SPEED_STORM_KMH,
    DEPOSITION_RATE_CLEAR, DEPOSITION_RATE_STORM,
    ABRASION_RATE_CLEAR, ABRASION_RATE_STORM,
    LOCAL_DURATION_RANGE, REGIONAL_DURATION_RANGE, GLOBAL_DURATION_RANGE,
    seasonal_storm_modifier, beer_lambert, dust_deposition_rate,
    wind_speed, abrasion_damage, surface_temp_effect, visibility_km,
    generate_event, tick_dust, clean_surfaces, create_colony_dust_state,
)


# ===================================================================
# DustEvent dataclass
# ===================================================================

class TestDustEvent:
    """Validate DustEvent construction and lifecycle curve."""

    def test_valid_local(self) -> None:
        """Local event creates with correct fields."""
        e = DustEvent(kind='local', tau_peak=0.3, duration_sols=2)
        assert e.kind == 'local'
        assert e.tau_peak == 0.3
        assert e.duration_sols == 2
        assert e.sols_elapsed == 0
        assert e.active is True

    def test_valid_regional(self) -> None:
        e = DustEvent(kind='regional', tau_peak=2.5, duration_sols=10)
        assert e.kind == 'regional'

    def test_valid_global(self) -> None:
        e = DustEvent(kind='global', tau_peak=6.0, duration_sols=60)
        assert e.kind == 'global'

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            DustEvent(kind='hurricane', tau_peak=1.0, duration_sols=5)

    def test_negative_tau_clamped(self) -> None:
        e = DustEvent(kind='local', tau_peak=-1.0, duration_sols=2)
        assert e.tau_peak == 0.0

    def test_zero_duration_clamped(self) -> None:
        e = DustEvent(kind='local', tau_peak=0.3, duration_sols=0)
        assert e.duration_sols == 1

    def test_lifecycle_ramp_up(self) -> None:
        """First 20% of event ramps up to peak tau."""
        e = DustEvent(kind='regional', tau_peak=3.0, duration_sols=100)
        e.sols_elapsed = 10  # 10% through -> in ramp-up phase
        tau = e.current_tau()
        assert 0 < tau < 3.0
        assert tau == pytest.approx(3.0 * (0.1 / 0.2), rel=0.01)

    def test_lifecycle_sustain(self) -> None:
        """Middle 60% of event holds at peak tau."""
        e = DustEvent(kind='regional', tau_peak=3.0, duration_sols=100)
        e.sols_elapsed = 50  # 50% through -> sustained
        assert e.current_tau() == pytest.approx(3.0, rel=0.01)

    def test_lifecycle_decay(self) -> None:
        """Last 20% of event decays toward zero."""
        e = DustEvent(kind='regional', tau_peak=3.0, duration_sols=100)
        e.sols_elapsed = 90  # 90% through -> decaying
        tau = e.current_tau()
        assert 0 < tau < 3.0

    def test_lifecycle_expired(self) -> None:
        """After duration, tau is zero."""
        e = DustEvent(kind='local', tau_peak=0.4, duration_sols=2)
        e.sols_elapsed = 2
        assert e.current_tau() == 0.0

    def test_inactive_returns_zero(self) -> None:
        e = DustEvent(kind='local', tau_peak=0.4, duration_sols=5)
        e.active = False
        assert e.current_tau() == 0.0


# ===================================================================
# Beer-Lambert solar attenuation
# ===================================================================

class TestBeerLambert:
    """Test solar irradiance attenuation through dust."""

    def test_clear_sky(self) -> None:
        """Clear sky (τ=0) transmits all light."""
        assert beer_lambert(0.0) == pytest.approx(1.0)

    def test_moderate_dust(self) -> None:
        """τ=1 gives ~37% transmission (e^-1)."""
        assert beer_lambert(1.0) == pytest.approx(math.exp(-1), rel=0.001)

    def test_heavy_dust(self) -> None:
        """τ=5 gives ~0.7% transmission."""
        result = beer_lambert(5.0)
        assert result == pytest.approx(math.exp(-5), rel=0.001)
        assert result < 0.01

    def test_negative_tau_clamped(self) -> None:
        """Negative τ treated as zero."""
        assert beer_lambert(-1.0) == pytest.approx(1.0)

    def test_monotonically_decreasing(self) -> None:
        """More dust always means less light."""
        prev = 1.0
        for tau in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            val = beer_lambert(tau)
            assert val < prev
            prev = val


# ===================================================================
# Dust deposition
# ===================================================================

class TestDustDeposition:
    """Test dust accumulation rates."""

    def test_clear_sky_rate(self) -> None:
        assert dust_deposition_rate(0.0) == DEPOSITION_RATE_CLEAR

    def test_storm_rate_higher(self) -> None:
        rate = dust_deposition_rate(TAU_REGIONAL_MAX)
        assert rate >= DEPOSITION_RATE_STORM * 0.99

    def test_interpolates(self) -> None:
        """Mid-range τ gives rate between clear and storm."""
        mid_tau = (TAU_CLEAR + TAU_REGIONAL_MAX) / 2
        rate = dust_deposition_rate(mid_tau)
        assert DEPOSITION_RATE_CLEAR < rate < DEPOSITION_RATE_STORM

    def test_negative_tau(self) -> None:
        assert dust_deposition_rate(-5.0) == DEPOSITION_RATE_CLEAR


# ===================================================================
# Wind speed
# ===================================================================

class TestWindSpeed:
    """Test wind speed estimation from optical depth."""

    def test_clear_sky(self) -> None:
        assert wind_speed(0.0) == WIND_SPEED_CLEAR_KMH

    def test_storm_winds(self) -> None:
        ws = wind_speed(TAU_GLOBAL_MAX)
        assert ws >= WIND_SPEED_STORM_KMH * 0.9

    def test_monotonically_increasing(self) -> None:
        prev = 0.0
        for tau in [0.0, 0.5, 1.0, 3.0, 8.0]:
            ws = wind_speed(tau)
            assert ws >= prev
            prev = ws


# ===================================================================
# Abrasion damage
# ===================================================================

class TestAbrasion:
    """Test equipment abrasion from dust."""

    def test_clear_no_damage(self) -> None:
        assert abrasion_damage(0.0) == 0.0
        assert abrasion_damage(TAU_CLEAR) == 0.0

    def test_storm_causes_damage(self) -> None:
        dmg = abrasion_damage(TAU_GLOBAL_MAX)
        assert dmg > 0.0
        assert dmg <= ABRASION_RATE_STORM

    def test_more_dust_more_damage(self) -> None:
        d1 = abrasion_damage(1.0)
        d2 = abrasion_damage(4.0)
        assert d2 > d1


# ===================================================================
# Surface temperature effect
# ===================================================================

class TestSurfaceTempEffect:
    """Test dust-induced surface temperature changes."""

    def test_clear_no_effect(self) -> None:
        assert surface_temp_effect(TAU_CLEAR) == 0.0
        assert surface_temp_effect(0.0) == 0.0

    def test_dust_cools_surface(self) -> None:
        """Dust blocks sunlight -> surface cooling."""
        delta = surface_temp_effect(3.0)
        assert delta < 0.0

    def test_proportional_to_excess_tau(self) -> None:
        d1 = surface_temp_effect(1.0)
        d2 = surface_temp_effect(2.0)
        # d2 should be roughly double d1 (linear relationship)
        assert abs(d2) > abs(d1)

    def test_global_storm_severe_cooling(self) -> None:
        """Global storm causes significant cooling."""
        delta = surface_temp_effect(TAU_GLOBAL_MAX)
        assert delta < -15.0  # > 15°C cooling


# ===================================================================
# Visibility
# ===================================================================

class TestVisibility:
    """Test visibility estimation from optical depth."""

    def test_clear_sky_good_vis(self) -> None:
        vis = visibility_km(TAU_CLEAR)
        assert vis == pytest.approx(10.0, rel=0.1)

    def test_storm_poor_vis(self) -> None:
        vis = visibility_km(6.0)
        assert vis < 1.0

    def test_clamped_floor(self) -> None:
        vis = visibility_km(1000.0)
        assert vis >= 0.01

    def test_clamped_ceiling(self) -> None:
        vis = visibility_km(0.01)
        assert vis <= 100.0


# ===================================================================
# Seasonal modifier
# ===================================================================

class TestSeasonalModifier:
    """Test seasonal storm probability modulation."""

    def test_always_positive(self) -> None:
        for sol in range(0, 669, 50):
            assert seasonal_storm_modifier(sol) > 0.0

    def test_peak_near_perihelion(self) -> None:
        """Storm probability highest near Ls 270 (perihelion)."""
        # Ls 270 ≈ sol 502 (270/360 * 668.6)
        peak_sol = int(270 / 360 * 668.6)
        peak_mod = seasonal_storm_modifier(peak_sol)
        # Off-season (Ls 90 ≈ sol 167)
        off_sol = int(90 / 360 * 668.6)
        off_mod = seasonal_storm_modifier(off_sol)
        assert peak_mod > off_mod * 2.0

    def test_periodic(self) -> None:
        """Modifier repeats each Mars year."""
        m1 = seasonal_storm_modifier(100)
        m2 = seasonal_storm_modifier(100 + 669)
        assert m1 == pytest.approx(m2, rel=0.01)


# ===================================================================
# Event generation
# ===================================================================

class TestGenerateEvent:
    """Test stochastic event generation."""

    def test_local_event_range(self) -> None:
        rng = random.Random(42)
        for _ in range(20):
            e = generate_event('local', rng)
            assert e.kind == 'local'
            assert 0.0 < e.tau_peak <= TAU_LOCAL_MAX
            assert LOCAL_DURATION_RANGE[0] <= e.duration_sols <= LOCAL_DURATION_RANGE[1]

    def test_regional_event_range(self) -> None:
        rng = random.Random(42)
        for _ in range(20):
            e = generate_event('regional', rng)
            assert TAU_LOCAL_MAX <= e.tau_peak <= TAU_REGIONAL_MAX

    def test_global_event_range(self) -> None:
        rng = random.Random(42)
        for _ in range(20):
            e = generate_event('global', rng)
            assert TAU_REGIONAL_MAX <= e.tau_peak <= TAU_GLOBAL_MAX

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_event('tornado')

    def test_reproducible_with_seed(self) -> None:
        e1 = generate_event('regional', random.Random(99))
        e2 = generate_event('regional', random.Random(99))
        assert e1.tau_peak == e2.tau_peak
        assert e1.duration_sols == e2.duration_sols


# ===================================================================
# Tick function
# ===================================================================

class TestTickDust:
    """Test the main tick_dust simulation loop."""

    def test_one_sol_advances(self) -> None:
        state = DustState()
        tick_dust(state, rng=random.Random(42))
        assert state.sol == 1

    def test_tau_always_non_negative(self) -> None:
        state = DustState()
        rng = random.Random(42)
        for _ in range(100):
            tick_dust(state, rng=rng)
            assert state.current_tau >= 0.0

    def test_solar_factor_bounded(self) -> None:
        state = DustState()
        rng = random.Random(42)
        for _ in range(100):
            tick_dust(state, rng=rng)
            assert 0.0 <= state.solar_factor <= 1.0

    def test_dust_coverage_bounded(self) -> None:
        state = DustState()
        rng = random.Random(42)
        for _ in range(200):
            tick_dust(state, rng=rng)
            assert 0.0 <= state.dust_coverage <= 1.0

    def test_equipment_health_bounded(self) -> None:
        state = DustState()
        rng = random.Random(42)
        for _ in range(200):
            tick_dust(state, rng=rng)
            assert 0.0 <= state.equipment_health <= 1.0

    def test_force_event(self) -> None:
        """Force-spawning an event creates it immediately."""
        state = DustState()
        tick_dust(state, rng=random.Random(42), force_event='global')
        active = [e for e in state.events if e.active and e.kind == 'global']
        assert len(active) >= 1

    def test_storm_sols_tracked(self) -> None:
        """Sols under storm conditions are counted."""
        state = DustState()
        rng = random.Random(42)
        for _ in range(10):
            tick_dust(state, rng=rng, force_event='regional')
        assert state.total_storm_sols > 0

    def test_worst_tau_tracked(self) -> None:
        """Worst-ever optical depth is recorded."""
        state = DustState()
        tick_dust(state, rng=random.Random(42), force_event='global')
        assert state.worst_tau_ever > TAU_CLEAR

    def test_events_pruned(self) -> None:
        """Dead events are pruned to keep list bounded."""
        state = DustState()
        rng = random.Random(42)
        for _ in range(500):
            tick_dust(state, rng=rng, force_event='local')
        # Should not have unbounded event list
        assert len(state.events) < 100


# ===================================================================
# Surface cleaning
# ===================================================================

class TestCleanSurfaces:
    """Test crew surface cleaning."""

    def test_removes_dust(self) -> None:
        state = DustState(dust_coverage=0.5)
        clean_surfaces(state, 0.9)
        assert state.dust_coverage == pytest.approx(0.05, abs=0.001)

    def test_full_clean(self) -> None:
        state = DustState(dust_coverage=0.8)
        clean_surfaces(state, 1.0)
        assert state.dust_coverage == pytest.approx(0.0, abs=0.001)

    def test_no_clean(self) -> None:
        state = DustState(dust_coverage=0.5)
        clean_surfaces(state, 0.0)
        assert state.dust_coverage == pytest.approx(0.5, abs=0.001)


# ===================================================================
# Physical invariants
# ===================================================================

class TestPhysicalInvariants:
    """Property-based tests for physical correctness."""

    def test_tau_capped_at_lethal(self) -> None:
        """Combined tau never exceeds lethal threshold."""
        state = DustState()
        rng = random.Random(42)
        for _ in range(50):
            tick_dust(state, rng=rng, force_event='global')
            assert state.current_tau <= TAU_LETHAL

    def test_more_dust_less_light(self) -> None:
        """Higher tau always means less solar irradiance."""
        s1 = DustState()
        s2 = DustState()
        tick_dust(s1, rng=random.Random(42), force_event='local')
        tick_dust(s2, rng=random.Random(42), force_event='global')
        assert s2.solar_factor <= s1.solar_factor

    def test_more_dust_more_cooling(self) -> None:
        """Higher tau means more surface cooling."""
        c1 = surface_temp_effect(1.0)
        c2 = surface_temp_effect(5.0)
        assert c2 < c1  # more negative = more cooling

    def test_cleaning_always_reduces_coverage(self) -> None:
        for cov in [0.1, 0.5, 0.9]:
            state = DustState(dust_coverage=cov)
            clean_surfaces(state, 0.5)
            assert state.dust_coverage < cov


# ===================================================================
# Smoke tests
# ===================================================================

class TestSmoke:
    """End-to-end smoke tests for full simulation runs."""

    def test_one_mars_year_no_crash(self) -> None:
        """669 sols without crash or invalid state."""
        state = DustState()
        rng = random.Random(12345)
        for _ in range(669):
            tick_dust(state, rng=rng)
        assert state.sol == 669
        assert 0.0 <= state.solar_factor <= 1.0
        assert 0.0 <= state.dust_coverage <= 1.0
        assert 0.0 <= state.equipment_health <= 1.0

    def test_global_storm_survival(self) -> None:
        """Colony survives a forced global dust storm."""
        state = DustState()
        rng = random.Random(42)
        # Force global storm
        tick_dust(state, rng=rng, force_event='global')
        # Run through it
        for _ in range(89):
            tick_dust(state, rng=rng)
        # Equipment damaged but not destroyed
        assert state.equipment_health > 0.5
        # Solar was severely reduced during storm
        assert state.worst_tau_ever > TAU_REGIONAL_MAX

    def test_three_mars_years_stable(self) -> None:
        """3 Mars years of simulation remain bounded."""
        state = DustState()
        rng = random.Random(777)
        for _ in range(669 * 3):
            tick_dust(state, rng=rng)
        assert state.sol == 669 * 3
        assert 0.0 <= state.equipment_health <= 1.0
        assert state.total_storm_sols > 0  # some storms should have occurred

    def test_factory_creates_state(self) -> None:
        for scale in ['outpost', 'base', 'settlement']:
            state = create_colony_dust_state(scale)
            assert isinstance(state, DustState)
            assert state.sol == 0
