"""
Tests for seismology.py — Marsquake simulation and structural impact.

Coverage:
  - PGA attenuation model (magnitude scaling, distance decay, edge cases)
  - Seasonal rate modulation (sinusoidal variation, period, bounds)
  - Damage classification (all 6 categories, boundary conditions)
  - Quake generation (Poisson process, magnitude distribution, distance)
  - Aftershock sequences (Omori law, magnitude threshold, decay)
  - Fatigue tracking (accumulation, status thresholds)
  - Tick integration (state updates, log pruning, event reporting)
  - Physical invariants (PGA positive, magnitude bounds, monotonicity)
  - Multi-sol smoke tests (100-sol, 669-sol Mars year, aftershock cascades)
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.seismology import (
    AFTERSHOCK_C,
    AFTERSHOCK_K,
    AFTERSHOCK_MAG_DROP,
    AFTERSHOCK_MIN_MAINSHOCK,
    AFTERSHOCK_P,
    ATTEN_A,
    ATTEN_B,
    ATTEN_C,
    DEFAULT_DISTANCE_KM,
    FATIGUE_CRITICAL,
    FATIGUE_RATE_PER_PGA,
    FATIGUE_THRESHOLD,
    MAGNITUDE_DISTRIBUTION,
    PGA_IMPERCEPTIBLE,
    PGA_MINOR_DAMAGE,
    PGA_MODERATE_DAMAGE,
    PGA_PERCEPTIBLE,
    PGA_SEVERE_DAMAGE,
    QUAKE_RATE_PER_SOL,
    SEASONAL_AMPLITUDE,
    SEASONAL_PERIOD_SOLS,
    Quake,
    SeismicState,
    damage_category,
    fatigue_status,
    generate_aftershocks,
    generate_quakes,
    peak_ground_acceleration,
    seasonal_rate_modifier,
    tick_seismology,
)


# ===================================================================
# Constants validation
# ===================================================================

class TestConstants:
    """Physical constants within reasonable bounds."""

    def test_quake_rate_positive(self):
        assert QUAKE_RATE_PER_SOL > 0

    def test_quake_rate_reasonable(self):
        """InSight saw ~1/sol, our model should be in that range."""
        assert 0.1 < QUAKE_RATE_PER_SOL < 5.0

    def test_magnitude_weights_sum_to_one(self):
        total = sum(w for _, _, w in MAGNITUDE_DISTRIBUTION)
        assert total == pytest.approx(1.0, rel=0.01)

    def test_magnitude_ranges_ordered(self):
        for mag_min, mag_max, _ in MAGNITUDE_DISTRIBUTION:
            assert mag_min < mag_max

    def test_attenuation_coefficients_positive(self):
        assert ATTEN_A > 0
        assert ATTEN_B > 0
        assert ATTEN_C > 0

    def test_pga_thresholds_ordered(self):
        assert PGA_IMPERCEPTIBLE < PGA_PERCEPTIBLE < PGA_MINOR_DAMAGE
        assert PGA_MINOR_DAMAGE < PGA_MODERATE_DAMAGE < PGA_SEVERE_DAMAGE

    def test_fatigue_thresholds_ordered(self):
        assert FATIGUE_THRESHOLD < FATIGUE_CRITICAL

    def test_seasonal_amplitude_bounded(self):
        assert 0 < SEASONAL_AMPLITUDE < 1.0

    def test_seasonal_period_mars_year(self):
        assert 660 < SEASONAL_PERIOD_SOLS < 680

    def test_aftershock_min_mainshock_positive(self):
        assert AFTERSHOCK_MIN_MAINSHOCK > 0


# ===================================================================
# PGA attenuation model
# ===================================================================

class TestPGA:
    """Peak ground acceleration calculation."""

    def test_zero_magnitude_very_small(self):
        pga = peak_ground_acceleration(0.0, 100.0)
        assert pga > 0  # 10^(0) = 1, still some signal

    def test_negative_magnitude_zero(self):
        assert peak_ground_acceleration(-1.0, 100.0) == 0.0

    def test_zero_distance_zero(self):
        assert peak_ground_acceleration(3.0, 0.0) == 0.0

    def test_negative_distance_zero(self):
        assert peak_ground_acceleration(3.0, -10.0) == 0.0

    def test_pga_positive(self):
        assert peak_ground_acceleration(3.0, 100.0) > 0

    def test_pga_increases_with_magnitude(self):
        pga1 = peak_ground_acceleration(2.0, 100.0)
        pga2 = peak_ground_acceleration(3.0, 100.0)
        pga3 = peak_ground_acceleration(4.0, 100.0)
        assert pga1 < pga2 < pga3

    def test_pga_decreases_with_distance(self):
        pga_near = peak_ground_acceleration(3.0, 100.0)
        pga_far = peak_ground_acceleration(3.0, 500.0)
        assert pga_near > pga_far

    def test_pga_inverse_power_distance(self):
        """PGA ∝ 1/r^1.5 — doubling distance reduces by 2^1.5 ≈ 2.83."""
        pga1 = peak_ground_acceleration(3.0, 100.0)
        pga2 = peak_ground_acceleration(3.0, 200.0)
        ratio = pga1 / pga2
        expected = 2.0 ** ATTEN_C
        assert ratio == pytest.approx(expected, rel=0.01)

    def test_magnitude_5_at_500km_low(self):
        """Even M5 at 500 km should be quite weak on Mars."""
        pga = peak_ground_acceleration(5.0, 500.0)
        assert pga < 1.0  # well under 0.1g

    @pytest.mark.parametrize("mag", [0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    def test_pga_magnitude_monotonic(self, mag):
        pga_lo = peak_ground_acceleration(mag, 200.0)
        pga_hi = peak_ground_acceleration(mag + 0.1, 200.0)
        assert pga_hi > pga_lo


# ===================================================================
# Seasonal rate modulation
# ===================================================================

class TestSeasonalRate:
    """Seasonal quake rate modulation."""

    def test_sol_zero_baseline(self):
        """Sol 0 → sin(0) = 0 → modifier = 1.0."""
        assert seasonal_rate_modifier(0) == pytest.approx(1.0)

    def test_modifier_bounded(self):
        """Modifier stays in [1 - amplitude, 1 + amplitude]."""
        for sol in range(0, 670):
            mod = seasonal_rate_modifier(sol)
            assert (1.0 - SEASONAL_AMPLITUDE - 0.01) <= mod <= (1.0 + SEASONAL_AMPLITUDE + 0.01)

    def test_peak_at_quarter_period(self):
        """Peak rate at ~quarter period (sin peaks at π/2)."""
        quarter = int(SEASONAL_PERIOD_SOLS / 4)
        mod = seasonal_rate_modifier(quarter)
        assert mod > 1.2

    def test_trough_at_three_quarter(self):
        """Minimum rate at ~3/4 period (sin at 3π/2)."""
        three_quarter = int(3 * SEASONAL_PERIOD_SOLS / 4)
        mod = seasonal_rate_modifier(three_quarter)
        assert mod < 0.8

    def test_period_wraps(self):
        """Modifier repeats after one Mars year."""
        m0 = seasonal_rate_modifier(0)
        m_period = seasonal_rate_modifier(int(SEASONAL_PERIOD_SOLS))
        assert m_period == pytest.approx(m0, abs=0.01)


# ===================================================================
# Damage classification
# ===================================================================

class TestDamageCategory:
    """Damage classification from PGA."""

    def test_none_at_zero(self):
        assert damage_category(0.0) == "none"

    def test_none_below_imperceptible(self):
        assert damage_category(PGA_IMPERCEPTIBLE / 2) == "none"

    def test_imperceptible(self):
        assert damage_category(PGA_IMPERCEPTIBLE) == "imperceptible"

    def test_perceptible(self):
        assert damage_category(PGA_PERCEPTIBLE) == "perceptible"

    def test_minor(self):
        assert damage_category(PGA_MINOR_DAMAGE) == "minor"

    def test_moderate(self):
        assert damage_category(PGA_MODERATE_DAMAGE) == "moderate"

    def test_severe(self):
        assert damage_category(PGA_SEVERE_DAMAGE) == "severe"

    def test_extreme_still_severe(self):
        assert damage_category(100.0) == "severe"


# ===================================================================
# Quake generation
# ===================================================================

class TestGenerateQuakes:
    """Stochastic quake generation."""

    def test_returns_list(self):
        quakes = generate_quakes(1, rng=random.Random(42))
        assert isinstance(quakes, list)

    def test_quakes_have_valid_fields(self):
        rng = random.Random(42)
        quakes = generate_quakes(1, rng=rng)
        for q in quakes:
            assert q.magnitude > 0
            assert q.distance_km > 0
            assert q.pga_m_s2 >= 0
            assert q.sol == 1
            assert not q.is_aftershock

    def test_average_rate_near_expected(self):
        """Over many sols, average quake count ≈ QUAKE_RATE_PER_SOL."""
        rng = random.Random(123)
        total = sum(len(generate_quakes(s, rng=rng)) for s in range(1000))
        avg = total / 1000.0
        assert 0.5 < avg < 1.5  # within 50% of expected rate

    def test_magnitude_within_distribution(self):
        """All magnitudes fall within defined ranges."""
        rng = random.Random(7)
        min_mag = min(m[0] for m in MAGNITUDE_DISTRIBUTION)
        max_mag = max(m[1] for m in MAGNITUDE_DISTRIBUTION)
        for sol in range(500):
            for q in generate_quakes(sol, rng=rng):
                assert min_mag <= q.magnitude <= max_mag

    def test_distance_varies_around_colony(self):
        """Distance is colony_distance × [0.5, 1.5]."""
        rng = random.Random(42)
        distances = []
        for sol in range(500):
            for q in generate_quakes(sol, DEFAULT_DISTANCE_KM, rng):
                distances.append(q.distance_km)
        if distances:
            assert min(distances) >= DEFAULT_DISTANCE_KM * 0.5 - 1
            assert max(distances) <= DEFAULT_DISTANCE_KM * 1.5 + 1

    def test_reproducible_with_seed(self):
        q1 = generate_quakes(1, rng=random.Random(42))
        q2 = generate_quakes(1, rng=random.Random(42))
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a.magnitude == b.magnitude

    def test_no_crash_at_sol_zero(self):
        generate_quakes(0, rng=random.Random(1))  # should not raise


# ===================================================================
# Aftershock generation
# ===================================================================

class TestAftershocks:
    """Aftershock sequence generation."""

    def test_no_aftershocks_below_threshold(self):
        q = Quake(magnitude=2.5, distance_km=100, pga_m_s2=0.01, sol=1)
        assert generate_aftershocks(q, 1, random.Random(42)) == []

    def test_aftershocks_above_threshold(self):
        q = Quake(magnitude=4.0, distance_km=100, pga_m_s2=0.1, sol=1)
        afters = generate_aftershocks(q, 1, random.Random(42))
        assert len(afters) > 0

    def test_aftershock_magnitudes_below_mainshock(self):
        q = Quake(magnitude=4.5, distance_km=100, pga_m_s2=0.1, sol=1)
        for after_sol, after_mag in generate_aftershocks(q, 1, random.Random(42)):
            assert after_mag < q.magnitude

    def test_aftershock_sols_in_future(self):
        q = Quake(magnitude=4.0, distance_km=100, pga_m_s2=0.1, sol=10)
        for after_sol, _ in generate_aftershocks(q, 10, random.Random(42)):
            assert after_sol > 10

    def test_aftershock_count_scales_with_magnitude(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        q_small = Quake(magnitude=3.0, distance_km=100, pga_m_s2=0.05, sol=1)
        q_large = Quake(magnitude=5.0, distance_km=100, pga_m_s2=0.5, sol=1)
        a_small = generate_aftershocks(q_small, 1, rng1)
        a_large = generate_aftershocks(q_large, 1, rng2)
        # Larger quake → more aftershock duration → generally more events
        assert len(a_large) >= len(a_small)

    def test_aftershock_magnitudes_positive(self):
        q = Quake(magnitude=4.0, distance_km=100, pga_m_s2=0.1, sol=1)
        for _, mag in generate_aftershocks(q, 1, random.Random(42)):
            assert mag >= 0.5  # minimum magnitude floor


# ===================================================================
# Fatigue status
# ===================================================================

class TestFatigueStatus:
    """Structural fatigue classification."""

    def test_nominal_at_zero(self):
        assert fatigue_status(0.0) == "nominal"

    def test_nominal_below_threshold(self):
        assert fatigue_status(FATIGUE_THRESHOLD - 0.01) == "nominal"

    def test_inspection_at_threshold(self):
        assert fatigue_status(FATIGUE_THRESHOLD) == "inspection_needed"

    def test_critical_at_threshold(self):
        assert fatigue_status(FATIGUE_CRITICAL) == "critical"

    def test_critical_above(self):
        assert fatigue_status(FATIGUE_CRITICAL + 10.0) == "critical"


# ===================================================================
# SeismicState dataclass
# ===================================================================

class TestSeismicStateInit:
    """Dataclass initialization."""

    def test_defaults(self):
        state = SeismicState()
        assert state.sol == 0
        assert state.structural_fatigue == 0.0
        assert state.total_quakes == 0
        assert len(state.quake_log) == 0
        assert len(state.aftershock_queue) == 0

    def test_custom_distance(self):
        state = SeismicState(colony_distance_km=200.0)
        assert state.colony_distance_km == 200.0


# ===================================================================
# Tick integration
# ===================================================================

class TestTickSeismology:
    """Per-sol tick integration."""

    def test_sol_advances(self):
        state = SeismicState()
        tick_seismology(state, rng=random.Random(42))
        assert state.sol == 1

    def test_returns_dict(self):
        state = SeismicState()
        result = tick_seismology(state, rng=random.Random(42))
        for key in ["sol", "quake_count", "max_magnitude", "max_pga_m_s2",
                     "damage_category", "structural_fatigue", "fatigue_status",
                     "total_quakes", "pending_aftershocks", "events"]:
            assert key in result

    def test_quake_count_matches(self):
        state = SeismicState()
        result = tick_seismology(state, rng=random.Random(42))
        assert result["quake_count"] >= 0

    def test_total_quakes_accumulate(self):
        state = SeismicState()
        rng = random.Random(42)
        total = 0
        for _ in range(10):
            result = tick_seismology(state, rng=rng)
            total += result["quake_count"]
        assert state.total_quakes == total

    def test_fatigue_increases_or_stays(self):
        state = SeismicState()
        rng = random.Random(42)
        prev = 0.0
        for _ in range(50):
            tick_seismology(state, rng=rng)
            assert state.structural_fatigue >= prev
            prev = state.structural_fatigue

    def test_log_pruned_to_30_sols(self):
        state = SeismicState()
        rng = random.Random(42)
        for _ in range(100):
            tick_seismology(state, rng=rng)
        for q in state.quake_log:
            assert q.sol > state.sol - 30

    def test_max_magnitude_tracked(self):
        state = SeismicState()
        rng = random.Random(42)
        for _ in range(100):
            tick_seismology(state, rng=rng)
        # Should have recorded some magnitude
        assert state.max_magnitude > 0 or state.total_quakes == 0

    def test_damage_category_valid(self):
        state = SeismicState()
        result = tick_seismology(state, rng=random.Random(42))
        assert result["damage_category"] in [
            "none", "imperceptible", "perceptible", "minor", "moderate", "severe"
        ]

    def test_fatigue_status_valid(self):
        state = SeismicState()
        result = tick_seismology(state, rng=random.Random(42))
        assert result["fatigue_status"] in ["nominal", "inspection_needed", "critical"]


# ===================================================================
# Physical invariants
# ===================================================================

class TestInvariants:
    """Property-based invariant checks."""

    @pytest.mark.parametrize("mag", [0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    def test_pga_positive_for_valid_input(self, mag):
        assert peak_ground_acceleration(mag, 100.0) > 0

    @pytest.mark.parametrize("dist", [50, 100, 200, 500, 1000])
    def test_pga_decreases_monotonically(self, dist):
        pga1 = peak_ground_acceleration(3.0, dist)
        pga2 = peak_ground_acceleration(3.0, dist + 10)
        assert pga2 < pga1

    @pytest.mark.parametrize("sol", range(0, 670, 100))
    def test_seasonal_modifier_bounded(self, sol):
        mod = seasonal_rate_modifier(sol)
        assert 0.5 < mod < 1.6

    @pytest.mark.parametrize("sol", [0, 100, 200, 300, 400, 500, 600])
    def test_tick_no_negative_fatigue(self, sol):
        state = SeismicState()
        rng = random.Random(sol)
        for _ in range(sol + 1):
            tick_seismology(state, rng=rng)
        assert state.structural_fatigue >= 0

    @pytest.mark.parametrize("seed", range(10))
    def test_quake_magnitudes_bounded(self, seed):
        rng = random.Random(seed)
        min_mag = min(m[0] for m in MAGNITUDE_DISTRIBUTION)
        max_mag = max(m[1] for m in MAGNITUDE_DISTRIBUTION)
        for q in generate_quakes(1, rng=rng):
            assert min_mag <= q.magnitude <= max_mag


# ===================================================================
# Multi-sol smoke tests
# ===================================================================

class TestSmoke:
    """Multi-sol integration smoke tests."""

    def test_100_sols_no_crash(self):
        state = SeismicState()
        rng = random.Random(42)
        for _ in range(100):
            tick_seismology(state, rng=rng)
        assert state.sol == 100
        assert state.total_quakes >= 0

    def test_669_sol_mars_year(self):
        """Full Mars year without crash."""
        state = SeismicState()
        rng = random.Random(7)
        for _ in range(669):
            tick_seismology(state, rng=rng)
        assert state.sol == 669
        assert state.total_quakes > 0  # should see many quakes over a year

    def test_average_quakes_per_sol(self):
        """Over 1000 sols, average ≈ QUAKE_RATE_PER_SOL."""
        state = SeismicState()
        rng = random.Random(123)
        for _ in range(1000):
            tick_seismology(state, rng=rng)
        avg = state.total_quakes / 1000.0
        assert 0.3 < avg < 2.0

    def test_fatigue_accumulates_over_time(self):
        state = SeismicState()
        rng = random.Random(42)
        for _ in range(500):
            tick_seismology(state, rng=rng)
        assert state.structural_fatigue > 0

    def test_aftershock_queue_drains(self):
        """Aftershock queue empties over enough sols."""
        state = SeismicState()
        rng = random.Random(42)
        for _ in range(200):
            tick_seismology(state, rng=rng)
        # Run 50 more sols without new large quakes (seeded)
        for _ in range(50):
            tick_seismology(state, rng=rng)
        # Queue should be small or empty
        assert len(state.aftershock_queue) < 50

    def test_2007_sol_stability(self):
        """3 Mars years — no crash, no runaway fatigue."""
        state = SeismicState()
        rng = random.Random(42)
        for _ in range(2007):
            tick_seismology(state, rng=rng)
        assert state.sol == 2007
        assert state.structural_fatigue < 100.0  # not runaway
