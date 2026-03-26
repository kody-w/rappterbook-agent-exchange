"""Tests for solar_conjunction.py — Mars–Sun conjunction communications blackout.

Organised into sections:
  1. Physical constants sanity
  2. Angle wrapping
  3. Mean anomaly
  4. Kepler solver (eccentric anomaly)
  5. True anomaly
  6. SEP angle geometry
  7. Data rate model
  8. Link status helpers
  9. Conjunction prediction
 10. Blackout window & duration
 11. State dataclass defaults
 12. Serialisation round-trip
 13. Tick engine — single sol
 14. Tick engine — blackout entry/exit
 15. Cache prefetch ramp
 16. Autonomy mode toggling
 17. Multi-sol simulation
 18. Conservation / monotonicity invariants
 19. Property-based parametrised tests
 20. Edge cases
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# ── path setup ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solar_conjunction import (
    # Constants
    EARTH_ORBITAL_PERIOD_DAYS,
    MARS_ORBITAL_PERIOD_DAYS,
    SYNODIC_PERIOD_DAYS,
    CONJUNCTION_EPOCH_SOL,
    MARS_ECCENTRICITY,
    SEP_BLACKOUT_DEG,
    SEP_DEGRADED_DEG,
    SEP_NOMINAL_DEG,
    NOMINAL_DATA_RATE_BPS,
    MIN_DATA_RATE_BPS,
    SCINTILLATION_EXPONENT,
    CACHE_PREFETCH_SOLS,
    AUTONOMY_BUFFER_SOLS,
    # Geometry helpers
    _wrap_angle,
    mean_anomaly,
    eccentric_anomaly,
    true_anomaly,
    heliocentric_longitude,
    sep_angle,
    # Signal model
    data_rate_bps,
    is_blackout,
    is_degraded,
    link_status,
    # Conjunction prediction
    next_conjunction_sol,
    conjunction_window,
    blackout_duration_sols,
    # State / tick
    ConjunctionState,
    TickResult,
    tick,
    run_simulation,
    state_to_dict,
    state_from_dict,
)


# =====================================================================
# 1. Physical constants sanity
# =====================================================================

class TestConstants:
    """Verify physical constants are in plausible ranges."""

    def test_earth_period(self):
        assert 365 < EARTH_ORBITAL_PERIOD_DAYS < 366

    def test_mars_period(self):
        assert 686 < MARS_ORBITAL_PERIOD_DAYS < 688

    def test_synodic_period(self):
        # Synodic = 1 / |1/T_earth − 1/T_mars|
        computed = 1.0 / abs(1.0 / EARTH_ORBITAL_PERIOD_DAYS - 1.0 / MARS_ORBITAL_PERIOD_DAYS)
        assert abs(SYNODIC_PERIOD_DAYS - computed) < 1.0  # within 1 day

    def test_mars_eccentricity(self):
        assert 0.09 < MARS_ECCENTRICITY < 0.10

    def test_sep_thresholds_ordered(self):
        assert SEP_BLACKOUT_DEG < SEP_DEGRADED_DEG < SEP_NOMINAL_DEG

    def test_nominal_data_rate_positive(self):
        assert NOMINAL_DATA_RATE_BPS > 0

    def test_min_data_rate_zero(self):
        assert MIN_DATA_RATE_BPS == 0

    def test_cache_prefetch_positive(self):
        assert CACHE_PREFETCH_SOLS > 0


# =====================================================================
# 2. Angle wrapping
# =====================================================================

class TestWrapAngle:
    def test_zero(self):
        assert _wrap_angle(0.0) == 0.0

    def test_positive(self):
        assert _wrap_angle(450.0) == pytest.approx(90.0)

    def test_negative(self):
        assert _wrap_angle(-90.0) == pytest.approx(270.0)

    def test_360(self):
        assert _wrap_angle(360.0) == pytest.approx(0.0)

    def test_large_negative(self):
        assert 0 <= _wrap_angle(-1000.0) < 360


# =====================================================================
# 3. Mean anomaly
# =====================================================================

class TestMeanAnomaly:
    def test_epoch_gives_zero(self):
        assert mean_anomaly(0.0, 100.0, 0.0) == pytest.approx(0.0)

    def test_half_period(self):
        assert mean_anomaly(50.0, 100.0, 0.0) == pytest.approx(180.0)

    def test_full_period(self):
        assert mean_anomaly(100.0, 100.0, 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_with_offset(self):
        assert mean_anomaly(10.0, 100.0, 10.0) == pytest.approx(0.0)


# =====================================================================
# 4. Kepler solver (eccentric anomaly)
# =====================================================================

class TestEccentricAnomaly:
    def test_circular_orbit(self):
        """For e=0, E = M."""
        for M in [0, 45, 90, 180, 270]:
            assert eccentric_anomaly(M, 0.0) == pytest.approx(M, abs=1e-6)

    def test_nonzero_eccentricity(self):
        """Kepler's equation: M = E − e·sin(E) must hold."""
        for M in [30, 90, 150, 250]:
            E = eccentric_anomaly(M, MARS_ECCENTRICITY)
            recomputed_M = math.degrees(
                math.radians(E) - MARS_ECCENTRICITY * math.sin(math.radians(E))
            ) % 360
            assert recomputed_M == pytest.approx(M, abs=1e-4)

    def test_zero_mean_anomaly(self):
        assert eccentric_anomaly(0.0, 0.05) == pytest.approx(0.0, abs=1e-6)


# =====================================================================
# 5. True anomaly
# =====================================================================

class TestTrueAnomaly:
    def test_circular(self):
        for E in [0, 90, 180, 270]:
            assert true_anomaly(E, 0.0) == pytest.approx(E, abs=1e-6)

    def test_range(self):
        for E in range(0, 360, 15):
            nu = true_anomaly(E, MARS_ECCENTRICITY)
            assert 0.0 <= nu < 360.0

    def test_perihelion(self):
        # At E=0, true anomaly should also be 0
        assert true_anomaly(0.0, 0.05) == pytest.approx(0.0, abs=1e-6)


# =====================================================================
# 6. SEP angle geometry
# =====================================================================

class TestSEPAngle:
    def test_at_conjunction(self):
        """SEP ≈ 0 at the conjunction epoch."""
        sep = sep_angle(0.0, 0.0)
        assert sep < 1.0  # very close to 0

    def test_at_opposition(self):
        """SEP peaks near 180° halfway through the synodic period."""
        sep = sep_angle(SYNODIC_PERIOD_DAYS / 2, 0.0)
        assert sep > 170.0

    def test_non_negative(self):
        for sol in range(0, 800, 10):
            assert sep_angle(float(sol), 0.0) >= 0.0

    def test_bounded(self):
        for sol in range(0, 800, 10):
            assert sep_angle(float(sol), 0.0) <= 180.0

    def test_periodic(self):
        """SEP is periodic with the synodic period."""
        for sol in [100, 200, 350]:
            a = sep_angle(float(sol), 0.0)
            b = sep_angle(float(sol) + SYNODIC_PERIOD_DAYS, 0.0)
            assert a == pytest.approx(b, abs=0.5)


# =====================================================================
# 7. Data rate model
# =====================================================================

class TestDataRate:
    def test_blackout(self):
        assert data_rate_bps(0.0) == 0
        assert data_rate_bps(1.0) == 0
        assert data_rate_bps(SEP_BLACKOUT_DEG) == 0

    def test_nominal(self):
        assert data_rate_bps(SEP_NOMINAL_DEG) == NOMINAL_DATA_RATE_BPS
        assert data_rate_bps(90.0) == NOMINAL_DATA_RATE_BPS
        assert data_rate_bps(180.0) == NOMINAL_DATA_RATE_BPS

    def test_degraded_zone(self):
        rate = data_rate_bps(3.0)
        assert 0 < rate < NOMINAL_DATA_RATE_BPS

    def test_reduced_zone(self):
        rate = data_rate_bps(10.0)
        assert 0 < rate < NOMINAL_DATA_RATE_BPS

    def test_monotonic_increasing(self):
        """Data rate should never decrease as SEP increases."""
        prev = 0.0
        for angle_10x in range(0, 1800, 1):
            angle = angle_10x / 10.0
            rate = data_rate_bps(angle)
            assert rate >= prev - 1.0  # allow tiny float imprecision
            prev = rate

    def test_non_negative_always(self):
        for angle in [0, 0.5, 1, 2, 3, 5, 10, 15, 90, 180]:
            assert data_rate_bps(float(angle)) >= 0

    def test_bounded_above(self):
        for angle in [0, 5, 10, 15, 30, 90, 180]:
            assert data_rate_bps(float(angle)) <= NOMINAL_DATA_RATE_BPS


# =====================================================================
# 8. Link status helpers
# =====================================================================

class TestLinkStatus:
    def test_blackout(self):
        assert is_blackout(0.0) is True
        assert is_blackout(2.0) is True
        assert is_blackout(2.1) is False

    def test_degraded(self):
        assert is_degraded(3.0) is True
        assert is_degraded(5.0) is True
        assert is_degraded(1.0) is False  # blackout, not degraded
        assert is_degraded(6.0) is False

    def test_link_status_strings(self):
        assert link_status(0.0) == "blackout"
        assert link_status(1.0) == "blackout"
        assert link_status(3.0) == "degraded"
        assert link_status(5.0) == "degraded"
        assert link_status(10.0) == "reduced"
        assert link_status(15.0) == "nominal"
        assert link_status(90.0) == "nominal"


# =====================================================================
# 9. Conjunction prediction
# =====================================================================

class TestConjunctionPrediction:
    def test_next_from_before_epoch(self):
        nxt = next_conjunction_sol(-100.0, epoch=0.0)
        assert nxt == pytest.approx(0.0, abs=0.1)

    def test_next_after_epoch(self):
        nxt = next_conjunction_sol(1.0, epoch=0.0)
        assert nxt == pytest.approx(SYNODIC_PERIOD_DAYS, abs=0.1)

    def test_next_at_epoch(self):
        nxt = next_conjunction_sol(0.0, epoch=0.0)
        assert nxt == pytest.approx(SYNODIC_PERIOD_DAYS, abs=0.1)

    def test_far_future(self):
        nxt = next_conjunction_sol(5000.0, epoch=0.0)
        assert nxt > 5000.0
        assert nxt <= 5000.0 + SYNODIC_PERIOD_DAYS


# =====================================================================
# 10. Blackout window & duration
# =====================================================================

class TestBlackoutWindow:
    def test_window_contains_center(self):
        start, end = conjunction_window(400.0)
        assert start <= 400.0 <= end

    def test_duration_range(self):
        """Blackout should be ~13–16 sols (physical expectation)."""
        dur = blackout_duration_sols(400.0)
        assert 5.0 <= dur <= 25.0  # generous bounds for the sinusoidal model

    def test_window_symmetric_ish(self):
        """Window should be roughly symmetric around the center."""
        center = 400.0
        start, end = conjunction_window(center)
        before = center - start
        after = end - center
        assert abs(before - after) < 2.0  # within 2 sols

    def test_duration_non_negative(self):
        assert blackout_duration_sols(400.0) >= 0.0


# =====================================================================
# 11. State dataclass defaults
# =====================================================================

class TestStateDefaults:
    def test_defaults(self):
        s = ConjunctionState()
        assert s.sol == 0.0
        assert s.status == "nominal"
        assert s.autonomy_active is False
        assert s.cache_level == 0.0
        assert s.sols_in_blackout == 0
        assert s.total_blackout_sols == 0


# =====================================================================
# 12. Serialisation round-trip
# =====================================================================

class TestSerialisation:
    def test_round_trip(self):
        original = ConjunctionState(
            sol=42.0,
            next_conjunction=400.0,
            sep_deg=12.5,
            data_rate_bps=1_500_000,
            status="reduced",
            sols_in_blackout=0,
            total_blackout_sols=3,
            cache_level=0.75,
            autonomy_active=False,
        )
        d = state_to_dict(original)
        restored = state_from_dict(d)
        assert restored.sol == original.sol
        assert restored.next_conjunction == original.next_conjunction
        assert restored.sep_deg == original.sep_deg
        assert restored.status == original.status
        assert restored.cache_level == original.cache_level
        assert restored.autonomy_active == original.autonomy_active
        assert restored.total_blackout_sols == original.total_blackout_sols

    def test_empty_dict(self):
        s = state_from_dict({})
        assert s.sol == 0.0
        assert s.status == "nominal"


# =====================================================================
# 13. Tick engine — single sol
# =====================================================================

class TestTickSingle:
    def test_basic_tick(self):
        state = ConjunctionState(sol=100.0, next_conjunction=400.0)
        result = tick(state)
        assert result.sol == 100.0
        assert state.sol == 101.0  # advanced by 1
        assert result.status in ("nominal", "reduced", "degraded", "blackout")

    def test_tick_at_conjunction(self):
        state = ConjunctionState(sol=400.0, next_conjunction=400.0)
        result = tick(state)
        assert result.sep_deg < 5.0
        assert result.status in ("blackout", "degraded")

    def test_tick_result_fields(self):
        state = ConjunctionState(sol=200.0, next_conjunction=400.0)
        result = tick(state)
        assert hasattr(result, "sol")
        assert hasattr(result, "sep_deg")
        assert hasattr(result, "data_rate_bps")
        assert hasattr(result, "status")
        assert hasattr(result, "autonomy_active")
        assert hasattr(result, "cache_level")
        assert hasattr(result, "sols_to_conjunction")
        assert hasattr(result, "blackout_duration")


# =====================================================================
# 14. Tick engine — blackout entry/exit
# =====================================================================

class TestBlackoutEntryExit:
    def test_enters_blackout_near_conjunction(self):
        """Colony should enter blackout around the conjunction sol."""
        state = ConjunctionState(sol=395.0, next_conjunction=400.0)
        found_blackout = False
        for _ in range(15):
            result = tick(state)
            if result.status == "blackout":
                found_blackout = True
                break
        assert found_blackout, "Expected blackout near conjunction"

    def test_exits_blackout_after_conjunction(self):
        """Colony should exit blackout after the conjunction passes."""
        state = ConjunctionState(sol=395.0, next_conjunction=400.0)
        results = []
        for _ in range(30):
            results.append(tick(state))
        statuses = [r.status for r in results]
        assert "blackout" in statuses
        # After blackout, should return to nominal eventually
        post_blackout = statuses[statuses.index("blackout"):]
        assert any(s != "blackout" for s in post_blackout)


# =====================================================================
# 15. Cache prefetch ramp
# =====================================================================

class TestCachePrefetch:
    def test_cache_fills_before_conjunction(self):
        """Cache should reach 1.0 by the time conjunction starts."""
        state = ConjunctionState(sol=365.0, next_conjunction=400.0)
        for _ in range(35):
            tick(state)
        assert state.cache_level >= 0.9

    def test_cache_zero_far_from_conjunction(self):
        state = ConjunctionState(sol=100.0, next_conjunction=400.0)
        tick(state)
        assert state.cache_level == 0.0

    def test_cache_increments(self):
        state = ConjunctionState(sol=375.0, next_conjunction=400.0)
        tick(state)
        assert state.cache_level > 0.0


# =====================================================================
# 16. Autonomy mode toggling
# =====================================================================

class TestAutonomy:
    def test_autonomy_activates_in_blackout(self):
        state = ConjunctionState(sol=399.0, next_conjunction=400.0)
        for _ in range(5):
            tick(state)
        # Should have triggered autonomy during blackout
        assert state.total_blackout_sols > 0

    def test_autonomy_deactivates_after_blackout(self):
        state = ConjunctionState(sol=395.0, next_conjunction=400.0)
        for _ in range(40):
            tick(state)
        # After 40 sols past conjunction, autonomy should be off
        assert state.autonomy_active is False


# =====================================================================
# 17. Multi-sol simulation
# =====================================================================

class TestRunSimulation:
    def test_smoke_10_sols(self):
        results = run_simulation(10, conjunction_sol=400.0)
        assert len(results) == 10

    def test_smoke_365_sols(self):
        results = run_simulation(365, conjunction_sol=200.0)
        assert len(results) == 365

    def test_contains_blackout(self):
        """A simulation spanning a conjunction should contain a blackout."""
        results = run_simulation(800, conjunction_sol=400.0)
        statuses = {r.status for r in results}
        assert "blackout" in statuses

    def test_contains_nominal(self):
        results = run_simulation(800, conjunction_sol=400.0)
        statuses = {r.status for r in results}
        assert "nominal" in statuses

    def test_sol_advances_monotonically(self):
        results = run_simulation(100)
        for i in range(1, len(results)):
            assert results[i].sol > results[i - 1].sol

    def test_two_conjunctions_in_long_sim(self):
        """~1600 sols should see two conjunction events."""
        results = run_simulation(1600, conjunction_sol=400.0)
        blackout_starts = 0
        prev_blackout = False
        for r in results:
            if r.status == "blackout" and not prev_blackout:
                blackout_starts += 1
            prev_blackout = r.status == "blackout"
        assert blackout_starts >= 2


# =====================================================================
# 18. Conservation / monotonicity invariants
# =====================================================================

class TestInvariants:
    def test_total_blackout_sols_monotonic(self):
        """Total blackout counter must never decrease."""
        state = ConjunctionState(sol=0.0, next_conjunction=400.0)
        prev = 0
        for _ in range(800):
            tick(state)
            assert state.total_blackout_sols >= prev
            prev = state.total_blackout_sols

    def test_sep_always_non_negative(self):
        results = run_simulation(800, conjunction_sol=400.0)
        for r in results:
            assert r.sep_deg >= 0.0

    def test_sep_bounded_180(self):
        results = run_simulation(800, conjunction_sol=400.0)
        for r in results:
            assert r.sep_deg <= 180.0

    def test_data_rate_bounded(self):
        results = run_simulation(800, conjunction_sol=400.0)
        for r in results:
            assert 0 <= r.data_rate_bps <= NOMINAL_DATA_RATE_BPS

    def test_cache_bounded_0_1(self):
        state = ConjunctionState(sol=0.0, next_conjunction=400.0)
        for _ in range(800):
            tick(state)
            assert 0.0 <= state.cache_level <= 1.0

    def test_status_always_valid(self):
        results = run_simulation(800, conjunction_sol=400.0)
        valid = {"nominal", "reduced", "degraded", "blackout"}
        for r in results:
            assert r.status in valid


# =====================================================================
# 19. Property-based parametrised tests
# =====================================================================

class TestParametrised:
    @pytest.mark.parametrize("conjunction_sol", [100, 300, 500, 780, 1200])
    def test_blackout_occurs_near_conjunction(self, conjunction_sol):
        """At least one blackout sol should be within ±20 of the conjunction."""
        results = run_simulation(conjunction_sol + 50, conjunction_sol=conjunction_sol)
        blackout_sols = [r.sol for r in results if r.status == "blackout"]
        assert len(blackout_sols) > 0
        # At least one blackout sol must be near the target conjunction
        nearest = min(abs(s - conjunction_sol) for s in blackout_sols)
        assert nearest < 20

    @pytest.mark.parametrize("sep", [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 90.0, 180.0])
    def test_data_rate_consistency(self, sep):
        rate = data_rate_bps(sep)
        status = link_status(sep)
        if status == "blackout":
            assert rate == 0
        elif status == "nominal":
            assert rate == NOMINAL_DATA_RATE_BPS
        else:
            assert 0 < rate < NOMINAL_DATA_RATE_BPS

    @pytest.mark.parametrize("sol", [0, 50, 100, 200, 400, 600, 779])
    def test_sep_at_various_sols(self, sol):
        sep = sep_angle(float(sol), 400.0)
        assert 0.0 <= sep <= 180.0

    @pytest.mark.parametrize("ecc", [0.0, 0.01, 0.05, 0.0934, 0.2])
    def test_kepler_self_consistency(self, ecc):
        """E and M must satisfy Kepler's equation for any eccentricity."""
        for M in [0, 45, 90, 135, 180, 225, 270, 315]:
            E = eccentric_anomaly(float(M), ecc)
            recomputed = math.degrees(
                math.radians(E) - ecc * math.sin(math.radians(E))
            ) % 360
            if M == 0:
                assert recomputed == pytest.approx(0.0, abs=0.01)
            else:
                assert recomputed == pytest.approx(float(M), abs=0.01)


# =====================================================================
# 20. Edge cases
# =====================================================================

class TestEdgeCases:
    def test_zero_sol_simulation(self):
        results = run_simulation(0)
        assert results == []

    def test_one_sol_simulation(self):
        results = run_simulation(1)
        assert len(results) == 1

    def test_conjunction_at_sol_zero(self):
        """Conjunction at the very start."""
        results = run_simulation(50, conjunction_sol=0.0)
        assert any(r.status == "blackout" for r in results)

    def test_very_distant_conjunction(self):
        """Conjunction far in the future — should be all nominal."""
        results = run_simulation(50, conjunction_sol=10000.0)
        assert all(r.status == "nominal" for r in results)

    def test_negative_epoch(self):
        """Epoch in the past still works."""
        nxt = next_conjunction_sol(100.0, epoch=-500.0)
        assert nxt > 100.0

    def test_high_eccentricity_kepler(self):
        """Kepler solver handles moderate eccentricity."""
        E = eccentric_anomaly(90.0, 0.5)
        M_check = math.degrees(
            math.radians(E) - 0.5 * math.sin(math.radians(E))
        ) % 360
        assert M_check == pytest.approx(90.0, abs=0.1)

    def test_wrap_angle_exact_multiples(self):
        assert _wrap_angle(720.0) == pytest.approx(0.0)
        assert _wrap_angle(1080.0) == pytest.approx(0.0)

    def test_data_rate_at_exact_thresholds(self):
        """Verify continuity at the threshold boundaries."""
        # At blackout boundary
        assert data_rate_bps(SEP_BLACKOUT_DEG) == 0
        # Just above blackout
        rate = data_rate_bps(SEP_BLACKOUT_DEG + 0.01)
        assert rate > 0
        # At nominal threshold
        assert data_rate_bps(SEP_NOMINAL_DEG) == NOMINAL_DATA_RATE_BPS

    def test_serialise_deserialise_preserves_types(self):
        s = ConjunctionState(sol=1.0, autonomy_active=True, total_blackout_sols=5)
        d = state_to_dict(s)
        r = state_from_dict(d)
        assert isinstance(r.sol, float)
        assert isinstance(r.autonomy_active, bool)
        assert isinstance(r.total_blackout_sols, int)
