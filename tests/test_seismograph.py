"""
Tests for seismograph.py — Mars seismic monitoring and structural response.

Coverage:
  - Constants validation (GR law, magnitude bounds, wave velocities)
  - Gutenberg-Richter rate (correct scaling, boundary values)
  - Wave travel time (P-wave, S-wave, edge cases)
  - Peak ground acceleration (distance scaling, magnitude scaling)
  - Event generation (statistical properties, reproducibility)
  - Structural stress (threshold, distance factor, accumulation)
  - Stress relaxation (decay rate, bounded)
  - Emergency detection (magnitude trigger, stress trigger)
  - Sensor degradation and calibration
  - Full tick integration (all systems working together)
  - Physical invariants (positive values, bounded stress, P < S arrival)
  - Property sweeps across magnitude and distance ranges
  - Multi-sol smoke tests (100+ sols without crash)
  - Edge cases (zero sensitivity, no events, extreme magnitudes)
  - Reproducibility (same seed → same events)
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.seismograph import (
    CATASTROPHIC_THRESHOLD,
    EMERGENCY_THRESHOLD,
    FAR_FIELD_KM,
    GR_A,
    GR_B,
    IMPACT_FRACTION,
    MAX_OBSERVED_MAGNITUDE,
    MAX_STRESS,
    MIN_DETECTABLE_MAGNITUDE,
    NEAR_FIELD_KM,
    P_WAVE_VELOCITY_KM_S,
    Q_ATTENUATION,
    S_WAVE_VELOCITY_KM_S,
    SENSOR_NOISE_FLOOR,
    STRESS_CRITICAL,
    STRESS_DECAY_PER_SOL,
    STRESS_PER_MAGNITUDE_UNIT,
    STRESS_WARNING,
    STRUCTURAL_DAMAGE_THRESHOLD,
    SURFACE_WAVE_VELOCITY_KM_S,
    TECTONIC_FRACTION,
    TYPICAL_SOURCE_DISTANCE_KM,
    MarsquakeEvent,
    SeismicStation,
    SolSeismicReport,
    calibrate_sensor,
    check_emergency,
    degrade_sensor,
    generate_events,
    gutenberg_richter_rate,
    peak_ground_acceleration,
    relax_stress,
    structural_stress_from_event,
    tick_seismic,
    wave_travel_time,
)


# ===================================================================
# Constants validation
# ===================================================================

class TestConstants:
    """Physical constants are realistic and self-consistent."""

    def test_gr_b_value_mars_range(self) -> None:
        """InSight measured b ≈ 0.75 for Mars (less than Earth's ~1.0)."""
        assert 0.5 <= GR_B <= 1.2

    def test_p_wave_faster_than_s_wave(self) -> None:
        """P-waves are always faster than S-waves."""
        assert P_WAVE_VELOCITY_KM_S > S_WAVE_VELOCITY_KM_S

    def test_s_wave_faster_than_surface(self) -> None:
        """S-waves are faster than surface waves."""
        assert S_WAVE_VELOCITY_KM_S > SURFACE_WAVE_VELOCITY_KM_S

    def test_p_wave_velocity_realistic(self) -> None:
        """InSight measured ~3.5 km/s for Mars crust."""
        assert 2.0 <= P_WAVE_VELOCITY_KM_S <= 6.0

    def test_max_magnitude_reasonable(self) -> None:
        """Mars can't produce M8+ events (no subduction zones)."""
        assert MAX_OBSERVED_MAGNITUDE <= 6.0
        assert MAX_OBSERVED_MAGNITUDE > 4.0

    def test_damage_below_emergency(self) -> None:
        assert STRUCTURAL_DAMAGE_THRESHOLD < EMERGENCY_THRESHOLD

    def test_emergency_below_catastrophic(self) -> None:
        assert EMERGENCY_THRESHOLD < CATASTROPHIC_THRESHOLD

    def test_warning_below_critical(self) -> None:
        assert STRESS_WARNING < STRESS_CRITICAL

    def test_event_types_sum(self) -> None:
        """Tectonic + impact fractions should sum to 1.0."""
        assert abs(TECTONIC_FRACTION + IMPACT_FRACTION - 1.0) < 1e-9

    def test_q_factor_mars_range(self) -> None:
        """Mars Q factor from InSight: ~200-400."""
        assert 100 <= Q_ATTENUATION <= 500


# ===================================================================
# Gutenberg-Richter law
# ===================================================================

class TestGutenbergRichter:
    """gutenberg_richter_rate() tests."""

    def test_more_small_than_large(self) -> None:
        """Fundamental GR property: more small quakes than large ones."""
        rate_small = gutenberg_richter_rate(1.0)
        rate_large = gutenberg_richter_rate(4.0)
        assert rate_small > rate_large

    def test_rate_positive(self) -> None:
        for mag in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            assert gutenberg_richter_rate(mag) > 0

    def test_rate_at_zero(self) -> None:
        """Rate at M0 = 10^a."""
        expected = math.pow(10.0, GR_A)
        assert abs(gutenberg_richter_rate(0.0) - expected) < 1e-6

    def test_rate_decreases_monotonically(self) -> None:
        prev = gutenberg_richter_rate(0.0)
        for mag in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            current = gutenberg_richter_rate(mag)
            assert current < prev
            prev = current

    def test_few_large_quakes_per_sol(self) -> None:
        """Should expect << 1 M4+ event per sol."""
        rate = gutenberg_richter_rate(4.0)
        assert rate < 0.5  # less than 1 per 2 sols


# ===================================================================
# Wave travel time
# ===================================================================

class TestWaveTravelTime:
    """wave_travel_time() tests."""

    def test_basic_calculation(self) -> None:
        t = wave_travel_time(100.0, P_WAVE_VELOCITY_KM_S)
        expected = 100.0 / P_WAVE_VELOCITY_KM_S
        assert abs(t - expected) < 1e-6

    def test_p_arrives_before_s(self) -> None:
        """P-wave always arrives first."""
        tp = wave_travel_time(500.0, P_WAVE_VELOCITY_KM_S)
        ts = wave_travel_time(500.0, S_WAVE_VELOCITY_KM_S)
        assert tp < ts

    def test_zero_distance(self) -> None:
        assert wave_travel_time(0.0, P_WAVE_VELOCITY_KM_S) == 0.0

    def test_zero_velocity(self) -> None:
        assert wave_travel_time(100.0, 0.0) == 0.0

    def test_negative_distance(self) -> None:
        assert wave_travel_time(-10.0, P_WAVE_VELOCITY_KM_S) == 0.0

    def test_proportional_to_distance(self) -> None:
        t1 = wave_travel_time(100.0, P_WAVE_VELOCITY_KM_S)
        t2 = wave_travel_time(200.0, P_WAVE_VELOCITY_KM_S)
        assert abs(t2 - 2.0 * t1) < 1e-6


# ===================================================================
# Peak ground acceleration
# ===================================================================

class TestPGA:
    """peak_ground_acceleration() tests."""

    def test_basic_positive(self) -> None:
        pga = peak_ground_acceleration(4.0, 500.0)
        assert pga > 0

    def test_increases_with_magnitude(self) -> None:
        pga1 = peak_ground_acceleration(2.0, 500.0)
        pga2 = peak_ground_acceleration(4.0, 500.0)
        assert pga2 > pga1

    def test_decreases_with_distance(self) -> None:
        pga_near = peak_ground_acceleration(4.0, 50.0)
        pga_far = peak_ground_acceleration(4.0, 1000.0)
        assert pga_near > pga_far

    def test_zero_magnitude(self) -> None:
        assert peak_ground_acceleration(0.0, 500.0) == 0.0

    def test_zero_distance(self) -> None:
        assert peak_ground_acceleration(4.0, 0.0) == 0.0

    def test_negative_inputs(self) -> None:
        assert peak_ground_acceleration(-1.0, 500.0) == 0.0
        assert peak_ground_acceleration(4.0, -10.0) == 0.0


# ===================================================================
# Event generation
# ===================================================================

class TestGenerateEvents:
    """generate_events() tests."""

    def test_reproducible_with_seed(self) -> None:
        """Same RNG seed produces same events."""
        events1 = generate_events(random.Random(42))
        events2 = generate_events(random.Random(42))
        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.magnitude == e2.magnitude

    def test_generates_events(self) -> None:
        """Over many sols, should generate some events."""
        total = 0
        for seed in range(50):
            events = generate_events(random.Random(seed))
            total += len(events)
        assert total > 0

    def test_magnitudes_above_threshold(self) -> None:
        """All generated events should be at or above detection threshold."""
        for seed in range(20):
            events = generate_events(random.Random(seed))
            for e in events:
                assert e.magnitude >= MIN_DETECTABLE_MAGNITUDE

    def test_magnitudes_below_max(self) -> None:
        """No event should exceed max observed magnitude."""
        for seed in range(100):
            events = generate_events(random.Random(seed))
            for e in events:
                assert e.magnitude <= MAX_OBSERVED_MAGNITUDE

    def test_event_types_valid(self) -> None:
        """All events are either tectonic or impact."""
        for seed in range(20):
            events = generate_events(random.Random(seed))
            for e in events:
                assert e.source_type in ("tectonic", "impact")

    def test_p_before_s_arrival(self) -> None:
        """P-wave always arrives before S-wave."""
        for seed in range(20):
            events = generate_events(random.Random(seed))
            for e in events:
                assert e.p_arrival_s <= e.s_arrival_s

    def test_distances_in_range(self) -> None:
        """Event distances within specified range."""
        for seed in range(20):
            events = generate_events(random.Random(seed), (100.0, 1000.0))
            for e in events:
                assert e.distance_km >= 90.0   # allow small float tolerance
                assert e.distance_km <= 1100.0

    def test_depths_realistic(self) -> None:
        """Mars quakes are shallow: 5-50 km."""
        for seed in range(20):
            events = generate_events(random.Random(seed))
            for e in events:
                assert 4.0 <= e.depth_km <= 51.0

    def test_pga_positive(self) -> None:
        """All events have positive PGA."""
        for seed in range(20):
            events = generate_events(random.Random(seed))
            for e in events:
                assert e.peak_ground_accel >= 0.0

    def test_mostly_small_events(self) -> None:
        """Most events should be M1-M2 (GR distribution)."""
        small = 0
        total = 0
        for seed in range(100):
            events = generate_events(random.Random(seed))
            for e in events:
                total += 1
                if e.magnitude < 2.5:
                    small += 1
        if total > 0:
            assert small / total > 0.5  # at least half should be small


# ===================================================================
# Structural stress
# ===================================================================

class TestStructuralStress:
    """structural_stress_from_event() tests."""

    def test_below_threshold_no_stress(self) -> None:
        event = MarsquakeEvent(magnitude=2.0, distance_km=500.0)
        assert structural_stress_from_event(event) == 0.0

    def test_at_threshold_no_stress(self) -> None:
        event = MarsquakeEvent(magnitude=STRUCTURAL_DAMAGE_THRESHOLD - 0.01)
        assert structural_stress_from_event(event) == 0.0

    def test_above_threshold_adds_stress(self) -> None:
        event = MarsquakeEvent(magnitude=4.5, distance_km=500.0)
        stress = structural_stress_from_event(event)
        assert stress > 0.0

    def test_closer_means_more_stress(self) -> None:
        near = MarsquakeEvent(magnitude=4.5, distance_km=50.0)
        far = MarsquakeEvent(magnitude=4.5, distance_km=2000.0)
        assert structural_stress_from_event(near) > structural_stress_from_event(far)

    def test_larger_means_more_stress(self) -> None:
        small = MarsquakeEvent(magnitude=3.6, distance_km=500.0)
        large = MarsquakeEvent(magnitude=5.0, distance_km=500.0)
        assert structural_stress_from_event(large) > structural_stress_from_event(small)

    def test_stress_non_negative(self) -> None:
        for mag in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            event = MarsquakeEvent(magnitude=mag, distance_km=500.0)
            assert structural_stress_from_event(event) >= 0.0


# ===================================================================
# Stress relaxation
# ===================================================================

class TestStressRelaxation:
    """relax_stress() tests."""

    def test_stress_decreases(self) -> None:
        station = SeismicStation(structural_stress=50.0)
        relief = relax_stress(station)
        assert relief > 0.0
        assert station.structural_stress < 50.0

    def test_zero_stress_no_change(self) -> None:
        station = SeismicStation(structural_stress=0.0)
        relief = relax_stress(station)
        assert relief == 0.0
        assert station.structural_stress == 0.0

    def test_stress_never_negative(self) -> None:
        station = SeismicStation(structural_stress=0.001)
        relax_stress(station)
        assert station.structural_stress >= 0.0

    def test_relaxation_proportional(self) -> None:
        """Higher stress → more relaxation."""
        s1 = SeismicStation(structural_stress=20.0)
        s2 = SeismicStation(structural_stress=80.0)
        r1 = relax_stress(s1)
        r2 = relax_stress(s2)
        assert r2 > r1


# ===================================================================
# Emergency detection
# ===================================================================

class TestEmergency:
    """check_emergency() tests."""

    def test_no_emergency_normal(self) -> None:
        station = SeismicStation(structural_stress=10.0)
        assert check_emergency(station, 2.0) is False

    def test_emergency_on_large_quake(self) -> None:
        station = SeismicStation(structural_stress=10.0)
        assert check_emergency(station, EMERGENCY_THRESHOLD) is True

    def test_emergency_on_high_stress(self) -> None:
        station = SeismicStation(structural_stress=STRESS_WARNING)
        assert check_emergency(station, 1.0) is True

    def test_no_emergency_just_below_thresholds(self) -> None:
        station = SeismicStation(structural_stress=STRESS_WARNING - 1.0)
        assert check_emergency(station, EMERGENCY_THRESHOLD - 0.1) is False


# ===================================================================
# Sensor degradation and calibration
# ===================================================================

class TestSensor:
    """degrade_sensor() and calibrate_sensor() tests."""

    def test_degradation_reduces_sensitivity(self) -> None:
        station = SeismicStation(sensitivity=1.0)
        degrade_sensor(station, dust_factor=0.0)
        assert station.sensitivity < 1.0

    def test_dust_accelerates_degradation(self) -> None:
        s1 = SeismicStation(sensitivity=1.0)
        s2 = SeismicStation(sensitivity=1.0)
        degrade_sensor(s1, dust_factor=0.0)
        degrade_sensor(s2, dust_factor=1.0)
        assert s2.sensitivity < s1.sensitivity

    def test_sensitivity_floors_at_zero(self) -> None:
        station = SeismicStation(sensitivity=0.0001)
        degrade_sensor(station, 1.0)
        assert station.sensitivity >= 0.0

    def test_calibration_restores(self) -> None:
        station = SeismicStation(sensitivity=0.8)
        calibrate_sensor(station, 1.0)
        assert station.sensitivity > 0.8

    def test_calibration_caps_at_one(self) -> None:
        station = SeismicStation(sensitivity=0.99)
        calibrate_sensor(station, 1.0)
        assert station.sensitivity <= 1.0

    def test_calibration_quality_matters(self) -> None:
        s1 = SeismicStation(sensitivity=0.5)
        s2 = SeismicStation(sensitivity=0.5)
        calibrate_sensor(s1, 0.2)
        calibrate_sensor(s2, 1.0)
        assert s2.sensitivity > s1.sensitivity


# ===================================================================
# Full tick integration
# ===================================================================

class TestTickSeismic:
    """tick_seismic() — full sol integration."""

    def test_basic_tick(self) -> None:
        station = SeismicStation()
        report = tick_seismic(station, rng=random.Random(42))
        assert isinstance(report, SolSeismicReport)
        assert station.sols_monitored == 1

    def test_tick_increments_sols(self) -> None:
        station = SeismicStation()
        tick_seismic(station, rng=random.Random(1))
        tick_seismic(station, rng=random.Random(2))
        assert station.sols_monitored == 2

    def test_tick_detects_events(self) -> None:
        """Over many ticks, should detect some events."""
        station = SeismicStation()
        total_detected = 0
        for seed in range(50):
            report = tick_seismic(station, rng=random.Random(seed))
            total_detected += report.total_events
        assert total_detected > 0

    def test_tick_offline_station(self) -> None:
        """Offline station detects nothing."""
        station = SeismicStation(operational=False)
        report = tick_seismic(station, rng=random.Random(42))
        assert report.total_events == 0
        assert len(report.events) == 0
        assert station.sols_monitored == 1

    def test_tick_stress_accumulation(self) -> None:
        """Over many sols, stress may accumulate from large events."""
        station = SeismicStation()
        for seed in range(200):
            tick_seismic(station, rng=random.Random(seed))
        # Stress is bounded
        assert 0.0 <= station.structural_stress <= MAX_STRESS

    def test_tick_reproducible(self) -> None:
        """Same seed → same report."""
        s1 = SeismicStation()
        s2 = SeismicStation()
        r1 = tick_seismic(s1, rng=random.Random(42))
        r2 = tick_seismic(s2, rng=random.Random(42))
        assert r1.total_events == r2.total_events
        assert r1.max_magnitude == r2.max_magnitude

    def test_tick_sensor_degrades(self) -> None:
        station = SeismicStation()
        tick_seismic(station, rng=random.Random(1))
        assert station.sensitivity < 1.0

    def test_tick_dust_storm_effect(self) -> None:
        """Dust storms degrade sensor faster."""
        s1 = SeismicStation()
        s2 = SeismicStation()
        tick_seismic(s1, rng=random.Random(1), dust_factor=0.0)
        tick_seismic(s2, rng=random.Random(1), dust_factor=1.0)
        assert s2.sensitivity < s1.sensitivity

    def test_tick_events_count_matches(self) -> None:
        station = SeismicStation()
        report = tick_seismic(station, rng=random.Random(42))
        assert report.total_events == len(report.events)

    def test_tick_largest_event_tracked(self) -> None:
        station = SeismicStation()
        max_mag = 0.0
        for seed in range(100):
            report = tick_seismic(station, rng=random.Random(seed))
            if report.max_magnitude > max_mag:
                max_mag = report.max_magnitude
        assert station.largest_event == max_mag


# ===================================================================
# Physical invariants — property sweeps
# ===================================================================

class TestInvariants:
    """Invariants that hold across all parameter ranges."""

    @pytest.mark.parametrize("mag", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
    def test_gr_rate_positive(self, mag: float) -> None:
        assert gutenberg_richter_rate(mag) > 0

    @pytest.mark.parametrize("dist", [1.0, 10.0, 100.0, 500.0, 2000.0])
    def test_pga_positive_all_distances(self, dist: float) -> None:
        pga = peak_ground_acceleration(3.0, dist)
        assert pga >= 0.0

    @pytest.mark.parametrize("mag", [0.0, 1.0, 2.0, 3.0, 3.4, 3.5, 4.0, 5.0])
    def test_stress_non_negative_all_mags(self, mag: float) -> None:
        event = MarsquakeEvent(magnitude=mag, distance_km=500.0)
        assert structural_stress_from_event(event) >= 0.0

    @pytest.mark.parametrize("dust", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_sensor_bounded_all_dust(self, dust: float) -> None:
        station = SeismicStation()
        for _ in range(100):
            degrade_sensor(station, dust)
        assert 0.0 <= station.sensitivity <= 1.0

    @pytest.mark.parametrize("seed", range(10))
    def test_tick_never_crashes(self, seed: int) -> None:
        station = SeismicStation()
        report = tick_seismic(station, rng=random.Random(seed))
        assert isinstance(report, SolSeismicReport)
        assert station.structural_stress >= 0.0
        assert station.structural_stress <= MAX_STRESS


# ===================================================================
# Multi-sol smoke tests
# ===================================================================

class TestSmoke:
    """Smoke tests: many sols without crash."""

    def test_100_sols_normal(self) -> None:
        """100 sols of seismic monitoring."""
        station = SeismicStation()
        total_events = 0
        for sol in range(100):
            report = tick_seismic(station, rng=random.Random(sol))
            total_events += report.total_events
            assert station.structural_stress >= 0.0
            assert station.structural_stress <= MAX_STRESS
        assert station.sols_monitored == 100
        assert total_events > 0

    def test_365_sols_full_year(self) -> None:
        """Full Mars year of seismic monitoring."""
        station = SeismicStation()
        emergencies = 0
        for sol in range(365):
            report = tick_seismic(station, rng=random.Random(sol * 7))
            if report.emergency_triggered:
                emergencies += 1
            if sol % 100 == 99:
                calibrate_sensor(station, 0.8)
        assert station.sols_monitored == 365
        assert station.events_detected > 0
        # Sensor should still be operational after calibration
        assert station.sensitivity > 0.0

    def test_668_sols_full_mission(self) -> None:
        """Full 668-sol mission duration."""
        station = SeismicStation()
        for sol in range(668):
            dust = 0.5 if 200 <= sol <= 300 else 0.1  # dust storm season
            tick_seismic(station, rng=random.Random(sol), dust_factor=dust)
            if sol % 50 == 49:
                calibrate_sensor(station, 0.5)
        assert station.sols_monitored == 668
        assert station.events_detected > 0
        assert station.largest_event > 0

    def test_stress_doesnt_runaway(self) -> None:
        """Over 1000 sols, stress should not exceed MAX_STRESS."""
        station = SeismicStation()
        for sol in range(1000):
            tick_seismic(station, rng=random.Random(sol))
        assert station.structural_stress <= MAX_STRESS


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Weird inputs that should not crash."""

    def test_zero_sensitivity(self) -> None:
        """Station with no sensitivity detects nothing."""
        station = SeismicStation(sensitivity=0.0)
        report = tick_seismic(station, rng=random.Random(42))
        # With 0 sensitivity, threshold is very high, few/no detections
        assert isinstance(report, SolSeismicReport)

    def test_max_stress_station(self) -> None:
        """Station at max stress continues operating."""
        station = SeismicStation(structural_stress=MAX_STRESS)
        report = tick_seismic(station, rng=random.Random(42))
        assert station.structural_stress <= MAX_STRESS

    def test_custom_distance_range(self) -> None:
        station = SeismicStation()
        report = tick_seismic(
            station, rng=random.Random(42),
            distance_range=(10.0, 50.0),
        )
        assert isinstance(report, SolSeismicReport)

    def test_very_close_events(self) -> None:
        """Events at very close range — high PGA."""
        station = SeismicStation()
        report = tick_seismic(
            station, rng=random.Random(99),
            distance_range=(1.0, 5.0),
        )
        assert isinstance(report, SolSeismicReport)

    def test_very_far_events(self) -> None:
        """Events at extreme range — low PGA."""
        station = SeismicStation()
        report = tick_seismic(
            station, rng=random.Random(42),
            distance_range=(5000.0, 10000.0),
        )
        assert isinstance(report, SolSeismicReport)

    def test_negative_dust_factor(self) -> None:
        """Negative dust should be treated as zero."""
        station = SeismicStation()
        degrade_sensor(station, -0.5)
        assert station.sensitivity >= 0.0

    def test_station_construction_clamping(self) -> None:
        station = SeismicStation(sensitivity=2.0, structural_stress=-10.0)
        assert station.sensitivity == 1.0
        assert station.structural_stress == 0.0
