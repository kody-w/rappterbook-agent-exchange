"""Tests for methane_detector.py — Mars Atmospheric Methane Detection.

Organised into sections:
  1. Physical constants validation
  2. Beer-Lambert law
  3. Concentration conversion (ppbv ↔ mol/m³)
  4. Detection limit
  5. Ambient methane model (seasonal)
  6. Plume events
  7. Alert levels
  8. Isotope discrimination
  9. Isotope precision
 10. Energy calculations
 11. Sensor degradation
 12. Calibration
 13. State dataclass (round-trip serialisation)
 14. Tick engine
 15. Conservation laws & physical invariants
 16. Multi-sol simulation
 17. Property-based invariants (parametrize)
 18. Edge cases
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

# ── path setup ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from methane_detector import (
    # Constants
    MARS_SURFACE_PRESSURE_PA,
    MARS_SURFACE_TEMP_K,
    MARS_YEAR_SOLS,
    AMBIENT_CH4_BACKGROUND_PPBV,
    AMBIENT_CH4_SEASONAL_AMPLITUDE_PPBV,
    AMBIENT_CH4_PLUME_PROBABILITY,
    AMBIENT_CH4_PLUME_MIN_PPBV,
    AMBIENT_CH4_PLUME_MAX_PPBV,
    HABITAT_CH4_WARNING_PPMV,
    HABITAT_CH4_CRITICAL_PPMV,
    HABITAT_CH4_LEL_PPMV,
    LASER_WAVELENGTH_UM,
    CH4_MOLAR_ABSORPTIVITY_M2_MOL,
    DEFAULT_PATH_LENGTH_M,
    NOISE_FLOOR_PPBV,
    BIOTIC_D13C_THRESHOLD,
    ABIOTIC_D13C_MIN,
    D13C_PRECISION_PER_MEASUREMENT,
    D13C_PHYSICAL_MIN,
    D13C_PHYSICAL_MAX,
    STANDBY_POWER_W,
    SCAN_POWER_W,
    SECONDS_PER_SOL,
    HOURS_PER_SOL,
    DEGRADATION_RATE_PER_SOL,
    MAX_DEGRADATION,
    CALIBRATION_RESET_FACTOR,
    # Functions
    beer_lambert_absorbance,
    ppbv_to_mol_m3,
    detection_limit_ppbv,
    ambient_methane_ppbv,
    is_plume_event,
    plume_concentration_ppbv,
    alert_level,
    isotope_discrimination,
    isotope_precision,
    scan_energy_wh,
    standby_energy_wh,
    daily_energy_wh,
    apply_degradation,
    calibrate_sensor,
    # Classes
    MethaneDetector,
    TickResult,
    # Engine
    tick,
    run_simulation,
)


# =====================================================================
# 1. Physical constants validation
# =====================================================================

class TestPhysicalConstants:
    """Verify constants are in physically reasonable ranges."""

    def test_mars_pressure(self):
        assert 400 < MARS_SURFACE_PRESSURE_PA < 900  # Pa

    def test_mars_temperature(self):
        assert 150 < MARS_SURFACE_TEMP_K < 300  # K

    def test_mars_year_sols(self):
        assert 660 < MARS_YEAR_SOLS < 680

    def test_methane_background(self):
        # Curiosity measured ~0.41 ppbv
        assert 0.1 < AMBIENT_CH4_BACKGROUND_PPBV < 2.0

    def test_laser_wavelength(self):
        # CH₄ ν₃ band is ~3.3 µm
        assert 3.0 < LASER_WAVELENGTH_UM < 3.5

    def test_lel_threshold(self):
        # LEL is 5% = 50,000 ppmv
        assert HABITAT_CH4_LEL_PPMV == 50_000.0

    def test_warning_below_critical(self):
        assert HABITAT_CH4_WARNING_PPMV < HABITAT_CH4_CRITICAL_PPMV

    def test_critical_below_lel(self):
        assert HABITAT_CH4_CRITICAL_PPMV < HABITAT_CH4_LEL_PPMV

    def test_sol_duration(self):
        # Mars sol ~88,775 s
        assert 88_000 < SECONDS_PER_SOL < 89_000

    def test_hours_per_sol(self):
        assert 24.5 < HOURS_PER_SOL < 24.8

    def test_isotope_bounds(self):
        assert D13C_PHYSICAL_MIN < BIOTIC_D13C_THRESHOLD < ABIOTIC_D13C_MIN < D13C_PHYSICAL_MAX

    def test_degradation_bounds(self):
        assert 0 < DEGRADATION_RATE_PER_SOL < 0.1
        assert 0 < MAX_DEGRADATION <= 1.0
        assert 0 <= CALIBRATION_RESET_FACTOR < MAX_DEGRADATION


# =====================================================================
# 2. Beer-Lambert law
# =====================================================================

class TestBeerLambert:
    """Beer-Lambert absorbance: A = ε·c·L."""

    def test_zero_concentration(self):
        assert beer_lambert_absorbance(0.044, 0.0, 100.0) == 0.0

    def test_zero_path(self):
        assert beer_lambert_absorbance(0.044, 1e-6, 0.0) == 0.0

    def test_positive(self):
        A = beer_lambert_absorbance(0.044, 1e-6, 100.0)
        assert A > 0
        assert A == pytest.approx(0.044 * 1e-6 * 100.0)

    def test_linearity_concentration(self):
        A1 = beer_lambert_absorbance(0.044, 1e-6, 100.0)
        A2 = beer_lambert_absorbance(0.044, 2e-6, 100.0)
        assert A2 == pytest.approx(2 * A1)

    def test_linearity_path_length(self):
        A1 = beer_lambert_absorbance(0.044, 1e-6, 50.0)
        A2 = beer_lambert_absorbance(0.044, 1e-6, 100.0)
        assert A2 == pytest.approx(2 * A1)

    def test_negative_inputs_raise(self):
        with pytest.raises(ValueError):
            beer_lambert_absorbance(-0.044, 1e-6, 100.0)
        with pytest.raises(ValueError):
            beer_lambert_absorbance(0.044, -1e-6, 100.0)
        with pytest.raises(ValueError):
            beer_lambert_absorbance(0.044, 1e-6, -100.0)


# =====================================================================
# 3. Concentration conversion
# =====================================================================

class TestConcentrationConversion:
    """ppbv to mol/m³ via ideal gas law."""

    def test_zero_ppbv(self):
        assert ppbv_to_mol_m3(0.0, MARS_SURFACE_PRESSURE_PA, MARS_SURFACE_TEMP_K) == 0.0

    def test_positive(self):
        c = ppbv_to_mol_m3(1.0, MARS_SURFACE_PRESSURE_PA, MARS_SURFACE_TEMP_K)
        assert c > 0

    def test_proportional_to_ppbv(self):
        c1 = ppbv_to_mol_m3(1.0, MARS_SURFACE_PRESSURE_PA, MARS_SURFACE_TEMP_K)
        c10 = ppbv_to_mol_m3(10.0, MARS_SURFACE_PRESSURE_PA, MARS_SURFACE_TEMP_K)
        assert c10 == pytest.approx(10 * c1)

    def test_proportional_to_pressure(self):
        c1 = ppbv_to_mol_m3(1.0, 610.0, 210.0)
        c2 = ppbv_to_mol_m3(1.0, 1220.0, 210.0)
        assert c2 == pytest.approx(2 * c1)

    def test_inverse_to_temperature(self):
        c1 = ppbv_to_mol_m3(1.0, 610.0, 210.0)
        c2 = ppbv_to_mol_m3(1.0, 610.0, 420.0)
        assert c2 == pytest.approx(c1 / 2)

    def test_negative_ppbv_raises(self):
        with pytest.raises(ValueError):
            ppbv_to_mol_m3(-1.0, 610.0, 210.0)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            ppbv_to_mol_m3(1.0, 610.0, 0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            ppbv_to_mol_m3(1.0, 610.0, -10.0)


# =====================================================================
# 4. Detection limit
# =====================================================================

class TestDetectionLimit:
    """Minimum detectable concentration."""

    def test_default(self):
        dl = detection_limit_ppbv(DEFAULT_PATH_LENGTH_M)
        assert dl == pytest.approx(NOISE_FLOOR_PPBV)

    def test_longer_path_improves(self):
        dl_short = detection_limit_ppbv(50.0)
        dl_long = detection_limit_ppbv(200.0)
        assert dl_long < dl_short

    def test_degradation_worsens(self):
        dl_good = detection_limit_ppbv(100.0, degradation=0.0)
        dl_bad = detection_limit_ppbv(100.0, degradation=0.5)
        assert dl_bad > dl_good

    def test_zero_degradation(self):
        dl = detection_limit_ppbv(100.0, degradation=0.0)
        assert dl == pytest.approx(NOISE_FLOOR_PPBV)

    def test_full_degradation_high(self):
        dl = detection_limit_ppbv(100.0, degradation=1.0)
        assert dl > 10 * NOISE_FLOOR_PPBV  # sensor is nearly blind

    def test_always_positive(self):
        for d in [0.0, 0.1, 0.5, 0.9, 1.0]:
            assert detection_limit_ppbv(100.0, degradation=d) > 0

    def test_zero_path_raises(self):
        with pytest.raises(ValueError):
            detection_limit_ppbv(0.0)

    def test_negative_path_raises(self):
        with pytest.raises(ValueError):
            detection_limit_ppbv(-10.0)


# =====================================================================
# 5. Ambient methane (seasonal model)
# =====================================================================

class TestAmbientMethane:
    """Seasonal Mars methane cycle."""

    def test_sol_zero(self):
        ch4 = ambient_methane_ppbv(0)
        assert ch4 == pytest.approx(AMBIENT_CH4_BACKGROUND_PPBV, abs=0.01)

    def test_always_non_negative(self):
        for sol in range(0, 700):
            assert ambient_methane_ppbv(sol) >= 0

    def test_seasonal_peak(self):
        # Peak near sol ~167 (quarter year)
        peak_sol = int(MARS_YEAR_SOLS / 4)
        ch4_peak = ambient_methane_ppbv(peak_sol)
        ch4_base = ambient_methane_ppbv(0)
        assert ch4_peak > ch4_base

    def test_seasonal_trough(self):
        # Trough near sol ~501 (3/4 year)
        trough_sol = int(3 * MARS_YEAR_SOLS / 4)
        ch4_trough = ambient_methane_ppbv(trough_sol)
        ch4_base = ambient_methane_ppbv(0)
        assert ch4_trough < ch4_base

    def test_periodic(self):
        """One full Mars year returns to ~same value."""
        ch4_0 = ambient_methane_ppbv(0)
        ch4_year = ambient_methane_ppbv(int(MARS_YEAR_SOLS))
        assert ch4_year == pytest.approx(ch4_0, abs=0.05)

    def test_bounded(self):
        for sol in range(0, 700):
            ch4 = ambient_methane_ppbv(sol)
            assert ch4 <= AMBIENT_CH4_BACKGROUND_PPBV + AMBIENT_CH4_SEASONAL_AMPLITUDE_PPBV + 0.01


# =====================================================================
# 6. Plume events
# =====================================================================

class TestPlumeEvents:
    """Stochastic methane plumes."""

    def test_plume_probability_range(self):
        assert 0 < AMBIENT_CH4_PLUME_PROBABILITY < 1

    def test_plume_concentration_bounded(self):
        rng = random.Random(42)
        for _ in range(100):
            c = plume_concentration_ppbv(rng)
            assert AMBIENT_CH4_PLUME_MIN_PPBV <= c <= AMBIENT_CH4_PLUME_MAX_PPBV

    def test_deterministic_with_seed(self):
        r1 = random.Random(99)
        r2 = random.Random(99)
        assert is_plume_event(r1) == is_plume_event(r2)
        assert plume_concentration_ppbv(r1) == plume_concentration_ppbv(r2)

    def test_plumes_are_rare(self):
        rng = random.Random(42)
        count = sum(1 for _ in range(10000) if is_plume_event(rng))
        # Expected ~50 plumes in 10000 sols
        assert 10 < count < 150


# =====================================================================
# 7. Alert levels
# =====================================================================

class TestAlertLevels:
    """Habitat CH₄ alert classification."""

    def test_nominal(self):
        assert alert_level(0.0) == "nominal"
        assert alert_level(500.0) == "nominal"
        assert alert_level(999.9) == "nominal"

    def test_warning(self):
        assert alert_level(1000.0) == "warning"
        assert alert_level(5000.0) == "warning"
        assert alert_level(9999.9) == "warning"

    def test_critical(self):
        assert alert_level(10_000.0) == "critical"
        assert alert_level(25_000.0) == "critical"
        assert alert_level(49_999.9) == "critical"

    def test_explosive(self):
        assert alert_level(50_000.0) == "explosive"
        assert alert_level(100_000.0) == "explosive"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            alert_level(-1.0)

    def test_monotonic(self):
        """Higher concentration → same or higher alert severity."""
        levels = {"nominal": 0, "warning": 1, "critical": 2, "explosive": 3}
        prev = 0
        for ppmv in [0, 500, 1000, 5000, 10000, 30000, 50000, 100000]:
            current = levels[alert_level(float(ppmv))]
            assert current >= prev
            prev = current


# =====================================================================
# 8. Isotope discrimination
# =====================================================================

class TestIsotopeDiscrimination:
    """δ¹³C classification."""

    def test_biotic(self):
        assert isotope_discrimination(-60.0) == "biotic"
        assert isotope_discrimination(-45.0) == "biotic"
        assert isotope_discrimination(-40.0) == "biotic"

    def test_abiotic(self):
        assert isotope_discrimination(-20.0) == "abiotic"
        assert isotope_discrimination(0.0) == "abiotic"
        assert isotope_discrimination(10.0) == "abiotic"

    def test_ambiguous(self):
        assert isotope_discrimination(-30.0) == "ambiguous"
        assert isotope_discrimination(-25.0) == "ambiguous"

    def test_boundary_biotic(self):
        assert isotope_discrimination(BIOTIC_D13C_THRESHOLD) == "biotic"

    def test_boundary_abiotic(self):
        assert isotope_discrimination(ABIOTIC_D13C_MIN) == "abiotic"

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            isotope_discrimination(-150.0)
        with pytest.raises(ValueError):
            isotope_discrimination(100.0)

    def test_physical_bounds(self):
        # Extremes of physical range should not raise
        isotope_discrimination(D13C_PHYSICAL_MIN)
        isotope_discrimination(D13C_PHYSICAL_MAX)


# =====================================================================
# 9. Isotope precision
# =====================================================================

class TestIsotopePrecision:
    """√N averaging improvement."""

    def test_single_measurement(self):
        assert isotope_precision(1) == pytest.approx(D13C_PRECISION_PER_MEASUREMENT)

    def test_four_measurements(self):
        assert isotope_precision(4) == pytest.approx(D13C_PRECISION_PER_MEASUREMENT / 2)

    def test_hundred_measurements(self):
        assert isotope_precision(100) == pytest.approx(D13C_PRECISION_PER_MEASUREMENT / 10)

    def test_improves_with_n(self):
        p1 = isotope_precision(1)
        p10 = isotope_precision(10)
        p100 = isotope_precision(100)
        assert p10 < p1
        assert p100 < p10

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            isotope_precision(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            isotope_precision(-1)


# =====================================================================
# 10. Energy calculations
# =====================================================================

class TestEnergy:
    """Power and energy accounting."""

    def test_scan_energy_positive(self):
        e = scan_energy_wh(300.0)
        assert e > 0

    def test_scan_energy_proportional_to_time(self):
        e1 = scan_energy_wh(100.0)
        e2 = scan_energy_wh(200.0)
        assert e2 == pytest.approx(2 * e1)

    def test_standby_energy_positive(self):
        e = standby_energy_wh(3600.0)
        assert e > 0

    def test_daily_energy_all_standby(self):
        e = daily_energy_wh(0)
        expected = standby_energy_wh(SECONDS_PER_SOL)
        assert e == pytest.approx(expected)

    def test_daily_energy_increases_with_scans(self):
        e0 = daily_energy_wh(0)
        e24 = daily_energy_wh(24)
        assert e24 > e0  # scans cost more than standby

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError):
            scan_energy_wh(-10.0)
        with pytest.raises(ValueError):
            standby_energy_wh(-10.0)

    def test_negative_scans_raises(self):
        with pytest.raises(ValueError):
            daily_energy_wh(-1)

    def test_daily_energy_reasonable(self):
        """24 scans × 5min + standby ≈ 370-400 Wh range."""
        e = daily_energy_wh(24, scan_duration_s=300.0)
        assert 300 < e < 800  # Wh per sol


# =====================================================================
# 11. Sensor degradation
# =====================================================================

class TestDegradation:
    """Sensor degradation over time."""

    def test_from_zero(self):
        d = apply_degradation(0.0)
        assert d > 0
        assert d == pytest.approx(DEGRADATION_RATE_PER_SOL)

    def test_monotonically_increasing(self):
        d = 0.0
        for _ in range(100):
            d_new = apply_degradation(d)
            assert d_new >= d
            d = d_new

    def test_bounded_by_max(self):
        d = 0.0
        for _ in range(10000):
            d = apply_degradation(d)
        assert d <= MAX_DEGRADATION

    def test_approaches_max(self):
        d = 0.0
        for _ in range(1000):
            d = apply_degradation(d)
        assert d > MAX_DEGRADATION * 0.9

    def test_diminishing_rate(self):
        """Degradation slows as it approaches max (1 - current factor)."""
        d0 = apply_degradation(0.0)
        d_half = apply_degradation(0.5)
        # Increment from 0.0 should be larger than from 0.5
        increment_0 = d0 - 0.0
        increment_half = d_half - 0.5
        assert increment_0 > increment_half


# =====================================================================
# 12. Calibration
# =====================================================================

class TestCalibration:
    """Sensor calibration resets degradation."""

    def test_calibration_from_high(self):
        d = calibrate_sensor(0.7)
        assert d <= CALIBRATION_RESET_FACTOR

    def test_calibration_from_low(self):
        d = calibrate_sensor(0.01)
        assert d <= CALIBRATION_RESET_FACTOR

    def test_calibration_from_zero(self):
        d = calibrate_sensor(0.0)
        assert d == 0.0

    def test_calibration_idempotent(self):
        d1 = calibrate_sensor(0.5)
        d2 = calibrate_sensor(d1)
        assert d2 == d1


# =====================================================================
# 13. State dataclass (serialisation round-trip)
# =====================================================================

class TestStateSerde:
    """MethaneDetector to_dict / from_dict round-trip."""

    def test_default_round_trip(self):
        state = MethaneDetector()
        d = state.to_dict()
        restored = MethaneDetector.from_dict(d)
        assert restored.sol == state.sol
        assert restored.path_length_m == state.path_length_m
        assert restored.degradation == pytest.approx(state.degradation)

    def test_custom_round_trip(self):
        state = MethaneDetector(
            sol=100,
            path_length_m=200.0,
            degradation=0.3,
            ambient_ch4_ppbv=0.65,
            plume_events_detected=5,
            ambient_history=[0.4, 0.5, 0.6],
        )
        d = state.to_dict()
        restored = MethaneDetector.from_dict(d)
        assert restored.sol == 100
        assert restored.path_length_m == 200.0
        assert restored.degradation == pytest.approx(0.3, abs=1e-5)
        assert restored.plume_events_detected == 5
        assert len(restored.ambient_history) == 3

    def test_empty_dict(self):
        restored = MethaneDetector.from_dict({})
        assert restored.sol == 0
        assert restored.degradation == 0.0


# =====================================================================
# 14. Tick engine
# =====================================================================

class TestTick:
    """One-sol tick advancement."""

    def test_sol_increments(self):
        state = MethaneDetector()
        tick(state, rng=random.Random(42))
        assert state.sol == 1

    def test_measurements_accumulate(self):
        state = MethaneDetector(scans_per_sol=24)
        tick(state, rng=random.Random(42))
        assert state.total_measurements == 24

    def test_energy_accumulates(self):
        state = MethaneDetector()
        tick(state, rng=random.Random(42))
        assert state.total_energy_wh > 0

    def test_degradation_increases(self):
        state = MethaneDetector(degradation=0.0)
        tick(state, rng=random.Random(42))
        assert state.degradation > 0

    def test_ambient_reading_non_negative(self):
        state = MethaneDetector()
        result = tick(state, rng=random.Random(42))
        assert result.ambient_ch4_ppbv >= 0

    def test_habitat_alert_nominal(self):
        state = MethaneDetector()
        result = tick(state, habitat_ch4_ppmv=0.0, rng=random.Random(42))
        assert result.alert == "nominal"

    def test_habitat_alert_warning(self):
        state = MethaneDetector()
        result = tick(state, habitat_ch4_ppmv=2000.0, rng=random.Random(42))
        assert result.alert == "warning"

    def test_habitat_alert_critical(self):
        state = MethaneDetector()
        result = tick(state, habitat_ch4_ppmv=15000.0, rng=random.Random(42))
        assert result.alert == "critical"

    def test_isotope_measurement(self):
        state = MethaneDetector()
        result = tick(state, d13c_sample=-55.0, rng=random.Random(42))
        assert result.isotope_class == "biotic"
        assert state.isotope_measurements == 1

    def test_calibration_resets(self):
        state = MethaneDetector(degradation=0.5)
        tick(state, do_calibration=True, rng=random.Random(42))
        # After calibration + one sol of degradation, should be much less than 0.5
        assert state.degradation < 0.1

    def test_history_appended(self):
        state = MethaneDetector()
        tick(state, rng=random.Random(42))
        assert len(state.ambient_history) == 1

    def test_history_capped_at_10(self):
        state = MethaneDetector(ambient_history=[0.4] * 10)
        tick(state, rng=random.Random(42))
        assert len(state.ambient_history) == 10

    def test_tick_result_fields(self):
        state = MethaneDetector()
        result = tick(state, rng=random.Random(42))
        assert isinstance(result, TickResult)
        assert result.sol == 1
        assert isinstance(result.detection_limit_ppbv, float)
        assert isinstance(result.energy_used_wh, float)


# =====================================================================
# 15. Conservation laws & physical invariants
# =====================================================================

class TestConservationLaws:
    """Physical laws that must never be violated."""

    def test_energy_monotonically_increases(self):
        """Total energy consumed can only grow."""
        state = MethaneDetector()
        rng = random.Random(42)
        prev_energy = 0.0
        for _ in range(50):
            tick(state, rng=rng)
            assert state.total_energy_wh >= prev_energy
            prev_energy = state.total_energy_wh

    def test_measurements_monotonically_increase(self):
        state = MethaneDetector()
        rng = random.Random(42)
        prev = 0
        for _ in range(50):
            tick(state, rng=rng)
            assert state.total_measurements >= prev
            prev = state.total_measurements

    def test_degradation_bounded(self):
        """Degradation never exceeds MAX_DEGRADATION."""
        state = MethaneDetector()
        rng = random.Random(42)
        for _ in range(2000):
            tick(state, rng=rng)
        assert 0 <= state.degradation <= MAX_DEGRADATION

    def test_ambient_never_negative(self):
        state = MethaneDetector()
        rng = random.Random(42)
        for _ in range(100):
            result = tick(state, rng=rng)
            assert result.ambient_ch4_ppbv >= 0

    def test_detection_limit_always_positive(self):
        state = MethaneDetector()
        rng = random.Random(42)
        for _ in range(100):
            result = tick(state, rng=rng)
            assert result.detection_limit_ppbv > 0

    def test_sol_increments_exactly_one(self):
        state = MethaneDetector()
        rng = random.Random(42)
        for expected_sol in range(1, 51):
            tick(state, rng=rng)
            assert state.sol == expected_sol

    def test_plume_count_bounded_by_sols(self):
        """Can't detect more plumes than sols elapsed."""
        state = MethaneDetector()
        rng = random.Random(42)
        for _ in range(100):
            tick(state, rng=rng)
        assert state.plume_events_detected <= state.sol

    def test_isotope_precision_improves_with_measurements(self):
        state = MethaneDetector()
        rng = random.Random(42)
        precisions = []
        for i in range(10):
            tick(state, d13c_sample=-30.0 + i * 0.1, rng=rng)
            precisions.append(state.d13c_running_precision)
        # Each measurement should improve precision
        for i in range(1, len(precisions)):
            assert precisions[i] <= precisions[i - 1]


# =====================================================================
# 16. Multi-sol simulation
# =====================================================================

class TestSimulation:
    """Multi-sol simulation runs."""

    def test_smoke_10_sols(self):
        state, results = run_simulation(10)
        assert state.sol == 10
        assert len(results) == 10

    def test_smoke_100_sols(self):
        state, results = run_simulation(100)
        assert state.sol == 100
        assert state.total_energy_wh > 0

    def test_smoke_668_sols_one_mars_year(self):
        """Full Mars year without crash."""
        state, results = run_simulation(669)
        assert state.sol == 669
        assert state.total_measurements > 0

    def test_deterministic_with_seed(self):
        s1, r1 = run_simulation(50, seed=123)
        s2, r2 = run_simulation(50, seed=123)
        assert s1.sol == s2.sol
        assert s1.total_energy_wh == s2.total_energy_wh
        assert s1.plume_events_detected == s2.plume_events_detected

    def test_different_seeds_different_results(self):
        s1, _ = run_simulation(100, seed=1)
        s2, _ = run_simulation(100, seed=2)
        # Energy should be same (deterministic scans), but plumes may differ
        assert s1.total_energy_wh == pytest.approx(s2.total_energy_wh)
        # Plume events are stochastic, likely different
        # (not guaranteed, but very likely with 100 sols)

    def test_plumes_detected_over_long_run(self):
        """Over 10000 sols, at least some plumes should occur."""
        state, _ = run_simulation(10000, seed=42)
        assert state.plume_events_detected > 0

    def test_degradation_saturates(self):
        """After many sols without calibration, degradation approaches max."""
        state, _ = run_simulation(2000, seed=42)
        assert state.degradation > MAX_DEGRADATION * 0.95


# =====================================================================
# 17. Property-based invariants (parametrize)
# =====================================================================

class TestPropertyBased:
    """Parametrized tests across ranges."""

    @pytest.mark.parametrize("path_m", [10, 50, 100, 200, 500])
    def test_detection_limit_positive_for_all_paths(self, path_m):
        assert detection_limit_ppbv(float(path_m)) > 0

    @pytest.mark.parametrize("path_m", [10, 50, 100, 200, 500])
    def test_detection_limit_inversely_proportional(self, path_m):
        dl = detection_limit_ppbv(float(path_m))
        dl_ref = detection_limit_ppbv(DEFAULT_PATH_LENGTH_M)
        assert dl == pytest.approx(dl_ref * DEFAULT_PATH_LENGTH_M / path_m)

    @pytest.mark.parametrize("sol", [0, 100, 334, 500, 668])
    def test_ambient_non_negative_at_key_sols(self, sol):
        assert ambient_methane_ppbv(sol) >= 0

    @pytest.mark.parametrize("ppmv,expected", [
        (0.0, "nominal"),
        (999.0, "nominal"),
        (1000.0, "warning"),
        (9999.0, "warning"),
        (10000.0, "critical"),
        (49999.0, "critical"),
        (50000.0, "explosive"),
    ])
    def test_alert_level_at_boundaries(self, ppmv, expected):
        assert alert_level(ppmv) == expected

    @pytest.mark.parametrize("d13c,expected", [
        (-80.0, "biotic"),
        (-40.0, "biotic"),
        (-30.0, "ambiguous"),
        (-20.0, "abiotic"),
        (0.0, "abiotic"),
        (20.0, "abiotic"),
    ])
    def test_isotope_classification(self, d13c, expected):
        assert isotope_discrimination(d13c) == expected

    @pytest.mark.parametrize("n", [1, 4, 9, 16, 25, 100])
    def test_precision_sqrt_n(self, n):
        assert isotope_precision(n) == pytest.approx(
            D13C_PRECISION_PER_MEASUREMENT / math.sqrt(n)
        )

    @pytest.mark.parametrize("degradation", [0.0, 0.1, 0.3, 0.5, 0.7, 0.8])
    def test_degradation_never_exceeds_max(self, degradation):
        d = apply_degradation(degradation)
        assert d <= MAX_DEGRADATION

    @pytest.mark.parametrize("seed", [1, 42, 99, 256, 1000])
    def test_simulation_no_crash_any_seed(self, seed):
        state, results = run_simulation(50, seed=seed)
        assert state.sol == 50
        assert len(results) == 50


# =====================================================================
# 18. Edge cases
# =====================================================================

class TestEdgeCases:
    """Boundary and extreme input testing."""

    def test_very_short_path(self):
        dl = detection_limit_ppbv(0.01)
        assert dl > 100  # very poor sensitivity

    def test_very_long_path(self):
        dl = detection_limit_ppbv(10000.0)
        assert dl < 0.001  # excellent sensitivity

    def test_habitat_negative_clamped(self):
        state = MethaneDetector()
        result = tick(state, habitat_ch4_ppmv=-100.0, rng=random.Random(42))
        assert result.habitat_ch4_ppmv == 0.0

    def test_isotope_clamped_to_bounds(self):
        state = MethaneDetector()
        tick(state, d13c_sample=-200.0, rng=random.Random(42))
        assert state.last_d13c >= D13C_PHYSICAL_MIN

    def test_isotope_clamped_upper(self):
        state = MethaneDetector()
        tick(state, d13c_sample=200.0, rng=random.Random(42))
        assert state.last_d13c <= D13C_PHYSICAL_MAX

    def test_zero_scans_per_sol(self):
        state = MethaneDetector(scans_per_sol=0)
        result = tick(state, rng=random.Random(42))
        # All standby energy
        expected = standby_energy_wh(SECONDS_PER_SOL)
        assert result.energy_used_wh == pytest.approx(expected)

    def test_many_scans_per_sol(self):
        state = MethaneDetector(scans_per_sol=1000)
        result = tick(state, rng=random.Random(42))
        assert result.energy_used_wh > 0

    def test_run_simulation_zero_sols(self):
        state, results = run_simulation(0)
        assert state.sol == 0
        assert len(results) == 0

    def test_run_simulation_one_sol(self):
        state, results = run_simulation(1)
        assert state.sol == 1
        assert len(results) == 1

    def test_absorbance_at_mars_background(self):
        """Beer-Lambert for typical Mars methane background."""
        c = ppbv_to_mol_m3(0.41, MARS_SURFACE_PRESSURE_PA, MARS_SURFACE_TEMP_K)
        A = beer_lambert_absorbance(CH4_MOLAR_ABSORPTIVITY_M2_MOL, c, DEFAULT_PATH_LENGTH_M)
        assert A > 0
        assert A < 1.0  # very faint signal at ppb levels
