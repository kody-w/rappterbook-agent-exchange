"""Tests for water_ice_radar.py - Mars GPR for Subsurface Ice Detection.

Sections: constants, signal speed, travel time, depth estimation, attenuation,
signal power, reflection, SNR, resolution, detection, max depth, ScanResult,
IceRadar, tick, serialisation, conservation, multi-sol, parametric, edge cases.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from water_ice_radar import (
    SPEED_OF_LIGHT_M_S, REGOLITH_EPSILON_R, REGOLITH_LOSS_TANGENT,
    ICE_EPSILON_R, ICE_LOSS_TANGENT, DEFAULT_CENTER_FREQ_HZ,
    DEFAULT_BANDWIDTH_HZ, DEFAULT_TRANSMIT_POWER_W, NOISE_FLOOR_W,
    SNR_MIN_DB, MAX_DETECTABLE_DEPTH_M, RADAR_POWER_DRAW_W,
    SCANS_PER_SOL, DUST_DEGRADATION_PER_SOL, DUST_CLEANING_RESTORATION,
    SOL_HOURS, signal_speed, two_way_travel_time, estimate_depth,
    attenuation_coefficient, signal_power_at_depth, reflection_coefficient,
    snr_db, depth_resolution, detect_ice_at_depth, max_detection_depth,
    ScanResult, IceRadar,
)


class TestPhysicalConstants:
    def test_speed_of_light(self):
        assert 2.99e8 < SPEED_OF_LIGHT_M_S < 3.01e8

    def test_regolith_epsilon(self):
        assert 1.0 < REGOLITH_EPSILON_R < 10.0

    def test_ice_epsilon(self):
        assert 2.0 < ICE_EPSILON_R < 5.0

    def test_regolith_loss_tangent(self):
        assert 0.0 < REGOLITH_LOSS_TANGENT < 0.2

    def test_ice_loss_tangent(self):
        assert 0.0 <= ICE_LOSS_TANGENT < 0.01

    def test_noise_floor_positive(self):
        assert NOISE_FLOOR_W > 0.0

    def test_snr_threshold(self):
        assert 3.0 <= SNR_MIN_DB <= 20.0

    def test_freq_range(self):
        assert 1e6 <= DEFAULT_CENTER_FREQ_HZ <= 1e9

    def test_sol_hours(self):
        assert 24.0 < SOL_HOURS < 25.0


class TestSignalSpeed:
    def test_vacuum(self):
        assert abs(signal_speed(1.0) - SPEED_OF_LIGHT_M_S) < 1.0

    def test_regolith_slower(self):
        assert signal_speed(REGOLITH_EPSILON_R) < SPEED_OF_LIGHT_M_S

    def test_higher_eps_slower(self):
        assert signal_speed(4.0) < signal_speed(2.0)

    def test_known_value(self):
        expected = SPEED_OF_LIGHT_M_S / math.sqrt(REGOLITH_EPSILON_R)
        assert abs(signal_speed(REGOLITH_EPSILON_R) - expected) < 1.0

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            signal_speed(0.5)

    @pytest.mark.parametrize("eps", [1.0, 2.0, 3.0, 4.0, 5.0, 9.0])
    def test_positive(self, eps):
        assert signal_speed(eps) > 0.0


class TestTwoWayTravelTime:
    def test_zero_depth(self):
        assert two_way_travel_time(0.0, REGOLITH_EPSILON_R) == 0.0

    def test_deeper_longer(self):
        assert two_way_travel_time(100.0, 3.0) > two_way_travel_time(10.0, 3.0)

    def test_known_vacuum(self):
        t = two_way_travel_time(10.0, 1.0)
        assert abs(t - 2.0 * 10.0 / SPEED_OF_LIGHT_M_S) < 1e-12

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            two_way_travel_time(-1.0, 3.0)

    def test_positive_for_positive_depth(self):
        assert two_way_travel_time(50.0, 3.0) > 0.0


class TestEstimateDepth:
    def test_zero(self):
        assert estimate_depth(0.0, 3.0) == 0.0

    def test_round_trip_identity(self):
        for d in [1.0, 10.0, 50.0, 100.0, 200.0]:
            t = two_way_travel_time(d, REGOLITH_EPSILON_R)
            assert abs(estimate_depth(t, REGOLITH_EPSILON_R) - d) < 1e-6

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            estimate_depth(-1e-9, 3.0)

    @pytest.mark.parametrize("depth", [0.1, 1.0, 10.0, 100.0])
    def test_inverse(self, depth):
        t = two_way_travel_time(depth, REGOLITH_EPSILON_R)
        assert abs(estimate_depth(t, REGOLITH_EPSILON_R) - depth) < 1e-6


class TestAttenuationCoefficient:
    def test_zero_loss(self):
        assert attenuation_coefficient(DEFAULT_CENTER_FREQ_HZ, 3.0, 0.0) == 0.0

    def test_positive(self):
        assert attenuation_coefficient(DEFAULT_CENTER_FREQ_HZ, 3.0, 0.05) > 0.0

    def test_higher_freq(self):
        a1 = attenuation_coefficient(10e6, 3.0, 0.05)
        a2 = attenuation_coefficient(100e6, 3.0, 0.05)
        assert a2 > a1

    def test_higher_loss(self):
        a1 = attenuation_coefficient(20e6, 3.0, 0.01)
        a2 = attenuation_coefficient(20e6, 3.0, 0.1)
        assert a2 > a1

    def test_invalid_freq(self):
        with pytest.raises(ValueError):
            attenuation_coefficient(0.0, 3.0, 0.05)

    def test_negative_loss(self):
        with pytest.raises(ValueError):
            attenuation_coefficient(20e6, 3.0, -0.01)

    def test_known_value(self):
        expected = (math.pi * DEFAULT_CENTER_FREQ_HZ * REGOLITH_LOSS_TANGENT
                    * math.sqrt(REGOLITH_EPSILON_R) / SPEED_OF_LIGHT_M_S)
        actual = attenuation_coefficient(DEFAULT_CENTER_FREQ_HZ, 3.0, 0.05)
        assert abs(actual - expected) < 1e-12


class TestSignalPower:
    def test_zero_depth_full(self):
        p = signal_power_at_depth(10.0, 0.0, 20e6, 3.0, 0.05, 1.0)
        assert abs(p - 10.0) < 1e-9

    def test_decreases_with_depth(self):
        p1 = signal_power_at_depth(10.0, 10.0, 20e6, 3.0, 0.05)
        p2 = signal_power_at_depth(10.0, 100.0, 20e6, 3.0, 0.05)
        assert p2 < p1

    def test_non_negative(self):
        assert signal_power_at_depth(10.0, 1000.0, 20e6, 3.0, 0.05) >= 0.0

    def test_efficiency_scales(self):
        full = signal_power_at_depth(10.0, 50.0, 20e6, 3.0, 0.05, 1.0)
        half = signal_power_at_depth(10.0, 50.0, 20e6, 3.0, 0.05, 0.5)
        assert abs(half - full * 0.5) < 1e-12

    def test_zero_tx(self):
        assert signal_power_at_depth(0.0, 50.0, 20e6, 3.0, 0.05) == 0.0

    def test_neg_tx_raises(self):
        with pytest.raises(ValueError):
            signal_power_at_depth(-1.0, 50.0, 20e6, 3.0, 0.05)

    def test_conservation(self):
        for d in [0.0, 1.0, 10.0, 100.0]:
            assert signal_power_at_depth(10.0, d, 20e6, 3.0, 0.05) <= 10.0 + 1e-12


class TestReflectionCoefficient:
    def test_same_medium(self):
        assert abs(reflection_coefficient(3.0, 3.0)) < 1e-12

    def test_different_positive(self):
        assert reflection_coefficient(REGOLITH_EPSILON_R, ICE_EPSILON_R) > 0.0

    def test_bounded(self):
        r = reflection_coefficient(1.0, 10.0)
        assert 0.0 <= r <= 1.0

    def test_symmetric(self):
        assert abs(reflection_coefficient(2.0, 5.0) - reflection_coefficient(5.0, 2.0)) < 1e-12

    def test_larger_contrast(self):
        assert reflection_coefficient(3.0, 9.0) > reflection_coefficient(3.0, 3.15)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            reflection_coefficient(0.5, 3.0)

    def test_known_value(self):
        n1, n2 = math.sqrt(3.0), math.sqrt(3.15)
        expected = ((n1 - n2) / (n1 + n2)) ** 2
        assert abs(reflection_coefficient(3.0, 3.15) - expected) < 1e-12


class TestSNR:
    def test_equal_zero_db(self):
        assert abs(snr_db(1e-10, 1e-10)) < 1e-9

    def test_ten_times(self):
        assert abs(snr_db(1e-9, 1e-10) - 10.0) < 1e-6

    def test_hundred_times(self):
        assert abs(snr_db(1e-8, 1e-10) - 20.0) < 1e-6

    def test_zero_signal(self):
        assert snr_db(0.0, 1e-10) == -math.inf

    def test_neg_noise_raises(self):
        with pytest.raises(ValueError):
            snr_db(1e-10, 0.0)

    def test_higher_signal(self):
        assert snr_db(1e-8, NOISE_FLOOR_W) > snr_db(1e-10, NOISE_FLOOR_W)


class TestDepthResolution:
    def test_positive(self):
        assert depth_resolution(DEFAULT_BANDWIDTH_HZ, 3.0) > 0.0

    def test_wider_bw_finer(self):
        assert depth_resolution(20e6, 3.0) < depth_resolution(5e6, 3.0)

    def test_known_vacuum(self):
        expected = SPEED_OF_LIGHT_M_S / (2.0 * 10e6)
        assert abs(depth_resolution(10e6, 1.0) - expected) < 0.01

    def test_invalid_bw(self):
        with pytest.raises(ValueError):
            depth_resolution(0.0, 3.0)

    def test_default_system(self):
        r = depth_resolution(DEFAULT_BANDWIDTH_HZ, REGOLITH_EPSILON_R)
        assert 5.0 < r < 20.0


class TestDetectIce:
    def test_shallow_detected(self):
        r = detect_ice_at_depth(5.0, 2.0)
        assert r["detected"] is True
        assert r["snr_db"] > SNR_MIN_DB

    def test_deep_not_detected(self):
        r = detect_ice_at_depth(500.0, 2.0)
        assert r["detected"] is False

    def test_depth_estimate_accurate(self):
        r = detect_ice_at_depth(20.0, 5.0)
        assert abs(r["estimated_depth_m"] - 20.0) < 0.01

    def test_required_keys(self):
        r = detect_ice_at_depth(10.0, 3.0)
        assert {"detected", "snr_db", "estimated_depth_m",
                "ice_thickness_m", "signal_power_w",
                "reflection_coefficient"}.issubset(r.keys())

    def test_signal_power_positive(self):
        assert detect_ice_at_depth(10.0, 3.0)["signal_power_w"] > 0.0

    def test_reflection_bounded(self):
        assert 0.0 <= detect_ice_at_depth(10.0, 3.0)["reflection_coefficient"] <= 1.0

    def test_negative_depth_clamped(self):
        assert detect_ice_at_depth(-5.0, 2.0)["estimated_depth_m"] >= 0.0

    def test_snr_monotonic(self):
        snrs = [detect_ice_at_depth(d, 2.0)["snr_db"]
                for d in [5, 10, 20, 50, 100]]
        for i in range(len(snrs) - 1):
            assert snrs[i] >= snrs[i + 1]


class TestMaxDetectionDepth:
    def test_positive(self):
        assert max_detection_depth() > 0.0

    def test_reasonable_range(self):
        d = max_detection_depth()
        assert 50.0 < d < 500.0

    def test_more_power_deeper(self):
        assert max_detection_depth(50.0) > max_detection_depth(5.0)

    def test_degraded_antenna_shorter(self):
        assert (max_detection_depth(antenna_efficiency=0.5)
                < max_detection_depth(antenna_efficiency=1.0))

    def test_at_max_snr_near_threshold(self):
        d = max_detection_depth()
        r = detect_ice_at_depth(d, 2.0)
        assert abs(r["snr_db"] - SNR_MIN_DB) < 2.0


class TestScanResultClass:
    def test_create(self):
        s = ScanResult(10.0, 20.0, True, 5.0, 2.0, 25.0, 1)
        assert s.ice_detected is True

    def test_to_dict(self):
        d = ScanResult(0.0, 0.0, True, 5.0, 2.0, 25.0, 1).to_dict()
        assert d["ice_detected"] is True

    def test_from_dict(self):
        d = {"x_m": 1.0, "y_m": 2.0, "ice_detected": False,
             "ice_depth_m": 0.0, "ice_thickness_m": 0.0,
             "snr_db": 5.0, "sol": 3}
        assert ScanResult.from_dict(d).x_m == 1.0

    def test_round_trip(self):
        s = ScanResult(3.14, 2.71, True, 12.5, 4.0, 18.3, 7)
        s2 = ScanResult.from_dict(s.to_dict())
        assert s2.ice_detected == s.ice_detected


class TestIceRadarClass:
    def test_default(self):
        r = IceRadar()
        assert r.sol == 0
        assert r.antenna_efficiency == 1.0
        assert r.total_scans == 0

    def test_scan_shallow(self):
        r = IceRadar()
        assert r.scan(0.0, 0.0, 5.0, 2.0).ice_detected is True
        assert r.total_scans == 1

    def test_scan_deep(self):
        assert IceRadar().scan(0.0, 0.0, 500.0, 2.0).ice_detected is False

    def test_multiple_scans(self):
        r = IceRadar()
        r.scan(0.0, 0.0, 5.0, 2.0)
        r.scan(10.0, 0.0, 10.0, 3.0)
        r.scan(20.0, 0.0, 500.0, 1.0)
        assert r.total_scans == 3

    def test_deposit_count(self):
        r = IceRadar()
        r.scan(0.0, 0.0, 5.0, 2.0)       # detected
        r.scan(10.0, 0.0, 5.0, 2.0)       # detected, diff position
        r.scan(0.0, 0.0, 5.0, 2.0)        # detected, same position
        r.scan(20.0, 0.0, 500.0, 2.0)     # not detected
        assert r.ice_deposit_count() == 2

    def test_energy(self):
        r = IceRadar()
        expected = RADAR_POWER_DRAW_W * SOL_HOURS / 1000.0
        assert abs(r.energy_per_sol_kwh() - expected) < 0.001

    def test_detection_range(self):
        assert IceRadar().detection_range_m() > 0.0

    def test_resolution(self):
        assert IceRadar().resolution_m() > 0.0

    def test_clamp_neg_power(self):
        assert IceRadar(transmit_power_w=-5.0).transmit_power_w == 0.0

    def test_clamp_efficiency(self):
        assert IceRadar(antenna_efficiency=1.5).antenna_efficiency == 1.0
        assert IceRadar(antenna_efficiency=-0.5).antenna_efficiency == 0.0


class TestTick:
    def test_advances_sol(self):
        r = IceRadar()
        r.tick()
        assert r.sol == 1

    def test_degrades_antenna(self):
        r = IceRadar()
        before = r.antenna_efficiency
        r.tick()
        assert r.antenna_efficiency < before

    def test_returns_summary(self):
        keys = {"sol", "antenna_efficiency", "detection_range_m",
                "resolution_m", "total_scans", "ice_deposits_found",
                "energy_kwh"}
        assert keys.issubset(IceRadar().tick().keys())

    def test_floor_at_zero(self):
        r = IceRadar(antenna_efficiency=0.0005)
        r.tick()
        assert r.antenna_efficiency >= 0.0

    def test_monotonic_degradation(self):
        r = IceRadar()
        effs = [r.antenna_efficiency]
        for _ in range(10):
            r.tick()
            effs.append(r.antenna_efficiency)
        for i in range(len(effs) - 1):
            assert effs[i] >= effs[i + 1]

    def test_clean_restores(self):
        r = IceRadar()
        for _ in range(100):
            r.tick()
        dirty = r.antenna_efficiency
        r.clean_antenna()
        assert r.antenna_efficiency > dirty

    def test_clean_capped(self):
        r = IceRadar()
        r.clean_antenna()
        assert r.antenna_efficiency <= 1.0


class TestSerialisation:
    def test_empty_round_trip(self):
        r = IceRadar()
        r2 = IceRadar.from_dict(r.to_dict())
        assert r2.sol == r.sol

    def test_with_scans(self):
        r = IceRadar()
        r.scan(0.0, 0.0, 5.0, 2.0)
        r.scan(10.0, 10.0, 20.0, 5.0)
        r.tick()
        r2 = IceRadar.from_dict(r.to_dict())
        assert r2.total_scans == 2 and r2.sol == 1

    def test_json_safe(self):
        r = IceRadar()
        r.scan(5.0, 5.0, 10.0, 3.0)
        r.tick()
        assert json.loads(json.dumps(r.to_dict()))["sol"] == 1

    def test_defaults(self):
        r = IceRadar.from_dict({})
        assert r.transmit_power_w == DEFAULT_TRANSMIT_POWER_W


class TestConservationLaws:
    def test_received_leq_transmitted(self):
        for d in [0, 1, 10, 50, 100, 200]:
            p = signal_power_at_depth(
                DEFAULT_TRANSMIT_POWER_W, d, 20e6, 3.0, 0.05)
            assert p <= DEFAULT_TRANSMIT_POWER_W + 1e-12

    def test_snr_monotonic(self):
        snrs = [detect_ice_at_depth(d, 2.0)["snr_db"]
                for d in [1, 5, 10, 20, 50, 100, 150]]
        for i in range(len(snrs) - 1):
            assert snrs[i] >= snrs[i + 1]

    def test_depth_non_negative(self):
        for d in [0, 1, 10, 100]:
            assert detect_ice_at_depth(d, 2.0)["estimated_depth_m"] >= 0.0

    def test_antenna_bounded(self):
        r = IceRadar()
        for _ in range(2000):
            r.tick()
        assert 0.0 <= r.antenna_efficiency <= 1.0

    def test_energy_positive(self):
        r = IceRadar()
        for _ in range(10):
            assert r.tick()["energy_kwh"] >= 0.0

    def test_reflection_symmetric(self):
        for a, b in [(1.0, 3.0), (2.5, 4.0), (3.0, 3.15), (1.5, 8.0)]:
            assert abs(reflection_coefficient(a, b)
                       - reflection_coefficient(b, a)) < 1e-12


class TestMultiSol:
    def test_100_sol(self):
        r = IceRadar()
        for sol in range(100):
            r.tick()
            if sol % 10 == 0:
                r.scan(float(sol), 0.0, 15.0 + sol * 0.1, 3.0)
        assert r.sol == 100 and r.total_scans == 10

    def test_500_sol_with_cleaning(self):
        r = IceRadar()
        for sol in range(500):
            r.tick()
            if sol % 5 == 0:
                r.scan(float(sol), float(sol), 10.0, 2.0)
            if sol % 100 == 99:
                r.clean_antenna()
        assert r.sol == 500 and r.antenna_efficiency > 0.0

    def test_range_decreases(self):
        r = IceRadar()
        initial = r.detection_range_m()
        for _ in range(200):
            r.tick()
        assert r.detection_range_m() < initial


class TestParametric:
    @pytest.mark.parametrize("depth",
                             [0.0, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0])
    def test_detection_valid(self, depth):
        r = detect_ice_at_depth(depth, 2.0)
        assert isinstance(r["detected"], bool)
        assert r["signal_power_w"] >= 0.0

    @pytest.mark.parametrize("power",
                             [0.1, 1.0, 5.0, 10.0, 50.0, 100.0])
    def test_power_scaling(self, power):
        r1 = detect_ice_at_depth(50.0, 2.0, transmit_power_w=power)
        r2 = detect_ice_at_depth(50.0, 2.0, transmit_power_w=power * 10)
        assert r2["snr_db"] > r1["snr_db"]

    @pytest.mark.parametrize("eps", [1.5, 2.0, 3.0, 4.0, 5.0, 8.0])
    def test_speed_valid(self, eps):
        assert 0 < signal_speed(eps) <= SPEED_OF_LIGHT_M_S

    @pytest.mark.parametrize("bw_mhz", [1.0, 5.0, 10.0, 20.0, 50.0])
    def test_resolution(self, bw_mhz):
        assert depth_resolution(bw_mhz * 1e6, 3.0) > 0.0

    @pytest.mark.parametrize("eff", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_eff_range(self, eff):
        assert max_detection_depth(antenna_efficiency=eff) > 0.0


class TestEdgeCases:
    def test_zero_thickness(self):
        assert isinstance(
            detect_ice_at_depth(10.0, 0.0)["detected"], bool)

    def test_zero_power_radar(self):
        r = IceRadar(transmit_power_w=0.0)
        assert r.scan(0, 0, 5.0, 2.0).ice_detected is False

    def test_fully_degraded(self):
        r = IceRadar(antenna_efficiency=0.0)
        assert r.scan(0, 0, 5.0, 2.0).ice_detected is False

    def test_very_shallow(self):
        assert detect_ice_at_depth(0.01, 0.5)["detected"] is True

    def test_zero_depth(self):
        assert detect_ice_at_depth(0.0, 1.0)["estimated_depth_m"] == 0.0

    def test_sol_tracked(self):
        r = IceRadar()
        for _ in range(5):
            r.tick()
        assert r.scan(0, 0, 10.0, 2.0).sol == 5

    def test_from_dict_scans(self):
        r = IceRadar()
        r.scan(0, 0, 5.0, 2.0)
        r.scan(1, 1, 10.0, 3.0)
        assert IceRadar.from_dict(r.to_dict()).total_scans == 2

    def test_speed_exactly_one(self):
        assert signal_speed(1.0) == SPEED_OF_LIGHT_M_S

    @pytest.mark.parametrize("d",
                             [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    def test_travel_precision(self, d):
        t = two_way_travel_time(d, REGOLITH_EPSILON_R)
        assert abs(estimate_depth(t, REGOLITH_EPSILON_R) - d) < 1e-9
