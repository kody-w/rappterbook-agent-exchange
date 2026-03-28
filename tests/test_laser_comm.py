"""Tests for laser_comm.py -- Mars Deep Space Optical Communications Terminal.

120 tests covering beam physics, link budget, photon counting, DSOC-
calibrated throughput, orbital geometry, dust extinction, solar exclusion,
conservation laws, physical bounds, serialisation, and full-year smoke tests.
"""
from __future__ import annotations

import math
import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import laser_comm as lc


@pytest.fixture
def terminal():
    return lc.LaserTerminal()

@pytest.fixture
def opposition_sol():
    """Sol 0 is opposition (closest approach, SEP=180 deg)."""
    return 0

@pytest.fixture
def conjunction_sol():
    """Sol ~390 is conjunction (farthest, SEP~0 deg, solar exclusion)."""
    return int(lc.SYNODIC_PERIOD_SOLS / 2)


class TestConstants:
    def test_photon_energy_positive(self):
        assert lc.PHOTON_ENERGY_J > 0

    def test_photon_energy_near_infrared(self):
        assert 1.0e-19 < lc.PHOTON_ENERGY_J < 2.0e-19

    def test_beam_divergence_microradians(self):
        urad = lc.BEAM_DIVERGENCE_RAD * 1e6
        assert 5.0 < urad < 15.0

    def test_wavelength_consistency(self):
        assert abs(lc.WAVELENGTH_M * 1e9 - lc.WAVELENGTH_NM) < 0.01

    def test_speed_of_light_consistency(self):
        assert abs(lc.SPEED_OF_LIGHT_KM_S * 1000 - lc.SPEED_OF_LIGHT_M_S) < 1.0

    def test_solar_exclusion_positive(self):
        assert lc.SOLAR_EXCLUSION_DEG > 0

    def test_acquisition_time_reasonable(self):
        assert 10.0 <= lc.ACQUISITION_TIME_S <= 120.0

    def test_dsoc_ref_rate_plausible(self):
        assert 100.0 < lc.DSOC_REF_RATE_MBPS < 500.0

    def test_dsoc_ref_distance_plausible(self):
        assert 20e6 < lc.DSOC_REF_DISTANCE_KM < 50e6


class TestOrbitalGeometry:
    def test_distance_always_positive(self):
        for sol in range(0, 800, 10):
            assert lc.earth_mars_distance_km(sol) > 0

    def test_closest_approach(self):
        distances = [lc.earth_mars_distance_km(s) for s in range(800)]
        closest = min(distances)
        assert 60e6 < closest < 100e6

    def test_farthest_approach(self):
        distances = [lc.earth_mars_distance_km(s) for s in range(800)]
        farthest = max(distances)
        assert 350e6 < farthest < 420e6

    def test_distance_periodic(self):
        d0 = lc.earth_mars_distance_km(0)
        d_cycle = lc.earth_mars_distance_km(int(lc.SYNODIC_PERIOD_SOLS))
        assert abs(d0 - d_cycle) / d0 < 0.01

    def test_sep_angle_range(self):
        for sol in range(800):
            a = lc.sun_earth_probe_angle_deg(sol)
            assert 0.0 <= a <= 180.0

    def test_sep_angle_periodic(self):
        a0 = lc.sun_earth_probe_angle_deg(0)
        a_cycle = lc.sun_earth_probe_angle_deg(int(lc.SYNODIC_PERIOD_SOLS))
        assert abs(a0 - a_cycle) < 2.0  # within 2 degrees over one synodic period

    def test_light_delay_positive(self):
        assert lc.light_delay_seconds(100e6) > 0

    def test_light_delay_scales(self):
        d1 = lc.light_delay_seconds(100e6)
        d2 = lc.light_delay_seconds(200e6)
        assert abs(d2 / d1 - 2.0) < 0.001

    def test_light_delay_at_1au(self):
        delay = lc.light_delay_seconds(lc.AU_KM)
        assert 490 < delay < 510

    def test_solar_exclusion_at_conjunction(self):
        conj_sol = int(lc.SYNODIC_PERIOD_SOLS / 2)
        assert lc.is_solar_exclusion(conj_sol)

    def test_no_exclusion_at_opposition(self):
        assert not lc.is_solar_exclusion(0)


class TestBeamPhysics:
    def test_beam_diameter_increases_with_distance(self):
        d1 = lc.beam_diameter_at_target_m(100e6)
        d2 = lc.beam_diameter_at_target_m(200e6)
        assert d2 > d1

    def test_beam_diameter_linear_with_distance(self):
        d1 = lc.beam_diameter_at_target_m(100e6)
        d2 = lc.beam_diameter_at_target_m(200e6)
        assert abs(d2 / d1 - 2.0) < 0.01

    def test_beam_diameter_at_1_au(self):
        d_m = lc.beam_diameter_at_target_m(lc.AU_KM)
        d_km = d_m / 1000.0
        assert 1000 < d_km < 5000

    def test_beam_diameter_positive(self):
        assert lc.beam_diameter_at_target_m(1.0) > 0

    def test_beam_diameter_zero_at_zero(self):
        assert lc.beam_diameter_at_target_m(0.0) == 0.0

    def test_tx_gain_positive(self):
        assert lc.transmitter_gain_db() > 0

    def test_tx_gain_plausible(self):
        g = lc.transmitter_gain_db()
        assert 100 < g < 120

    def test_rx_gain_higher_than_tx(self):
        assert lc.receiver_gain_db() > lc.transmitter_gain_db()

    def test_rx_gain_plausible(self):
        g = lc.receiver_gain_db()
        assert 125 < g < 145

    def test_free_space_loss_increases_with_distance(self):
        l1 = lc.free_space_loss_db(100e6)
        l2 = lc.free_space_loss_db(200e6)
        assert l2 > l1

    def test_free_space_loss_6db_per_doubling(self):
        l1 = lc.free_space_loss_db(100e6)
        l2 = lc.free_space_loss_db(200e6)
        assert abs((l2 - l1) - 6.02) < 0.1

    def test_free_space_loss_zero_at_zero_distance(self):
        assert lc.free_space_loss_db(0.0) == 0.0


class TestAtmosphericLoss:
    def test_zero_for_zero_tau(self):
        assert lc.mars_atmosphere_loss_db(0.0, 30.0) == 0.0

    def test_increases_with_tau(self):
        l1 = lc.mars_atmosphere_loss_db(0.5, 30.0)
        l2 = lc.mars_atmosphere_loss_db(2.0, 30.0)
        assert l2 > l1

    def test_increases_with_zenith_angle(self):
        l1 = lc.mars_atmosphere_loss_db(0.5, 0.0)
        l2 = lc.mars_atmosphere_loss_db(0.5, 60.0)
        assert l2 > l1

    def test_positive_for_nonzero_tau(self):
        assert lc.mars_atmosphere_loss_db(0.5, 30.0) > 0

    def test_bounded_at_high_zenith(self):
        l = lc.mars_atmosphere_loss_db(0.5, 89.0)
        assert l < 1000.0

    def test_dust_storm_heavy_loss(self):
        l = lc.mars_atmosphere_loss_db(4.0, 30.0)
        assert l > 15.0

    def test_transmission_one_for_zero_tau(self):
        assert lc.mars_atmosphere_transmission(0.0, 30.0) == 1.0

    def test_transmission_decreases_with_tau(self):
        t1 = lc.mars_atmosphere_transmission(0.5, 30.0)
        t2 = lc.mars_atmosphere_transmission(2.0, 30.0)
        assert t2 < t1

    def test_transmission_bounded(self):
        for tau in [0.3, 0.5, 1.0, 2.0, 4.0]:
            t = lc.mars_atmosphere_transmission(tau, 30.0)
            assert 0.0 < t <= 1.0

    def test_pointing_loss_zero_for_perfect(self):
        assert lc.pointing_loss_db(0.0) == 0.0

    def test_pointing_loss_increases_with_error(self):
        l1 = lc.pointing_loss_db(0.5)
        l2 = lc.pointing_loss_db(2.0)
        assert l2 > l1

    def test_pointing_loss_fraction_at_zero(self):
        assert lc.pointing_loss_fraction(0.0) == 1.0

    def test_pointing_loss_fraction_bounded(self):
        for e in [0.0, 0.5, 1.0, 2.0, 5.0]:
            f = lc.pointing_loss_fraction(e)
            assert 0.0 <= f <= 1.0


class TestLinkBudget:
    def test_has_all_terms(self):
        budget = lc.link_budget_db(100e6)
        required = ["tx_power_dbw", "tx_gain_db", "free_space_loss_db",
                     "rx_gain_db", "received_power_dbw"]
        for key in required:
            assert key in budget

    def test_received_power_decreases_with_distance(self):
        p1 = lc.received_power_w(100e6)
        p2 = lc.received_power_w(300e6)
        assert p2 < p1

    def test_received_power_positive(self):
        assert lc.received_power_w(100e6) > 0

    def test_received_power_less_than_transmitted(self):
        assert lc.received_power_w(55e6) < lc.TX_POWER_W

    def test_dust_reduces_received_power(self):
        p_clear = lc.received_power_w(100e6, dust_tau=0.3)
        p_dusty = lc.received_power_w(100e6, dust_tau=2.0)
        assert p_dusty < p_clear

    def test_received_power_inverse_square(self):
        p1 = lc.received_power_w(100e6, dust_tau=0.0, zenith_deg=0.0)
        p2 = lc.received_power_w(200e6, dust_tau=0.0, zenith_deg=0.0)
        ratio = p1 / p2
        assert abs(ratio - 4.0) < 0.5


class TestPhotonCounting:
    def test_photon_rate_positive_at_opposition(self):
        rate = lc.received_photons_per_second(78.3e6)
        assert rate > 0

    def test_photon_rate_decreases_with_distance(self):
        r1 = lc.received_photons_per_second(100e6)
        r2 = lc.received_photons_per_second(300e6)
        assert r2 < r1

    def test_photon_rate_plausible(self):
        rate = lc.received_photons_per_second(lc.AU_KM, dust_tau=0.3)
        assert rate > 1e3

    def test_channel_capacity_positive(self):
        cap = lc.channel_capacity_mbps(1e6)
        assert cap > 0

    def test_channel_capacity_zero_for_zero(self):
        assert lc.channel_capacity_mbps(0.0) == 0.0

    def test_channel_capacity_increases(self):
        c1 = lc.channel_capacity_mbps(1e6)
        c2 = lc.channel_capacity_mbps(1e9)
        assert c2 > c1


class TestThroughput:
    def test_positive_at_opposition(self):
        thr = lc.achievable_throughput_mbps(78e6, dust_tau=0.3)
        assert thr > 1.0

    def test_zero_for_zero_distance(self):
        assert lc.achievable_throughput_mbps(0.0) == 0.0

    def test_decreases_with_distance(self):
        t1 = lc.achievable_throughput_mbps(78e6, dust_tau=0.5)
        t2 = lc.achievable_throughput_mbps(300e6, dust_tau=0.5)
        assert t2 < t1

    def test_decreases_with_dust(self):
        t1 = lc.achievable_throughput_mbps(100e6, dust_tau=0.3)
        t2 = lc.achievable_throughput_mbps(100e6, dust_tau=2.0)
        assert t2 < t1

    def test_dsoc_calibration_at_ref_distance(self):
        thr = lc.achievable_throughput_mbps(
            lc.DSOC_REF_DISTANCE_KM, dust_tau=0.0, zenith_deg=0.0)
        pointing = lc.pointing_loss_fraction(lc.TX_POINTING_ACCURACY_URAD)
        expected = lc.DSOC_REF_RATE_MBPS * pointing * lc.EARTH_ATMO_TRANSMISSION
        assert abs(thr - expected) / expected < 0.01

    def test_inverse_square_scaling(self):
        t1 = lc.achievable_throughput_mbps(100e6, dust_tau=0.0, zenith_deg=0.0)
        t2 = lc.achievable_throughput_mbps(200e6, dust_tau=0.0, zenith_deg=0.0)
        assert abs(t1 / t2 - 4.0) < 0.5

    def test_faster_than_rf_at_close_range(self):
        thr = lc.achievable_throughput_mbps(78e6, dust_tau=0.3)
        assert thr > lc.RF_BASELINE_MBPS


class TestSessionVolume:
    def test_positive_for_active_link(self):
        vol = lc.session_data_volume_gb(100.0)
        assert vol > 0

    def test_zero_for_zero_throughput(self):
        assert lc.session_data_volume_gb(0.0) == 0.0

    def test_scales_with_throughput(self):
        v1 = lc.session_data_volume_gb(50.0)
        v2 = lc.session_data_volume_gb(100.0)
        assert abs(v2 / v1 - 2.0) < 0.01

    def test_scales_with_session_hours(self):
        v1 = lc.session_data_volume_gb(100.0, session_hours=3.0)
        v2 = lc.session_data_volume_gb(100.0, session_hours=6.0)
        assert v2 > v1

    def test_zero_for_zero_hours(self):
        assert lc.session_data_volume_gb(100.0, session_hours=0.0) == 0.0


class TestUpgradeFactor:
    def test_positive_for_positive_laser(self):
        assert lc.upgrade_factor(100.0) > 0

    def test_zero_for_zero_laser(self):
        assert lc.upgrade_factor(0.0) == 0.0

    def test_100x_at_200_mbps(self):
        assert lc.upgrade_factor(200.0) == 100.0

    def test_1x_at_rf_baseline(self):
        assert lc.upgrade_factor(lc.RF_BASELINE_MBPS) == 1.0


class TestDustModel:
    def test_always_positive(self):
        for sol in range(700):
            assert lc.seasonal_dust_tau(sol) > 0

    def test_bounded(self):
        for sol in range(700):
            tau = lc.seasonal_dust_tau(sol)
            assert lc.CLEAR_DUST_TAU <= tau <= 2.0

    def test_seasonal_variation(self):
        taus = [lc.seasonal_dust_tau(s) for s in range(669)]
        assert max(taus) > min(taus)

    def test_periodic(self):
        t0 = lc.seasonal_dust_tau(0)
        t_year = lc.seasonal_dust_tau(669)
        assert abs(t0 - t_year) < 0.05


class TestTerminalState:
    def test_initial_state(self, terminal):
        assert terminal.sol == 0
        assert terminal.total_data_gb == 0.0
        assert terminal.total_sessions == 0

    def test_status_idle_default(self, terminal):
        terminal.link_active = False
        terminal.dust_tau = 0.5
        terminal.sol = 100
        assert terminal.status() == "idle"

    def test_status_transmitting(self, terminal):
        terminal.link_active = True
        terminal.dust_tau = 0.5
        terminal.sol = 100
        assert terminal.status() == "transmitting"

    def test_status_dust_storm(self, terminal):
        terminal.dust_tau = 4.0
        terminal.sol = 100
        assert terminal.status() == "dust_storm"

    def test_status_solar_exclusion(self, terminal):
        terminal.sol = int(lc.SYNODIC_PERIOD_SOLS / 2)  # conjunction
        assert terminal.status() == "solar_exclusion"


class TestTick:
    def test_returns_tick_result(self, terminal):
        result = lc.tick(terminal, sol=200)
        assert isinstance(result, lc.TickResult)

    def test_advances_sol(self, terminal):
        lc.tick(terminal, sol=100)
        assert terminal.sol == 101

    def test_link_active_at_opposition(self, terminal, opposition_sol):
        result = lc.tick(terminal, sol=opposition_sol, dust_tau=0.3)
        assert result.link_active is True

    def test_link_blocked_at_conjunction(self, terminal, conjunction_sol):
        result = lc.tick(terminal, sol=conjunction_sol)
        assert result.link_active is False
        assert result.status == "solar_exclusion"

    def test_link_blocked_in_dust_storm(self, terminal):
        result = lc.tick(terminal, sol=200, dust_tau=4.0)
        assert result.link_active is False
        assert result.status == "dust_storm"

    def test_throughput_positive_when_active(self, terminal, opposition_sol):
        result = lc.tick(terminal, sol=opposition_sol, dust_tau=0.3)
        assert result.throughput_mbps > 0

    def test_throughput_zero_when_blocked(self, terminal, conjunction_sol):
        result = lc.tick(terminal, sol=conjunction_sol)
        assert result.throughput_mbps == 0.0

    def test_data_volume_accumulates(self, terminal):
        lc.tick(terminal, sol=200, dust_tau=0.3)
        lc.tick(terminal, sol=201, dust_tau=0.3)
        assert terminal.total_data_gb > 0

    def test_sessions_count(self, terminal):
        lc.tick(terminal, sol=200, dust_tau=0.3)
        lc.tick(terminal, sol=201, dust_tau=0.3)
        assert terminal.total_sessions == 2

    def test_blocked_sols_counted(self, terminal, conjunction_sol):
        lc.tick(terminal, sol=conjunction_sol)
        assert terminal.total_blocked_sols >= 1

    def test_storm_sols_counted(self, terminal):
        lc.tick(terminal, sol=200, dust_tau=4.0)
        assert terminal.total_storm_sols == 1

    def test_peak_throughput_tracked(self, terminal, opposition_sol):
        lc.tick(terminal, sol=opposition_sol, dust_tau=0.3)
        assert terminal.peak_throughput_mbps > 0

    def test_beam_diameter_in_result(self, terminal):
        result = lc.tick(terminal, sol=200)
        assert result.beam_diameter_earth_km > 0

    def test_rf_upgrade_factor_in_result(self, terminal, opposition_sol):
        result = lc.tick(terminal, sol=opposition_sol, dust_tau=0.3)
        assert result.rf_upgrade_factor > 0

    def test_light_delay_in_result(self, terminal):
        result = lc.tick(terminal, sol=200)
        assert result.light_delay_s > 0

    def test_photon_rate_when_active(self, terminal, opposition_sol):
        result = lc.tick(terminal, sol=opposition_sol, dust_tau=0.3)
        assert result.photon_rate_hz > 0

    def test_capacity_when_active(self, terminal, opposition_sol):
        result = lc.tick(terminal, sol=opposition_sol, dust_tau=0.3)
        assert result.capacity_mbps > 0


class TestConservationLaws:
    def test_received_power_never_exceeds_transmitted(self):
        for sol in range(0, 800, 20):
            d = lc.earth_mars_distance_km(sol)
            if d > 0:
                pr = lc.received_power_w(d, dust_tau=0.0, zenith_deg=0.0)
                assert pr <= lc.TX_POWER_W

    def test_photon_rate_bounded_by_transmit_rate(self):
        tx_photon_rate = lc.TX_POWER_W / lc.PHOTON_ENERGY_J
        for sol in range(0, 800, 50):
            d = lc.earth_mars_distance_km(sol)
            if d > 0:
                rx_rate = lc.received_photons_per_second(d)
                assert rx_rate <= tx_photon_rate

    def test_data_volume_monotonically_increases(self):
        terminal = lc.LaserTerminal()
        prev_gb = 0.0
        for s in range(200):
            lc.tick(terminal, sol=s, dust_tau=0.5)
            assert terminal.total_data_gb >= prev_gb
            prev_gb = terminal.total_data_gb

    def test_free_space_loss_matches_inverse_square(self):
        for base_d in [50e6, 100e6, 200e6]:
            l1 = lc.free_space_loss_db(base_d)
            l2 = lc.free_space_loss_db(base_d * 2)
            assert abs((l2 - l1) - 6.02) < 0.15

    def test_sessions_equal_active_sols(self):
        terminal = lc.LaserTerminal()
        active_count = 0
        for s in range(100):
            result = lc.tick(terminal, sol=s, dust_tau=0.5)
            if result.link_active:
                active_count += 1
        assert terminal.total_sessions == active_count

    def test_dust_tau_non_negative(self):
        terminal = lc.LaserTerminal()
        for s in range(700):
            lc.tick(terminal, sol=s)
            assert terminal.dust_tau >= 0

    def test_throughput_within_dsoc_envelope(self):
        for sol in range(0, 800, 20):
            d = lc.earth_mars_distance_km(sol)
            thr = lc.achievable_throughput_mbps(d, dust_tau=0.0, zenith_deg=0.0)
            max_possible = lc.DSOC_REF_RATE_MBPS * (lc.DSOC_REF_DISTANCE_KM / max(d, 1)) ** 2
            assert thr <= max_possible * 1.01


class TestSerialisation:
    def test_round_trip(self, terminal):
        terminal.sol = 42
        terminal.total_data_gb = 123.456
        terminal.peak_throughput_mbps = 200.0
        d = lc.state_to_dict(terminal)
        restored = lc.state_from_dict(d)
        assert restored.sol == 42
        assert restored.total_data_gb == 123.456
        assert restored.peak_throughput_mbps == 200.0

    def test_json_serialisable(self, terminal):
        d = lc.state_to_dict(terminal)
        s = json.dumps(d)
        loaded = json.loads(s)
        restored = lc.state_from_dict(loaded)
        assert restored.sol == terminal.sol

    def test_empty_dict_gives_defaults(self):
        t = lc.state_from_dict({})
        assert t.sol == 0
        assert t.total_data_gb == 0.0

    def test_all_fields_preserved(self, terminal):
        for s in range(5):
            lc.tick(terminal, sol=s + 200, dust_tau=0.5)
        d = lc.state_to_dict(terminal)
        restored = lc.state_from_dict(d)
        assert restored.total_sessions == terminal.total_sessions
        assert abs(restored.total_data_gb - terminal.total_data_gb) < 0.001


class TestSimulation:
    def test_run_returns_correct_length(self):
        results = lc.run_simulation(sols=100)
        assert len(results) == 100

    def test_run_sols_sequential(self):
        results = lc.run_simulation(sols=50)
        for i, r in enumerate(results):
            assert r.sol == i

    def test_summarize_nonempty(self):
        results = lc.run_simulation(sols=100)
        summary = lc.summarize(results)
        assert summary["total_sols"] == 100
        assert summary["total_data_gb"] >= 0

    def test_summarize_empty(self):
        assert lc.summarize([]) == {}

    def test_run_with_fixed_dust(self):
        results = lc.run_simulation(sols=50, dust_tau=0.3)
        for r in results:
            if r.link_active:
                assert r.dust_tau == 0.3


class TestSmokeFullYear:
    def test_full_year_no_crash(self):
        results = lc.run_simulation(sols=668)
        assert len(results) == 668

    def test_full_year_has_active_and_blocked(self):
        results = lc.run_simulation(sols=668)
        active = sum(1 for r in results if r.link_active)
        blocked = sum(1 for r in results if not r.link_active)
        assert active > 0
        assert blocked > 0

    def test_full_year_transfers_data(self):
        results = lc.run_simulation(sols=668)
        assert results[-1].total_data_gb > 0

    def test_full_year_throughput_varies(self):
        results = lc.run_simulation(sols=668, dust_tau=0.5)
        active_thr = [r.throughput_mbps for r in results if r.link_active]
        assert len(active_thr) > 10
        assert max(active_thr) > 2.0 * min(active_thr)

    def test_full_year_rf_upgrade_meaningful(self):
        results = lc.run_simulation(sols=668, dust_tau=0.3)
        max_upgrade = max(r.rf_upgrade_factor for r in results)
        assert max_upgrade > 5.0

    def test_full_year_summary_consistent(self):
        results = lc.run_simulation(sols=668)
        summary = lc.summarize(results)
        assert summary["active_sessions"] + summary["blocked_sols"] + summary["storm_sols"] <= 668
        assert summary["peak_throughput_mbps"] >= summary["mean_throughput_mbps"]

    def test_full_year_substantial_data(self):
        results = lc.run_simulation(sols=668, dust_tau=0.5)
        assert results[-1].total_data_gb > 100.0


class TestEdgeCases:
    def test_negative_dust_tau_clamped(self):
        terminal = lc.LaserTerminal()
        lc.tick(terminal, sol=200, dust_tau=-1.0)
        assert terminal.dust_tau >= 0.0

    def test_very_large_sol(self):
        result = lc.tick(lc.LaserTerminal(), sol=100_000)
        assert isinstance(result, lc.TickResult)

    def test_very_small_distance(self):
        thr = lc.achievable_throughput_mbps(1.0, dust_tau=0.0, zenith_deg=0.0)
        assert thr >= 0.0

    def test_zero_zenith_angle(self):
        l = lc.mars_atmosphere_loss_db(0.5, 0.0)
        assert l >= 0.0

    def test_max_zenith_angle(self):
        l = lc.mars_atmosphere_loss_db(0.5, 89.0)
        assert l > 0.0 and l < 1000.0

    def test_multiple_years(self):
        results = lc.run_simulation(sols=2005)
        assert len(results) == 2005
        assert results[-1].total_data_gb > 0
