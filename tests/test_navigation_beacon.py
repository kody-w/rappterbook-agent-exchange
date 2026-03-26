"""Tests for navigation_beacon.py — Mars Surface Positioning Beacon Network."""
from __future__ import annotations

import json
import math
import os
import sys
import dataclasses

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import navigation_beacon as nb


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def beacon():
    return nb.create_beacon(beacon_id="b-0", x_m=0.0, y_m=0.0, mast_height_m=5.0)


@pytest.fixture
def network():
    return nb.create_network()


def _make_network(n_beacons, spacing_km=10.0):
    """Helper: create a network with n beacons in a circle."""
    configs = []
    r = spacing_km * 1000.0 / 2.0
    for i in range(n_beacons):
        angle = 2 * math.pi * i / n_beacons
        configs.append({
            "beacon_id": f"nav-{i}",
            "x_m": r * math.cos(angle),
            "y_m": r * math.sin(angle),
        })
    return nb.create_network(beacon_configs=configs)


# ── 1. Horizon geometry ────────────────────────────────────────────────

class TestHorizon:
    def test_zero_height(self):
        assert nb.horizon_distance_m(0.0) == 0.0

    def test_positive_height(self):
        d = nb.horizon_distance_m(2.0)
        expected = math.sqrt(2 * nb.MARS_RADIUS_M * 2.0)
        assert abs(d - expected) < 0.01

    def test_monotonic(self):
        assert nb.horizon_distance_m(10.0) > nb.horizon_distance_m(5.0)

    def test_mutual_horizon(self):
        mh = nb.mutual_horizon_m(5.0, 5.0)
        single = nb.horizon_distance_m(5.0)
        assert abs(mh - 2 * single) < 0.01

    def test_mutual_horizon_asymmetric(self):
        mh = nb.mutual_horizon_m(5.0, 10.0)
        assert mh == pytest.approx(
            nb.horizon_distance_m(5.0) + nb.horizon_distance_m(10.0), rel=1e-6
        )


# ── 2. Free-space path loss ────────────────────────────────────────────

class TestFSPL:
    def test_known_value(self):
        loss = nb.free_space_path_loss_db(1.0, 435.0)
        expected = 20 * math.log10(1.0) + 20 * math.log10(435.0) + 32.45
        assert abs(loss - expected) < 0.01

    def test_double_distance_adds_6db(self):
        l1 = nb.free_space_path_loss_db(10.0, 435.0)
        l2 = nb.free_space_path_loss_db(20.0, 435.0)
        assert abs((l2 - l1) - 6.02) < 0.1

    def test_positive(self):
        assert nb.free_space_path_loss_db(1.0, 435.0) > 0

    def test_zero_distance_returns_zero(self):
        assert nb.free_space_path_loss_db(0.0, 435.0) == 0.0


# ── 3. Dust attenuation ────────────────────────────────────────────────

class TestDust:
    def test_zero_tau(self):
        assert nb.dust_attenuation_db(10.0, 0.0) == 0.0

    def test_positive_tau(self):
        att = nb.dust_attenuation_db(10.0, 1.0)
        assert att > 0

    def test_monotonic_distance(self):
        assert nb.dust_attenuation_db(20.0, 1.0) > nb.dust_attenuation_db(10.0, 1.0)

    def test_monotonic_tau(self):
        assert nb.dust_attenuation_db(10.0, 2.0) > nb.dust_attenuation_db(10.0, 1.0)

    def test_storm_heavy(self):
        att = nb.dust_attenuation_db(50.0, 5.0)
        assert att > 10.0


# ── 4. Link budget ─────────────────────────────────────────────────────

class TestLinkBudget:
    def test_received_power_decreases_with_distance(self):
        p1 = nb.received_power_dbm(2.0, 3.0, 3.0, 10.0, 401.0, 0.0)
        p2 = nb.received_power_dbm(2.0, 3.0, 3.0, 50.0, 401.0, 0.0)
        assert p1 > p2

    def test_received_power_increases_with_tx(self):
        p1 = nb.received_power_dbm(1.0, 3.0, 3.0, 10.0, 401.0, 0.0)
        p2 = nb.received_power_dbm(5.0, 3.0, 3.0, 10.0, 401.0, 0.0)
        assert p2 > p1

    def test_dust_reduces_power(self):
        p_clear = nb.received_power_dbm(2.0, 3.0, 3.0, 10.0, 401.0, 0.0)
        p_dusty = nb.received_power_dbm(2.0, 3.0, 3.0, 10.0, 401.0, 2.0)
        assert p_clear > p_dusty


# ── 5. Max range ───────────────────────────────────────────────────────

class TestMaxRange:
    def test_positive(self):
        r = nb.max_range_km(2.0, 3.0, 3.0, 401.0, -110.0, 0.0)
        assert r > 0

    def test_more_power_more_range(self):
        r1 = nb.max_range_km(1.0, 3.0, 3.0, 401.0, -110.0, 0.0)
        r2 = nb.max_range_km(5.0, 3.0, 3.0, 401.0, -110.0, 0.0)
        assert r2 > r1

    def test_dust_reduces_range(self):
        r_clear = nb.max_range_km(2.0, 3.0, 3.0, 401.0, -110.0, 0.0)
        r_dusty = nb.max_range_km(2.0, 3.0, 3.0, 401.0, -110.0, 3.0)
        assert r_clear > r_dusty

    def test_high_power_range(self):
        r = nb.max_range_km(5.0, 3.0, 3.0, 401.0, -120.0, 0.0)
        assert r > 100


# ── 6. Ranging ─────────────────────────────────────────────────────────

class TestRanging:
    def test_roundtrip(self):
        d = nb.range_from_roundtrip_s(2e-5)
        expected = nb.SPEED_OF_LIGHT_M_S * 2e-5 / 2.0
        assert abs(d - expected) < 0.01

    def test_zero(self):
        assert nb.range_from_roundtrip_s(0.0) == 0.0

    def test_precision(self):
        p = nb.timing_to_range_precision_m(1.0)
        expected = nb.SPEED_OF_LIGHT_M_S * 1e-9 / 2.0
        assert abs(p - expected) < 0.001


# ── 7. Trilateration ──────────────────────────────────────────────────

class TestTrilateration:
    def test_exact_position(self):
        beacons_data = [
            (0.0, 0.0, math.sqrt(3000**2 + 4000**2)),
            (10000.0, 0.0, math.sqrt(7000**2 + 4000**2)),
            (0.0, 10000.0, math.sqrt(3000**2 + 6000**2)),
        ]
        result = nb.trilaterate_2d(beacons_data)
        assert result is not None
        x, y = result
        assert abs(x - 3000) < 1.0
        assert abs(y - 4000) < 1.0

    def test_at_origin(self):
        beacons_data = [
            (10000.0, 0.0, 10000.0),
            (0.0, 10000.0, 10000.0),
            (-10000.0, 0.0, 10000.0),
        ]
        result = nb.trilaterate_2d(beacons_data)
        assert result is not None
        x, y = result
        assert abs(x) < 1.0
        assert abs(y) < 1.0

    def test_insufficient_beacons_returns_none(self):
        result = nb.trilaterate_2d([(0.0, 0.0, 100.0)])
        assert result is None


# ── 8. GDOP ───────────────────────────────────────────────────────────

class TestGDOP:
    def test_square_arrangement(self):
        beacons_pos = [(0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0)]
        gdop = nb.geometric_dilution_of_precision(beacons_pos, (500.0, 500.0))
        assert gdop >= 1.0
        assert gdop < 5.0

    def test_collinear_elevated(self):
        beacons_pos = [(0.0, 0.0), (1000.0, 0.0), (2000.0, 0.0)]
        gdop = nb.geometric_dilution_of_precision(beacons_pos, (1000.0, 500.0))
        assert gdop > 1.0  # collinear geometry is suboptimal

    def test_minimum_one(self):
        beacons_pos = [(0.0, 0.0), (10000.0, 0.0), (0.0, 10000.0), (10000.0, 10000.0)]
        gdop = nb.geometric_dilution_of_precision(beacons_pos, (5000.0, 5000.0))
        assert gdop >= 1.0


# ── 9. Position error ─────────────────────────────────────────────────

class TestPositionError:
    def test_basic(self):
        err = nb.position_error_m(0.15, 1.5)
        assert abs(err - 1.5 * 0.15) < 0.001

    def test_zero_precision(self):
        assert nb.position_error_m(0.0, 1.0) == 0.0

    def test_higher_gdop_worse(self):
        e1 = nb.position_error_m(0.15, 1.5)
        e2 = nb.position_error_m(0.15, 3.0)
        assert e2 > e1


# ── 10. Battery and solar ──────────────────────────────────────────────

class TestBatterySolar:
    def test_capacity_at_warm(self):
        cap = nb.battery_capacity_at_temp(100.0, 293.0, 1.0)
        assert cap == 100.0

    def test_capacity_cold(self):
        cap = nb.battery_capacity_at_temp(100.0, 200.0, 1.0)
        assert cap < 100.0

    def test_capacity_minimum_floor(self):
        cap = nb.battery_capacity_at_temp(100.0, 100.0, 1.0)
        assert cap >= nb.BATTERY_MIN_CAPACITY_FRAC * 100.0

    def test_capacity_degraded(self):
        cap = nb.battery_capacity_at_temp(100.0, 293.0, 0.8)
        assert abs(cap - 80.0) < 0.01

    def test_solar_energy(self):
        e = nb.solar_energy_per_sol_wh(20.0, 0.3)
        assert e > 0

    def test_solar_zero_tau(self):
        e = nb.solar_energy_per_sol_wh(20.0, 0.0)
        assert e > 0

    def test_solar_heavy_dust(self):
        e_clear = nb.solar_energy_per_sol_wh(20.0, 0.0)
        e_dusty = nb.solar_energy_per_sol_wh(20.0, 5.0)
        assert e_clear > e_dusty

    def test_tx_energy(self):
        e = nb.tx_energy_per_sol_wh(2.0)
        assert e > 0

    def test_electronics_energy(self):
        e = nb.electronics_energy_per_sol_wh()
        assert e > 0


# ── 11. Beacon dataclass ─────────────────────────────────────────────

class TestBeaconDC:
    def test_create(self, beacon):
        assert beacon.beacon_id == "b-0"
        assert beacon.x_m == 0.0
        assert beacon.active

    def test_json_roundtrip(self, beacon):
        d = dataclasses.asdict(beacon)
        s = json.dumps(d)
        loaded = json.loads(s)
        b2 = nb.Beacon(**loaded)
        assert b2.beacon_id == beacon.beacon_id
        assert b2.x_m == beacon.x_m

    def test_defaults(self, beacon):
        assert beacon.battery_wh > 0
        assert beacon.battery_health == 1.0
        assert beacon.solar_panel_w > 0


# ── 12. Network state ────────────────────────────────────────────────

class TestNetworkState:
    def test_create_network(self, network):
        assert len(network.beacons) == 4
        assert network.sol == 0

    def test_beacon_positions_spaced(self, network):
        xs = [b.x_m for b in network.beacons]
        assert max(xs) - min(xs) > 0

    def test_json_roundtrip(self, network):
        d = dataclasses.asdict(network)
        s = json.dumps(d)
        loaded = json.loads(s)
        n2 = nb.NetworkState(
            beacons=[nb.Beacon(**b) for b in loaded["beacons"]],
            sol=loaded["sol"],
            total_fixes=loaded["total_fixes"],
            total_failed_fixes=loaded["total_failed_fixes"],
            best_precision_m=loaded["best_precision_m"],
            worst_precision_m=loaded["worst_precision_m"],
        )
        assert len(n2.beacons) == 4


# ── 13. Tick engine ──────────────────────────────────────────────────

class TestTick:
    def test_advances_sol(self, network):
        nb.tick(network)
        assert network.sol == 1

    def test_returns_tick_result(self, network):
        result = nb.tick(network)
        assert isinstance(result, nb.TickResult)
        assert isinstance(result.network_fix_available, bool)

    def test_gdop_positive(self, network):
        result = nb.tick(network)
        if result.network_fix_available:
            assert result.gdop >= 1.0

    def test_precision_positive(self, network):
        result = nb.tick(network)
        if result.network_fix_available:
            assert result.estimated_precision_m >= 0.0

    def test_battery_changes(self, network):
        nb.tick(network)
        final_battery = network.beacons[0].battery_wh
        assert not math.isnan(final_battery)

    def test_tick_battery_stays_bounded(self):
        state = nb.create_network()
        for _ in range(100):
            nb.tick(state)
            for b in state.beacons:
                effective_cap = nb.battery_capacity_at_temp(
                    b.battery_capacity_wh, b.temp_k, b.battery_health
                )
                assert b.battery_wh >= 0.0
                assert b.battery_wh <= effective_cap + 0.01

    def test_dead_battery_no_signal(self):
        b = nb.create_beacon("dead", 0.0, 0.0, 5.0)
        b.battery_wh = 0.0
        b.solar_panel_w = 0.0
        b.active = False
        net = nb.NetworkState(beacons=[b])
        result = nb.tick(net)
        assert result.network_fix_available is False

    def test_energy_bookkeeping(self, network):
        result = nb.tick(network)
        assert result.solar_harvest_wh > 0
        assert result.tx_consumed_wh > 0
        assert abs(result.net_energy_wh - (result.solar_harvest_wh - result.tx_consumed_wh)) < 0.01


# ── 14. Multi-sol simulation ─────────────────────────────────────────

class TestSimulation:
    def test_run_10_sols(self):
        net = nb.create_network()
        results = nb.run_simulation(net, sols=10)
        assert len(results) == 10

    def test_sol_counter_increments(self):
        net = nb.create_network()
        results = nb.run_simulation(net, sols=5)
        for i, r in enumerate(results):
            assert r.sol == i + 1

    def test_no_crash_100_sols(self):
        net = _make_network(6, spacing_km=5.0)
        results = nb.run_simulation(net, sols=100)
        assert len(results) == 100

    def test_dust_storm(self):
        net = nb.create_network()
        results = nb.run_simulation(net, sols=10, optical_depth=4.0)
        assert len(results) == 10

    def test_extreme_cold(self):
        net = nb.create_network()
        results = nb.run_simulation(net, sols=10, temp_k=150.0)
        assert len(results) == 10


# ── 15. Conservation and bounds ────────────────────────────────────────

class TestConservation:
    def test_battery_health_monotonic_decrease(self):
        net = nb.create_network()
        initial_health = net.beacons[0].battery_health
        nb.run_simulation(net, sols=50)
        assert net.beacons[0].battery_health < initial_health

    def test_battery_non_negative(self):
        net = nb.create_network()
        nb.run_simulation(net, sols=100, temp_k=180.0)
        for b in net.beacons:
            assert b.battery_wh >= 0.0

    def test_gdop_floor(self):
        net = nb.create_network()
        results = nb.run_simulation(net, sols=20)
        for r in results:
            if r.network_fix_available:
                assert r.gdop >= 1.0

    def test_position_error_non_negative(self):
        net = nb.create_network()
        results = nb.run_simulation(net, sols=20)
        for r in results:
            if r.network_fix_available:
                assert r.estimated_precision_m >= 0.0

    def test_total_fixes_accumulate(self):
        net = nb.create_network()
        nb.run_simulation(net, sols=10)
        assert net.total_fixes + net.total_failed_fixes == 10


# ── 16. Parametrised invariants ──────────────────────────────────────

class TestParametrised:
    @pytest.mark.parametrize("height", [1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    def test_horizon_positive(self, height):
        assert nb.horizon_distance_m(height) > 0

    @pytest.mark.parametrize("dist_km", [1.0, 5.0, 10.0, 50.0, 100.0])
    def test_fspl_positive(self, dist_km):
        assert nb.free_space_path_loss_db(dist_km, 435.0) > 0

    @pytest.mark.parametrize("tau", [0.0, 0.5, 1.0, 2.0, 5.0])
    def test_dust_non_negative(self, tau):
        assert nb.dust_attenuation_db(10.0, tau) >= 0

    @pytest.mark.parametrize("n_beacons", [3, 4, 6, 8])
    def test_network_sizes(self, n_beacons):
        net = _make_network(n_beacons, spacing_km=10.0)
        results = nb.run_simulation(net, sols=5)
        assert len(results) == 5

    @pytest.mark.parametrize("temp_k", [150.0, 200.0, 250.0, 293.0])
    def test_battery_capacity_bounds(self, temp_k):
        cap = nb.battery_capacity_at_temp(100.0, temp_k, 1.0)
        assert 0 < cap <= 100.0

    @pytest.mark.parametrize("n_beacons", [3, 4, 6, 8])
    def test_spacing_variants(self, n_beacons):
        net = _make_network(n_beacons, spacing_km=10.0)
        result = nb.tick(net)
        if result.network_fix_available:
            assert result.gdop >= 1.0


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_beacon_no_fix(self):
        net = _make_network(1, spacing_km=10.0)
        result = nb.tick(net)
        assert result.network_fix_available is False

    def test_two_beacons_no_fix(self):
        net = _make_network(2, spacing_km=10.0)
        result = nb.tick(net)
        assert result.network_fix_available is False

    def test_three_beacons_fix(self):
        net = _make_network(3, spacing_km=10.0)
        result = nb.tick(net)
        assert result.network_fix_available is True

    def test_very_large_network(self):
        net = _make_network(20, spacing_km=5.0)
        result = nb.tick(net)
        assert result.network_fix_available is True

    def test_all_beacons_dead(self):
        net = nb.create_network()
        for b in net.beacons:
            b.battery_wh = 0.0
            b.solar_panel_w = 0.0
            b.active = False
        result = nb.tick(net)
        assert result.network_fix_available is False

    def test_zero_spacing(self):
        configs = [{"beacon_id": f"nav-{i}", "x_m": 0.0, "y_m": 0.0} for i in range(4)]
        net = nb.create_network(beacon_configs=configs)
        result = nb.tick(net)
        assert isinstance(result.network_fix_available, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
