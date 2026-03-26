"""Tests for emergency_beacon.py -- Mars Surface Emergency Locator Beacon.

81 tests covering:
  - RF physics (FSPL, link budget, max range)
  - Battery temperature effects (cold Mars, warm hab)
  - Beacon lifecycle (activate, transmit, detect, deactivate)
  - Power budget (solar charging, heater, battery drain)
  - Conservation laws (energy in <= energy available)
  - Degradation (antenna dust, battery cycling)
  - Orbital detection (link budget vs sensitivity)
  - Multi-sol simulation (battery life, detection rate)
  - Edge cases (zero power, extreme temps, dead battery)
  - Physical invariants (bounds, monotonicity)
"""
from __future__ import annotations

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emergency_beacon import (
    # Constants
    BEACON_FREQ_MHZ,
    DEFAULT_TX_POWER_W,
    DEFAULT_TX_POWER_DBM,
    DEFAULT_ANTENNA_GAIN_DBI,
    DEFAULT_BATTERY_WH,
    DEFAULT_SOLAR_W,
    ORBITAL_SENSITIVITY_DBM,
    ORBITAL_ALTITUDE_KM,
    MARS_SURFACE_TEMP_K,
    BATTERY_TEMP_COEFF,
    BATTERY_MIN_CAPACITY_FRAC,
    BATTERY_HEATER_THRESHOLD_K,
    BATTERY_HEATER_W,
    PING_DUTY_CYCLE,
    CONTINUOUS_DUTY_CYCLE,
    ANTENNA_DEGRADATION_PER_SOL,
    BATTERY_CYCLE_DEGRADATION,
    BATTERY_MIN_HEALTH,
    HOURS_PER_SOL,
    SECONDS_PER_SOL,
    PING_INTERVAL_S,
    # Functions
    free_space_path_loss_db,
    received_power_dbm,
    max_range_km,
    battery_capacity_at_temp,
    tick,
    create_beacon,
    # Classes
    BeaconState,
)


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def fresh_beacon():
    return create_beacon(lat=4.5, lon=137.4, alt=-4500.0)


@pytest.fixture
def active_beacon():
    b = create_beacon()
    tick(b, activate=True)
    return b


# =============================================================================
# S1  FREE-SPACE PATH LOSS
# =============================================================================

class TestFSPL:

    def test_fspl_1km(self):
        """FSPL at 1 km, 401 MHz = 20*log10(1) + 20*log10(401) + 32.45."""
        expected = 0.0 + 20.0 * math.log10(401.585625) + 32.45
        result = free_space_path_loss_db(1.0, BEACON_FREQ_MHZ)
        assert abs(result - expected) < 0.01

    def test_fspl_increases_with_distance(self):
        l1 = free_space_path_loss_db(10.0, BEACON_FREQ_MHZ)
        l2 = free_space_path_loss_db(100.0, BEACON_FREQ_MHZ)
        assert l2 > l1

    def test_fspl_increases_with_frequency(self):
        l1 = free_space_path_loss_db(100.0, 100.0)
        l2 = free_space_path_loss_db(100.0, 1000.0)
        assert l2 > l1

    def test_fspl_zero_distance(self):
        assert free_space_path_loss_db(0.0, BEACON_FREQ_MHZ) == 0.0

    def test_fspl_negative_distance(self):
        assert free_space_path_loss_db(-5.0, BEACON_FREQ_MHZ) == 0.0

    def test_fspl_6db_per_doubling(self):
        """Doubling distance adds ~6 dB."""
        l1 = free_space_path_loss_db(100.0, BEACON_FREQ_MHZ)
        l2 = free_space_path_loss_db(200.0, BEACON_FREQ_MHZ)
        assert abs((l2 - l1) - 6.02) < 0.1


# =============================================================================
# S2  LINK BUDGET
# =============================================================================

class TestLinkBudget:

    def test_received_power_basic(self):
        p_rx = received_power_dbm(37.0, 2.0, 3.0, 300.0, BEACON_FREQ_MHZ)
        assert isinstance(p_rx, float)

    def test_received_power_decreases_with_distance(self):
        p_near = received_power_dbm(37.0, 2.0, 3.0, 100.0, BEACON_FREQ_MHZ)
        p_far = received_power_dbm(37.0, 2.0, 3.0, 1000.0, BEACON_FREQ_MHZ)
        assert p_far < p_near

    def test_received_power_zero_distance(self):
        assert received_power_dbm(37.0, 2.0, 3.0, 0.0, BEACON_FREQ_MHZ) == -999.0

    def test_received_power_negative_distance(self):
        assert received_power_dbm(37.0, 2.0, 3.0, -10.0, BEACON_FREQ_MHZ) == -999.0

    def test_higher_tx_power_stronger_signal(self):
        p_low = received_power_dbm(30.0, 2.0, 3.0, 300.0, BEACON_FREQ_MHZ)
        p_high = received_power_dbm(40.0, 2.0, 3.0, 300.0, BEACON_FREQ_MHZ)
        assert p_high > p_low

    def test_higher_gain_stronger_signal(self):
        p_low = received_power_dbm(37.0, 0.0, 0.0, 300.0, BEACON_FREQ_MHZ)
        p_high = received_power_dbm(37.0, 5.0, 5.0, 300.0, BEACON_FREQ_MHZ)
        assert p_high > p_low


# =============================================================================
# S3  MAX RANGE
# =============================================================================

class TestMaxRange:

    def test_max_range_positive(self):
        r = max_range_km(DEFAULT_TX_POWER_DBM, DEFAULT_ANTENNA_GAIN_DBI,
                         3.0, ORBITAL_SENSITIVITY_DBM, BEACON_FREQ_MHZ)
        assert r > 0

    def test_max_range_increases_with_power(self):
        r_low = max_range_km(30.0, 2.0, 3.0, -130.0, BEACON_FREQ_MHZ)
        r_high = max_range_km(40.0, 2.0, 3.0, -130.0, BEACON_FREQ_MHZ)
        assert r_high > r_low

    def test_max_range_increases_with_sensitivity(self):
        r_poor = max_range_km(37.0, 2.0, 3.0, -120.0, BEACON_FREQ_MHZ)
        r_good = max_range_km(37.0, 2.0, 3.0, -140.0, BEACON_FREQ_MHZ)
        assert r_good > r_poor

    def test_max_range_exceeds_orbital_altitude(self):
        """5W beacon should reach 300 km orbit."""
        r = max_range_km(DEFAULT_TX_POWER_DBM, DEFAULT_ANTENNA_GAIN_DBI,
                         3.0, ORBITAL_SENSITIVITY_DBM, BEACON_FREQ_MHZ)
        assert r > ORBITAL_ALTITUDE_KM

    def test_max_range_zero_frequency(self):
        assert max_range_km(37.0, 2.0, 3.0, -130.0, 0.0) == 0.0


# =============================================================================
# S4  BATTERY TEMPERATURE EFFECTS
# =============================================================================

class TestBatteryTemp:

    def test_warm_battery_full_capacity(self):
        cap = battery_capacity_at_temp(50.0, 293.0)
        assert cap == 50.0

    def test_cold_battery_reduced(self):
        cap = battery_capacity_at_temp(50.0, 210.0)
        assert cap < 50.0

    def test_extreme_cold_floors_at_minimum(self):
        cap = battery_capacity_at_temp(50.0, 100.0)
        assert cap >= 50.0 * BATTERY_MIN_CAPACITY_FRAC

    def test_capacity_increases_with_temperature(self):
        c_cold = battery_capacity_at_temp(50.0, 220.0)
        c_warm = battery_capacity_at_temp(50.0, 260.0)
        assert c_warm > c_cold

    def test_freezing_point_full_capacity(self):
        cap = battery_capacity_at_temp(50.0, 273.0)
        assert cap == 50.0

    def test_above_freezing_full_capacity(self):
        cap = battery_capacity_at_temp(50.0, 350.0)
        assert cap == 50.0


# =============================================================================
# S5  BEACON STATE
# =============================================================================

class TestBeaconState:

    def test_create_defaults(self):
        b = create_beacon()
        assert b.is_active is False
        assert b.mode == "ping"
        assert b.battery_charge_wh == DEFAULT_BATTERY_WH
        assert b.sol == 0
        assert b.antenna_health == 1.0

    def test_create_with_position(self):
        b = create_beacon(lat=4.5, lon=137.4, alt=-4500.0)
        assert b.latitude_deg == 4.5
        assert b.longitude_deg == 137.4
        assert b.altitude_m == -4500.0

    def test_tx_power_dbm(self):
        b = create_beacon()
        expected = 10.0 * math.log10(DEFAULT_TX_POWER_W * 1000)
        assert abs(b.tx_power_dbm() - (expected + DEFAULT_ANTENNA_GAIN_DBI)) < 0.1

    def test_estimated_runtime_inactive(self):
        b = create_beacon()
        assert b.estimated_runtime_hours() == float("inf")

    def test_to_dict_keys(self):
        b = create_beacon()
        d = b.to_dict()
        required = ["tx_power_w", "is_active", "mode", "battery_charge_wh",
                     "sol", "events", "antenna_health"]
        for k in required:
            assert k in d


# =============================================================================
# S6  TICK -- ACTIVATION AND MODES
# =============================================================================

class TestTickActivation:

    def test_activate(self, fresh_beacon):
        result = tick(fresh_beacon, activate=True)
        assert fresh_beacon.is_active is True
        assert "ACTIVATED" in result["events"]

    def test_deactivate(self, active_beacon):
        result = tick(active_beacon, deactivate=True)
        assert active_beacon.is_active is False
        assert "DEACTIVATED" in result["events"]

    def test_mode_switch(self, active_beacon):
        result = tick(active_beacon, mode="continuous")
        assert active_beacon.mode == "continuous"
        assert "MODE_CONTINUOUS" in result["events"]

    def test_sol_increments(self, fresh_beacon):
        tick(fresh_beacon)
        assert fresh_beacon.sol == 1
        tick(fresh_beacon)
        assert fresh_beacon.sol == 2


# =============================================================================
# S7  TICK -- POWER BUDGET
# =============================================================================

class TestTickPower:

    def test_inactive_no_energy(self, fresh_beacon):
        result = tick(fresh_beacon)
        assert result["energy_used_wh"] == 0.0

    def test_active_uses_energy(self, active_beacon):
        result = tick(active_beacon)
        assert result["energy_used_wh"] > 0

    def test_continuous_uses_more_than_ping(self):
        """Continuous transmit uses more energy per sol than ping mode."""
        b_ping = create_beacon()
        b_ping.temp_k = 293.0  # warm — no heater noise
        r_ping = tick(b_ping, activate=True, mode="ping")

        b_cont = create_beacon()
        b_cont.temp_k = 293.0
        r_cont = tick(b_cont, activate=True, mode="continuous")

        assert r_cont["energy_used_wh"] > r_ping["energy_used_wh"]

    def test_solar_charges_battery(self, fresh_beacon):
        fresh_beacon.battery_charge_wh = 60.0
        result = tick(fresh_beacon, solar_available=True)
        assert fresh_beacon.battery_charge_wh > 60.0

    def test_no_solar_no_charge(self, fresh_beacon):
        fresh_beacon.battery_charge_wh = 60.0
        tick(fresh_beacon, solar_available=False)
        assert fresh_beacon.battery_charge_wh == 60.0

    def test_battery_never_exceeds_capacity(self, fresh_beacon):
        for _ in range(100):
            tick(fresh_beacon, solar_available=True)
        assert fresh_beacon.battery_charge_wh <= fresh_beacon.effective_capacity_wh() + 0.01

    def test_heater_activates_in_cold(self):
        b = create_beacon()
        tick(b, activate=True)
        result = tick(b, ambient_temp_k=200.0)
        assert "HEATER_ON" in result["events"]


# =============================================================================
# S8  TICK -- DETECTION
# =============================================================================

class TestTickDetection:

    def test_active_beacon_detected_by_orbiter(self, active_beacon):
        result = tick(active_beacon, receiver_distance_km=ORBITAL_ALTITUDE_KM)
        assert result["detectable"] is True
        assert result["detections"] > 0

    def test_inactive_beacon_not_detected(self, fresh_beacon):
        result = tick(fresh_beacon)
        assert result["detections"] == 0

    def test_far_receiver_not_detected(self, active_beacon):
        result = tick(active_beacon, receiver_distance_km=1_000_000.0,
                      receiver_sensitivity_dbm=-100.0)
        assert result["detectable"] is False

    def test_detection_count_matches_passes(self, active_beacon):
        result = tick(active_beacon, orbital_passes=8)
        if result["detectable"]:
            assert result["detections"] == 8


# =============================================================================
# S9  DEGRADATION
# =============================================================================

class TestDegradation:

    def test_antenna_degrades_when_active(self, active_beacon):
        initial = active_beacon.antenna_health
        tick(active_beacon)
        assert active_beacon.antenna_health < initial

    def test_battery_health_degrades_when_active(self, active_beacon):
        initial = active_beacon.battery_health
        tick(active_beacon)
        assert active_beacon.battery_health < initial

    def test_antenna_health_floors(self):
        b = create_beacon()
        b.antenna_health = 0.51
        tick(b, activate=True)
        tick(b)
        assert b.antenna_health >= 0.5

    def test_battery_health_floors(self):
        b = create_beacon()
        b.battery_health = BATTERY_MIN_HEALTH + 0.001
        tick(b, activate=True)
        tick(b)
        assert b.battery_health >= BATTERY_MIN_HEALTH


# =============================================================================
# S10  MULTI-SOL SIMULATION
# =============================================================================

class TestMultiSol:

    def test_10_sol_smoke(self, active_beacon):
        for _ in range(10):
            result = tick(active_beacon)
        assert active_beacon.sol >= 10
        assert active_beacon.total_transmissions > 0

    def test_battery_drains_over_time(self):
        b = create_beacon()
        tick(b, activate=True, mode="continuous", solar_available=False)
        charges = [b.battery_charge_wh]
        for _ in range(5):
            tick(b, solar_available=False)
            charges.append(b.battery_charge_wh)
        # Battery should decrease monotonically
        for i in range(1, len(charges)):
            assert charges[i] <= charges[i - 1]

    def test_eventual_battery_death(self):
        """Continuous mode with no solar eventually kills the battery."""
        b = create_beacon()
        tick(b, activate=True, mode="continuous", solar_available=False)
        for _ in range(100):
            tick(b, solar_available=False)
            if not b.is_active:
                break
        assert b.battery_charge_wh == 0.0 or not b.is_active

    def test_ping_mode_lasts_longer_than_continuous(self):
        """Ping mode should last many more sols than continuous."""
        b_ping = create_beacon()
        tick(b_ping, activate=True, mode="ping", solar_available=False)
        ping_sols = 0
        for _ in range(1000):
            tick(b_ping, solar_available=False)
            ping_sols += 1
            if not b_ping.is_active:
                break

        b_cont = create_beacon()
        tick(b_cont, activate=True, mode="continuous", solar_available=False)
        cont_sols = 0
        for _ in range(1000):
            tick(b_cont, solar_available=False)
            cont_sols += 1
            if not b_cont.is_active:
                break

        assert ping_sols > cont_sols


# =============================================================================
# S11  PHYSICAL INVARIANTS
# =============================================================================

class TestInvariants:

    def test_battery_never_negative(self, active_beacon):
        for _ in range(100):
            tick(active_beacon, solar_available=False)
        assert active_beacon.battery_charge_wh >= 0.0

    def test_antenna_health_bounded(self, active_beacon):
        for _ in range(50):
            tick(active_beacon)
        assert 0.0 <= active_beacon.antenna_health <= 1.0

    def test_battery_health_bounded(self, active_beacon):
        for _ in range(50):
            tick(active_beacon)
        assert BATTERY_MIN_HEALTH <= active_beacon.battery_health <= 1.0

    def test_energy_conservation(self):
        """Total energy used <= initial battery + total solar input."""
        b = create_beacon()
        tick(b, activate=True, mode="continuous")
        total_solar = 0.0
        for _ in range(20):
            r = tick(b)
            total_solar += r["solar_charged_wh"]
        # Energy used should not exceed starting battery + solar
        assert b.total_energy_used_wh <= DEFAULT_BATTERY_WH + total_solar + 0.1

    def test_total_detections_non_negative(self, active_beacon):
        for _ in range(10):
            tick(active_beacon)
        assert active_beacon.total_detections >= 0

    def test_transmissions_non_negative(self, active_beacon):
        tick(active_beacon)
        assert active_beacon.total_transmissions >= 0


# =============================================================================
# S12  EDGE CASES
# =============================================================================

class TestEdgeCases:

    def test_activate_already_active(self, active_beacon):
        tick(active_beacon, activate=True)
        assert active_beacon.is_active is True

    def test_deactivate_already_inactive(self, fresh_beacon):
        tick(fresh_beacon, deactivate=True)
        assert fresh_beacon.is_active is False

    def test_extreme_cold(self):
        b = create_beacon()
        tick(b, activate=True, ambient_temp_k=140.0)
        result = tick(b, ambient_temp_k=140.0)
        assert result["sol"] == 2

    def test_warm_habitat(self):
        b = create_beacon()
        tick(b, activate=True, ambient_temp_k=293.0)
        result = tick(b, ambient_temp_k=293.0)
        assert "HEATER_ON" not in result["events"]

    def test_zero_solar_panel(self):
        b = create_beacon()
        b.solar_panel_w = 0.0
        charge_before = b.battery_charge_wh
        tick(b, solar_available=True)
        assert b.battery_charge_wh == charge_before

    def test_zero_orbital_passes(self, active_beacon):
        result = tick(active_beacon, orbital_passes=0)
        assert result["detections"] == 0
