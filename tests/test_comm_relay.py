"""
Tests for comm_relay.py — Mars-Earth communication relay model.

Coverage:
  - Orbital geometry (distance, angle, conjunction detection)
  - Light delay (one-way and round-trip)
  - Link budget (data rates, SNR)
  - Message queue (enqueue, priority ordering, capacity)
  - Per-sol tick (bandwidth drain, blackout behaviour)
  - Physical invariants (distances in bounds, delay positive, rates positive)
  - Property sweeps across full synodic period
  - Multi-sol smoke test
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.comm_relay import (
    CONJUNCTION_BLACKOUT_SOLS,
    CONJUNCTION_HALF_ANGLE_DEG,
    DSN_MIN_RATE_KBPS,
    DSN_REF_DISTANCE_KM,
    DSN_REF_RATE_KBPS,
    EARTH_ORBIT_KM,
    MARS_ORBIT_KM,
    MAX_QUEUE_DEPTH,
    MESSAGE_OVERHEAD_KB,
    PRIORITY_BULK,
    PRIORITY_EMERGENCY,
    PRIORITY_ROUTINE,
    PRIORITY_SCIENCE,
    SNR_MIN_DB,
    SNR_REF_DB,
    SPEED_OF_LIGHT_KM_S,
    SYNODIC_PERIOD_SOLS,
    CommRelay,
    Message,
    downlink_rate_kbps,
    earth_mars_angle,
    earth_mars_distance_km,
    is_conjunction,
    light_delay_seconds,
    round_trip_delay_seconds,
    signal_to_noise_db,
    tick_comm,
    uplink_rate_kbps,
)


# ===================================================================
# Orbital geometry
# ===================================================================

class TestOrbitalGeometry:
    """Earth-Mars orbital geometry calculations."""

    def test_angle_at_sol_zero(self):
        """Sol 0 → angle 180° (conjunction — planets on opposite sides of Sun)."""
        assert earth_mars_angle(0) == pytest.approx(180.0)

    def test_angle_at_half_synodic(self):
        """Halfway through synodic period → angle ~0° (opposition — closest approach)."""
        half = int(SYNODIC_PERIOD_SOLS / 2)
        angle = earth_mars_angle(half)
        assert angle < 5.0 or angle > 355.0  # close to 0°/360°

    def test_angle_wraps_at_synodic_period(self):
        """Angle wraps back near 180° after one synodic period."""
        angle = earth_mars_angle(int(SYNODIC_PERIOD_SOLS))
        assert 175.0 < angle < 185.0

    def test_angle_always_0_to_360(self):
        """Angle is always in [0, 360) for any sol."""
        for sol in range(0, 2000, 7):
            a = earth_mars_angle(sol)
            assert 0.0 <= a < 360.0

    def test_distance_at_conjunction(self):
        """At conjunction (angle ≈ 0°), distance is near max (sum of orbits)."""
        # Sol 0 is conjunction
        d = earth_mars_distance_km(0)
        max_dist = EARTH_ORBIT_KM + MARS_ORBIT_KM
        # Should be close to max (within 1%)
        assert d > max_dist * 0.95

    def test_distance_at_opposition(self):
        """At opposition (angle ≈ 180°), distance is near min (difference of orbits)."""
        half = int(SYNODIC_PERIOD_SOLS / 2)
        d = earth_mars_distance_km(half)
        min_dist = MARS_ORBIT_KM - EARTH_ORBIT_KM
        # Should be close to min (within 5% — integer sol rounding)
        assert d < min_dist * 1.10

    def test_distance_always_positive(self):
        """Distance is always positive."""
        for sol in range(0, 2000, 13):
            assert earth_mars_distance_km(sol) > 0

    def test_distance_bounded(self):
        """Distance is always between (Mars-Earth) and (Mars+Earth) orbit radii."""
        min_possible = MARS_ORBIT_KM - EARTH_ORBIT_KM
        max_possible = MARS_ORBIT_KM + EARTH_ORBIT_KM
        for sol in range(0, 2000, 11):
            d = earth_mars_distance_km(sol)
            assert min_possible * 0.99 <= d <= max_possible * 1.01

    def test_distance_varies_over_synodic_period(self):
        """Distance changes — it's not constant."""
        distances = [earth_mars_distance_km(sol) for sol in range(0, int(SYNODIC_PERIOD_SOLS), 50)]
        assert max(distances) > 2 * min(distances)


# ===================================================================
# Light delay
# ===================================================================

class TestLightDelay:
    """Speed-of-light communication delay."""

    def test_one_way_delay_at_min_distance(self):
        """Minimum delay ≈ 3–4 minutes (at opposition)."""
        min_dist = MARS_ORBIT_KM - EARTH_ORBIT_KM
        delay_s = light_delay_seconds(min_dist)
        delay_min = delay_s / 60.0
        assert 3.0 < delay_min < 5.0

    def test_one_way_delay_at_max_distance(self):
        """Maximum delay ≈ 20–24 minutes (at conjunction)."""
        max_dist = MARS_ORBIT_KM + EARTH_ORBIT_KM
        delay_s = light_delay_seconds(max_dist)
        delay_min = delay_s / 60.0
        assert 20.0 < delay_min < 25.0

    def test_round_trip_is_double(self):
        """Round trip is exactly 2× one-way."""
        d = 200_000_000.0
        assert round_trip_delay_seconds(d) == pytest.approx(2.0 * light_delay_seconds(d))

    def test_delay_always_positive(self):
        """Delay is positive for any positive distance."""
        for d in [1.0, 1000.0, 1e9, 1e12]:
            assert light_delay_seconds(d) > 0

    def test_delay_zero_at_zero_distance(self):
        """Zero distance → zero delay."""
        assert light_delay_seconds(0.0) == 0.0

    def test_delay_proportional_to_distance(self):
        """Double the distance → double the delay."""
        d1 = 100_000_000.0
        d2 = 200_000_000.0
        assert light_delay_seconds(d2) == pytest.approx(2.0 * light_delay_seconds(d1))


# ===================================================================
# Conjunction / blackout
# ===================================================================

class TestConjunction:
    """Solar conjunction detection."""

    def test_conjunction_at_sol_zero(self):
        """Sol 0 is conjunction (angle = 0°)."""
        assert is_conjunction(0) is True

    def test_no_conjunction_at_opposition(self):
        """Opposition (angle ≈ 180°) is not conjunction."""
        half = int(SYNODIC_PERIOD_SOLS / 2)
        assert is_conjunction(half) is False

    def test_conjunction_at_synodic_boundary(self):
        """Near-synodic-period sol should be conjunction."""
        # Just before full period wraps
        near_end = int(SYNODIC_PERIOD_SOLS) - 1
        angle = earth_mars_angle(near_end)
        # Might or might not be in blackout depending on exact angle
        if angle > (360.0 - CONJUNCTION_HALF_ANGLE_DEG):
            assert is_conjunction(near_end) is True

    def test_conjunction_count_per_synodic_period(self):
        """Roughly CONJUNCTION_BLACKOUT_SOLS of blackout per synodic period."""
        blackout_count = sum(
            1 for sol in range(int(SYNODIC_PERIOD_SOLS))
            if is_conjunction(sol)
        )
        # Should be in a reasonable range (4–20 sols)
        assert 2 <= blackout_count <= 25

    def test_most_sols_are_not_blackout(self):
        """The vast majority of sols have comm link."""
        total = int(SYNODIC_PERIOD_SOLS)
        blackout_count = sum(1 for sol in range(total) if is_conjunction(sol))
        assert blackout_count < total * 0.05  # less than 5%


# ===================================================================
# Link budget
# ===================================================================

class TestLinkBudget:
    """Data rates and signal-to-noise ratio."""

    def test_downlink_at_ref_distance(self):
        """At 1 AU, downlink equals reference rate."""
        rate = downlink_rate_kbps(DSN_REF_DISTANCE_KM)
        assert rate == pytest.approx(DSN_REF_RATE_KBPS)

    def test_downlink_decreases_with_distance(self):
        """Farther away → lower data rate."""
        close = downlink_rate_kbps(EARTH_ORBIT_KM)
        far = downlink_rate_kbps(EARTH_ORBIT_KM * 3)
        assert far < close

    def test_downlink_inverse_square(self):
        """Rate falls as 1/r² — doubling distance → 1/4 rate."""
        d1 = 100_000_000.0
        d2 = 200_000_000.0
        r1 = downlink_rate_kbps(d1)
        r2 = downlink_rate_kbps(d2)
        assert r2 == pytest.approx(r1 / 4.0, rel=0.01)

    def test_downlink_never_below_minimum(self):
        """Rate is clamped to minimum even at extreme distance."""
        rate = downlink_rate_kbps(1e12)  # way past solar system
        assert rate >= DSN_MIN_RATE_KBPS

    def test_downlink_at_zero_distance(self):
        """Zero distance → reference rate (no division by zero)."""
        rate = downlink_rate_kbps(0.0)
        assert rate == DSN_REF_RATE_KBPS

    def test_uplink_is_fraction_of_downlink(self):
        """Uplink is 25% of downlink."""
        d = 200_000_000.0
        dl = downlink_rate_kbps(d)
        ul = uplink_rate_kbps(d)
        assert ul == pytest.approx(dl * 0.25)

    def test_snr_at_ref_distance(self):
        """SNR at 1 AU equals reference."""
        assert signal_to_noise_db(DSN_REF_DISTANCE_KM) == pytest.approx(SNR_REF_DB)

    def test_snr_decreases_with_distance(self):
        """SNR drops with distance."""
        close = signal_to_noise_db(EARTH_ORBIT_KM)
        far = signal_to_noise_db(EARTH_ORBIT_KM * 5)
        assert far < close

    def test_snr_at_zero_distance(self):
        """Zero distance → reference SNR (no log(0))."""
        assert signal_to_noise_db(0.0) == SNR_REF_DB

    def test_downlink_always_positive(self):
        """Rate is positive for any distance."""
        for sol in range(0, 1000, 50):
            d = earth_mars_distance_km(sol)
            assert downlink_rate_kbps(d) > 0


# ===================================================================
# Message and queue
# ===================================================================

class TestMessageQueue:
    """CommRelay message queue operations."""

    def test_enqueue_returns_message(self):
        """Enqueueing returns a Message object."""
        relay = CommRelay()
        msg = relay.enqueue(PRIORITY_ROUTINE, 10.0, "hello")
        assert isinstance(msg, Message)
        assert msg.priority == PRIORITY_ROUTINE
        assert msg.size_kb == 10.0

    def test_message_id_increments(self):
        """Each message gets a unique incrementing ID."""
        relay = CommRelay()
        m1 = relay.enqueue(PRIORITY_ROUTINE, 1.0)
        m2 = relay.enqueue(PRIORITY_ROUTINE, 1.0)
        assert m2.msg_id == m1.msg_id + 1

    def test_priority_ordering(self):
        """Queue is sorted by priority (emergency first)."""
        relay = CommRelay()
        relay.enqueue(PRIORITY_BULK, 1.0, "bulk")
        relay.enqueue(PRIORITY_EMERGENCY, 1.0, "emergency")
        relay.enqueue(PRIORITY_SCIENCE, 1.0, "science")
        assert relay.queue[0].priority == PRIORITY_EMERGENCY
        assert relay.queue[1].priority == PRIORITY_SCIENCE
        assert relay.queue[2].priority == PRIORITY_BULK

    def test_same_priority_fifo(self):
        """Messages with same priority are in FIFO order (by created_sol)."""
        relay = CommRelay(sol=0)
        relay.enqueue(PRIORITY_ROUTINE, 1.0, "first")
        relay.sol = 1
        relay.enqueue(PRIORITY_ROUTINE, 1.0, "second")
        assert relay.queue[0].content == "first"
        assert relay.queue[1].content == "second"

    def test_queue_depth(self):
        """queue_depth() returns correct count."""
        relay = CommRelay()
        assert relay.queue_depth() == 0
        relay.enqueue(PRIORITY_ROUTINE, 1.0)
        relay.enqueue(PRIORITY_ROUTINE, 2.0)
        assert relay.queue_depth() == 2

    def test_queue_size_kb(self):
        """queue_size_kb() includes overhead."""
        relay = CommRelay()
        relay.enqueue(PRIORITY_ROUTINE, 10.0)
        relay.enqueue(PRIORITY_ROUTINE, 20.0)
        expected = (10.0 + MESSAGE_OVERHEAD_KB) + (20.0 + MESSAGE_OVERHEAD_KB)
        assert relay.queue_size_kb() == pytest.approx(expected)

    def test_queue_full_drops_message(self):
        """Queue at capacity drops new messages."""
        relay = CommRelay()
        for i in range(MAX_QUEUE_DEPTH):
            assert relay.enqueue(PRIORITY_ROUTINE, 0.1) is not None
        assert relay.enqueue(PRIORITY_ROUTINE, 0.1) is None
        assert relay.total_messages_dropped == 1

    def test_message_total_size(self):
        """total_size_kb includes overhead."""
        msg = Message(msg_id=1, priority=0, size_kb=10.0, created_sol=0)
        assert msg.total_size_kb == 10.0 + MESSAGE_OVERHEAD_KB

    def test_negative_size_clamped(self):
        """Negative size is clamped to 0."""
        relay = CommRelay()
        msg = relay.enqueue(PRIORITY_ROUTINE, -5.0)
        assert msg.size_kb == 0.0


# ===================================================================
# Per-sol tick
# ===================================================================

class TestTickComm:
    """Per-sol communication tick."""

    def test_tick_returns_dict(self):
        """tick_comm returns a dict with required keys."""
        relay = CommRelay()
        result = tick_comm(relay, 100)
        required_keys = {
            "sol", "blackout", "distance_km", "one_way_delay_s",
            "round_trip_delay_s", "snr_db", "downlink_kbps",
            "uplink_kbps", "messages_sent", "data_sent_kb",
            "queue_depth", "queue_size_kb", "blackout_sols_total",
        }
        assert required_keys.issubset(result.keys())

    def test_tick_updates_sol(self):
        """tick_comm sets relay.sol."""
        relay = CommRelay()
        tick_comm(relay, 42)
        assert relay.sol == 42

    def test_blackout_sends_nothing(self):
        """During blackout, no messages are sent."""
        relay = CommRelay()
        relay.enqueue(PRIORITY_EMERGENCY, 1.0, "mayday")
        # Find a blackout sol
        blackout_sol = None
        for s in range(int(SYNODIC_PERIOD_SOLS)):
            if is_conjunction(s):
                blackout_sol = s
                break
        assert blackout_sol is not None, "No blackout found in synodic period"
        result = tick_comm(relay, blackout_sol)
        assert result["blackout"] is True
        assert result["messages_sent"] == 0
        assert result["downlink_kbps"] == 0.0
        assert relay.queue_depth() == 1  # message still queued

    def test_non_blackout_sends_messages(self):
        """Outside blackout, messages are delivered."""
        relay = CommRelay()
        relay.enqueue(PRIORITY_ROUTINE, 1.0, "test")
        # Find a non-blackout sol with good distance
        good_sol = int(SYNODIC_PERIOD_SOLS / 2)  # opposition
        result = tick_comm(relay, good_sol)
        assert result["blackout"] is False
        assert result["messages_sent"] >= 1
        assert relay.queue_depth() == 0

    def test_priority_draining_order(self):
        """Emergency messages are sent before routine ones."""
        relay = CommRelay()
        relay.enqueue(PRIORITY_ROUTINE, 1.0, "routine")
        relay.enqueue(PRIORITY_EMERGENCY, 1.0, "emergency")
        good_sol = int(SYNODIC_PERIOD_SOLS / 2)
        tick_comm(relay, good_sol)
        # Both should be delivered; emergency first in delivered list
        assert len(relay.delivered) >= 2
        assert relay.delivered[0].content == "emergency"
        assert relay.delivered[1].content == "routine"

    def test_bandwidth_limits_messages(self):
        """Very large messages can't all be sent in one sol."""
        relay = CommRelay()
        # Queue many huge messages
        for i in range(100):
            relay.enqueue(PRIORITY_ROUTINE, 1_000_000.0, f"huge_{i}")
        good_sol = int(SYNODIC_PERIOD_SOLS / 2)
        result = tick_comm(relay, good_sol)
        # Can't send all 100 × 1GB in one sol
        assert result["messages_sent"] < 100
        assert relay.queue_depth() > 0

    def test_cumulative_stats_tracked(self):
        """total_sent_kb and total_messages_sent accumulate."""
        relay = CommRelay()
        relay.enqueue(PRIORITY_ROUTINE, 1.0)
        good_sol = int(SYNODIC_PERIOD_SOLS / 2)
        tick_comm(relay, good_sol)
        assert relay.total_messages_sent >= 1
        assert relay.total_sent_kb > 0

    def test_blackout_sols_accumulate(self):
        """Blackout sols counter increases during conjunction."""
        relay = CommRelay()
        # Run through some blackout sols
        blackout_total = 0
        for sol in range(int(SYNODIC_PERIOD_SOLS)):
            if is_conjunction(sol):
                tick_comm(relay, sol)
                blackout_total += 1
                if blackout_total >= 3:
                    break
        assert relay.blackout_sols == blackout_total

    def test_delivered_messages_marked(self):
        """Delivered messages have delivered=True and delivered_sol set."""
        relay = CommRelay()
        relay.enqueue(PRIORITY_ROUTINE, 1.0, "test")
        good_sol = int(SYNODIC_PERIOD_SOLS / 2)
        tick_comm(relay, good_sol)
        assert len(relay.delivered) == 1
        assert relay.delivered[0].delivered is True
        assert relay.delivered[0].delivered_sol == good_sol


# ===================================================================
# Physical invariants
# ===================================================================

class TestInvariants:
    """Physical realism checks across the synodic period."""

    @pytest.mark.parametrize("sol", list(range(0, int(SYNODIC_PERIOD_SOLS), 50)))
    def test_distance_in_physical_bounds(self, sol):
        """Distance is always between min and max possible."""
        d = earth_mars_distance_km(sol)
        min_d = MARS_ORBIT_KM - EARTH_ORBIT_KM
        max_d = MARS_ORBIT_KM + EARTH_ORBIT_KM
        assert min_d * 0.99 <= d <= max_d * 1.01

    @pytest.mark.parametrize("sol", list(range(0, int(SYNODIC_PERIOD_SOLS), 50)))
    def test_delay_in_physical_bounds(self, sol):
        """One-way delay is between 3 and 23 minutes."""
        d = earth_mars_distance_km(sol)
        delay_min = light_delay_seconds(d) / 60.0
        assert 3.0 <= delay_min <= 23.0

    @pytest.mark.parametrize("sol", list(range(0, int(SYNODIC_PERIOD_SOLS), 50)))
    def test_rate_positive(self, sol):
        """Data rate is always positive."""
        d = earth_mars_distance_km(sol)
        assert downlink_rate_kbps(d) > 0
        assert uplink_rate_kbps(d) > 0

    def test_rate_range_over_orbit(self):
        """Data rates span a wide range over the synodic period."""
        rates = [downlink_rate_kbps(earth_mars_distance_km(s))
                 for s in range(0, int(SYNODIC_PERIOD_SOLS), 10)]
        ratio = max(rates) / min(rates)
        assert ratio > 10  # at least 10× variation

    def test_snr_range_over_orbit(self):
        """SNR varies over the synodic period."""
        snrs = [signal_to_noise_db(earth_mars_distance_km(s))
                for s in range(0, int(SYNODIC_PERIOD_SOLS), 10)]
        assert max(snrs) - min(snrs) > 5.0  # at least 5 dB variation


# ===================================================================
# Multi-sol smoke test
# ===================================================================

class TestSmoke:
    """Run the relay for many sols without crashing."""

    def test_100_sol_simulation(self):
        """Run 100 sols with periodic message injection."""
        relay = CommRelay()
        for sol in range(100):
            if sol % 5 == 0:
                relay.enqueue(PRIORITY_SCIENCE, 50.0, f"data_packet_{sol}")
            if sol % 20 == 0:
                relay.enqueue(PRIORITY_EMERGENCY, 1.0, f"alert_{sol}")
            result = tick_comm(relay, sol)
            assert isinstance(result, dict)
            assert result["sol"] == sol
        # Some messages should have been delivered
        assert relay.total_messages_sent > 0

    def test_full_synodic_period(self):
        """Run through an entire synodic period (764 sols)."""
        relay = CommRelay()
        blackout_count = 0
        for sol in range(int(SYNODIC_PERIOD_SOLS)):
            if sol % 10 == 0:
                relay.enqueue(PRIORITY_ROUTINE, 10.0, f"routine_{sol}")
            result = tick_comm(relay, sol)
            if result["blackout"]:
                blackout_count += 1
        assert blackout_count > 0  # at least some blackout
        assert relay.total_messages_sent > 0
        # Most messages should eventually be delivered
        assert relay.queue_depth() < 10

    def test_queue_overflow_recovery(self):
        """Fill queue, then drain over time."""
        relay = CommRelay()
        # Flood the queue
        for i in range(MAX_QUEUE_DEPTH + 10):
            relay.enqueue(PRIORITY_ROUTINE, 0.1, f"flood_{i}")
        assert relay.total_messages_dropped == 10
        assert relay.queue_depth() == MAX_QUEUE_DEPTH
        # Run enough sols to drain
        for sol in range(100, 300):
            tick_comm(relay, sol)
        # Should have drained significantly
        assert relay.queue_depth() < MAX_QUEUE_DEPTH
        assert relay.total_messages_sent > 0
