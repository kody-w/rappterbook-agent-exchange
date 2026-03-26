"""
comm_relay.py — Mars-Earth communication relay model.

Models the physical constraints of interplanetary communication:
  - Light-speed delay (4–24 min one-way depending on orbital geometry)
  - Solar conjunction blackout (~14 sols every synodic period)
  - Deep Space Network bandwidth (limited uplink/downlink data rates)
  - Message queue with priority classes (emergency, science, routine)
  - Link budget: signal-to-noise ratio varies with distance

Physical references:
  - Speed of light: 299,792.458 km/s
  - Earth-Mars closest approach: ~55.7 million km (opposition)
  - Earth-Mars farthest: ~401 million km (conjunction)
  - Mars synodic period: ~780 Earth days ≈ 764 sols
  - DSN X-band downlink: 0.5–6 Mbps depending on distance
  - DSN Ka-band downlink: 2–25 Mbps (newer orbiters)
  - Solar conjunction blackout: Sun-Earth-Probe angle < 2°
  - Typical blackout duration: ~14 Earth days ≈ 13.6 sols

One tick = one sol.  Distances in km, rates in kbps, delays in seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT_KM_S = 299_792.458

# Orbital parameters (simplified circular orbits)
EARTH_ORBIT_KM = 149_597_870.7        # 1 AU
MARS_ORBIT_KM = 227_939_200.0         # 1.524 AU
SYNODIC_PERIOD_SOLS = 764.0           # ~780 Earth days in sols

# Conjunction blackout
CONJUNCTION_HALF_ANGLE_DEG = 2.0       # Sun-Earth-Probe angle threshold
CONJUNCTION_BLACKOUT_SOLS = 14         # approximate blackout duration

# DSN link parameters (X-band baseline)
DSN_REF_RATE_KBPS = 2000.0            # reference downlink at 1 AU distance
DSN_REF_DISTANCE_KM = EARTH_ORBIT_KM  # 1 AU reference
DSN_MIN_RATE_KBPS = 0.5               # minimum viable data rate
DSN_UPLINK_FRACTION = 0.25            # uplink is ~25% of downlink capacity

# Signal-to-noise
SNR_REF_DB = 40.0                     # reference SNR at 1 AU
SNR_MIN_DB = 3.0                      # minimum decodable SNR

# Message priority classes
PRIORITY_EMERGENCY = 0
PRIORITY_SCIENCE = 1
PRIORITY_ROUTINE = 2
PRIORITY_BULK = 3

# Queue limits
MAX_QUEUE_DEPTH = 500                  # max messages buffered
MESSAGE_OVERHEAD_KB = 0.5             # protocol overhead per message


# ---------------------------------------------------------------------------
# Orbital geometry
# ---------------------------------------------------------------------------

def earth_mars_angle(sol: int) -> float:
    """Angular separation of Earth and Mars as seen from the Sun (degrees).

    Simplified model: both planets in circular coplanar orbits.
    The angle oscillates with the synodic period.  At conjunction
    (sol 0) the angle is 180° (planets on opposite sides of Sun).
    At opposition (half synodic) the angle is ~0° (closest approach).
    """
    phase = (sol % SYNODIC_PERIOD_SOLS) / SYNODIC_PERIOD_SOLS * 360.0
    return (phase + 180.0) % 360.0


def earth_mars_distance_km(sol: int) -> float:
    """Distance between Earth and Mars (km) based on simplified geometry.

    Uses law of cosines with the Sun-centred angle between the two planets.
    d² = r_e² + r_m² - 2·r_e·r_m·cos(θ)
    """
    theta_deg = earth_mars_angle(sol)
    theta_rad = math.radians(theta_deg)
    d_sq = (EARTH_ORBIT_KM ** 2 + MARS_ORBIT_KM ** 2
            - 2 * EARTH_ORBIT_KM * MARS_ORBIT_KM * math.cos(theta_rad))
    return math.sqrt(max(d_sq, 0.0))


def light_delay_seconds(distance_km: float) -> float:
    """One-way light travel time in seconds."""
    return distance_km / SPEED_OF_LIGHT_KM_S


def round_trip_delay_seconds(distance_km: float) -> float:
    """Round-trip light travel time in seconds."""
    return 2.0 * light_delay_seconds(distance_km)


def is_conjunction(sol: int) -> bool:
    """True if Mars is in solar conjunction (comm blackout).

    Conjunction occurs when the Sun-centred angle is near 180°,
    meaning Earth and Mars are on opposite sides of the Sun.
    """
    angle = earth_mars_angle(sol)
    return abs(angle - 180.0) < CONJUNCTION_HALF_ANGLE_DEG


# ---------------------------------------------------------------------------
# Link budget
# ---------------------------------------------------------------------------

def downlink_rate_kbps(distance_km: float) -> float:
    """Achievable downlink data rate (kbps) given Earth-Mars distance.

    Signal power falls as 1/r².  Data rate scales linearly with received
    power (Shannon limit simplified).  Rate = ref_rate × (ref_dist / dist)².
    Clamped to minimum viable rate.
    """
    if distance_km <= 0:
        return DSN_REF_RATE_KBPS
    ratio = (DSN_REF_DISTANCE_KM / distance_km) ** 2
    rate = DSN_REF_RATE_KBPS * ratio
    return max(rate, DSN_MIN_RATE_KBPS)


def uplink_rate_kbps(distance_km: float) -> float:
    """Achievable uplink data rate (kbps).  Fraction of downlink."""
    return downlink_rate_kbps(distance_km) * DSN_UPLINK_FRACTION


def signal_to_noise_db(distance_km: float) -> float:
    """Signal-to-noise ratio in dB at given distance.

    SNR drops as 20·log10(ref/dist) — inverse-square law in dB.
    """
    if distance_km <= 0:
        return SNR_REF_DB
    return SNR_REF_DB - 20.0 * math.log10(distance_km / DSN_REF_DISTANCE_KM)


# ---------------------------------------------------------------------------
# Message and queue
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single communication message."""
    msg_id: int
    priority: int          # 0=emergency, 1=science, 2=routine, 3=bulk
    size_kb: float         # payload size in kilobytes
    created_sol: int       # sol when message was created
    content: str = ""      # human-readable summary
    delivered: bool = False
    delivered_sol: int = -1

    @property
    def total_size_kb(self) -> float:
        """Payload plus protocol overhead."""
        return self.size_kb + MESSAGE_OVERHEAD_KB


@dataclass
class CommRelay:
    """Mars-Earth communication relay state.

    Tracks the message queue, bandwidth usage, blackout status,
    and cumulative statistics.
    """
    sol: int = 0
    next_msg_id: int = 1
    queue: list = field(default_factory=list)       # pending messages
    delivered: list = field(default_factory=list)    # delivered messages
    blackout: bool = False
    total_sent_kb: float = 0.0
    total_messages_sent: int = 0
    total_messages_dropped: int = 0
    blackout_sols: int = 0                          # cumulative blackout sols

    def enqueue(self, priority: int, size_kb: float, content: str = "") -> Message | None:
        """Add a message to the outgoing queue.

        Returns the Message if accepted, None if queue is full.
        Messages are priority-sorted (lower number = higher priority).
        """
        if len(self.queue) >= MAX_QUEUE_DEPTH:
            self.total_messages_dropped += 1
            return None

        msg = Message(
            msg_id=self.next_msg_id,
            priority=priority,
            size_kb=max(size_kb, 0.0),
            created_sol=self.sol,
            content=content,
        )
        self.next_msg_id += 1
        self.queue.append(msg)
        self.queue.sort(key=lambda m: (m.priority, m.created_sol))
        return msg

    def queue_depth(self) -> int:
        """Number of messages waiting."""
        return len(self.queue)

    def queue_size_kb(self) -> float:
        """Total size of queued messages in KB."""
        return sum(m.total_size_kb for m in self.queue)


# ---------------------------------------------------------------------------
# Per-sol tick
# ---------------------------------------------------------------------------

def tick_comm(relay: CommRelay, sol: int) -> dict:
    """Advance the communication relay by one sol.

    Returns a status dict with all relevant metrics for the sol.
    """
    relay.sol = sol

    distance_km = earth_mars_distance_km(sol)
    one_way_s = light_delay_seconds(distance_km)
    rtt_s = round_trip_delay_seconds(distance_km)
    blackout = is_conjunction(sol)
    relay.blackout = blackout
    snr = signal_to_noise_db(distance_km)

    if blackout:
        relay.blackout_sols += 1
        return {
            "sol": sol,
            "blackout": True,
            "distance_km": distance_km,
            "one_way_delay_s": one_way_s,
            "round_trip_delay_s": rtt_s,
            "snr_db": snr,
            "downlink_kbps": 0.0,
            "uplink_kbps": 0.0,
            "messages_sent": 0,
            "data_sent_kb": 0.0,
            "queue_depth": relay.queue_depth(),
            "queue_size_kb": relay.queue_size_kb(),
            "blackout_sols_total": relay.blackout_sols,
        }

    dl_rate = downlink_rate_kbps(distance_km)
    ul_rate = uplink_rate_kbps(distance_km)

    # Available bandwidth for the sol (kbps × seconds_in_sol)
    # DSN contact window is ~8 hours per sol (shared among missions)
    contact_hours = 8.0
    contact_seconds = contact_hours * 3600.0
    bandwidth_kb = dl_rate * contact_seconds / 8.0  # bits to bytes

    # Drain queue within bandwidth budget
    sent_count = 0
    sent_kb = 0.0
    remaining_kb = bandwidth_kb
    still_queued = []

    for msg in relay.queue:
        if msg.total_size_kb <= remaining_kb:
            msg.delivered = True
            msg.delivered_sol = sol
            relay.delivered.append(msg)
            sent_count += 1
            sent_kb += msg.total_size_kb
            remaining_kb -= msg.total_size_kb
        else:
            still_queued.append(msg)

    relay.queue = still_queued
    relay.total_sent_kb += sent_kb
    relay.total_messages_sent += sent_count

    return {
        "sol": sol,
        "blackout": False,
        "distance_km": distance_km,
        "one_way_delay_s": one_way_s,
        "round_trip_delay_s": rtt_s,
        "snr_db": snr,
        "downlink_kbps": dl_rate,
        "uplink_kbps": ul_rate,
        "messages_sent": sent_count,
        "data_sent_kb": sent_kb,
        "queue_depth": relay.queue_depth(),
        "queue_size_kb": relay.queue_size_kb(),
        "blackout_sols_total": relay.blackout_sols,
    }
