"""navigation_beacon.py -- Mars Surface Positioning Beacon Network.

Mars has no GPS.  No constellation of navigation satellites.  Rovers
wander by dead reckoning and orbital imagery.  EVA crew rely on line-
of-sight to the hab.  Beyond the horizon, they are lost.

This module builds what GPS built for Earth: a network of surface
beacons that broadcast timing signals on UHF.  A receiver triangulates
its position from 3+ beacons via trilateration.  The colony finally
knows WHERE things are.

Physics modelled
----------------
* UHF signal propagation -- Line-of-sight on Mars.  Thin atmosphere
  (~600 Pa) causes negligible RF absorption.  Range limited by
  horizon: d = sqrt(2*R*h).  6 m mast -> ~6.4 km horizon.

* Trilateration -- 3+ range measurements solve for (x,y).
  Error amplified by Geometric Dilution of Precision (GDOP).

* Ranging via round-trip time -- Range = c*t/2.  10 ns timing -> ~1.5 m.

* Dust storm attenuation -- path loss proportional to optical depth tau.

* Power budget -- 50 W solar, 500 Wh LiFePO4, 2 W TX at 1% duty.

* Temperature -- LiFePO4 degrades ~1%/K below 273 K, floor 30%.

Conservation laws: error >= 0, range >= 0, battery in [0, cap],
GDOP >= 1.0, horizon increases with height, signal decreases with distance.

One tick = one sol.  Distances in metres, power in watts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


SPEED_OF_LIGHT_M_S = 299_792_458.0
MARS_RADIUS_M = 3_389_500.0
MARS_SURFACE_TEMP_K = 210.0

DEFAULT_MAST_HEIGHT_M = 6.0
DEFAULT_TX_POWER_W = 2.0
DEFAULT_TX_FREQ_MHZ = 401.0
DEFAULT_ANTENNA_GAIN_DBI = 3.0
DEFAULT_BATTERY_WH = 500.0
DEFAULT_SOLAR_PANEL_W = 50.0
PING_INTERVAL_S = 10.0
PING_DURATION_S = 0.1
TIMING_ACCURACY_NS = 10.0

RECEIVER_SENSITIVITY_DBM = -110.0

BATTERY_TEMP_COEFF = 0.01
BATTERY_MIN_CAPACITY_FRAC = 0.30
BATTERY_CYCLE_DEGRADATION = 0.0001

DUST_ATTENUATION_DB_PER_KM_PER_TAU = 0.12

HOURS_PER_SOL = 24.66
SECONDS_PER_SOL = 88_775.0
SOLAR_HOURS_PER_SOL = 12.0

MIN_BEACONS_FOR_FIX = 3
GDOP_PERFECT = 1.0


def horizon_distance_m(height_m: float) -> float:
    """Distance to radio horizon from antenna at given height (m)."""
    if height_m < 0.0:
        return 0.0
    return math.sqrt(2.0 * MARS_RADIUS_M * height_m)


def mutual_horizon_m(h1_m: float, h2_m: float) -> float:
    """Maximum line-of-sight between two antennas (m)."""
    return horizon_distance_m(h1_m) + horizon_distance_m(h2_m)


def free_space_path_loss_db(distance_km: float, freq_mhz: float) -> float:
    """Free-space path loss in dB."""
    if distance_km <= 0.0 or freq_mhz <= 0.0:
        return 0.0
    return (20.0 * math.log10(distance_km)
            + 20.0 * math.log10(freq_mhz) + 32.45)


def dust_attenuation_db(distance_km: float, optical_depth: float) -> float:
    """Additional signal loss due to dust (dB)."""
    if distance_km <= 0.0 or optical_depth <= 0.0:
        return 0.0
    return DUST_ATTENUATION_DB_PER_KM_PER_TAU * distance_km * optical_depth


def received_power_dbm(tx_power_w, tx_gain_dbi, rx_gain_dbi,
                       distance_km, freq_mhz, optical_depth=0.0):
    """Received signal power in dBm after path loss and dust."""
    if tx_power_w <= 0.0:
        return -999.0
    tx_dbm = 10.0 * math.log10(tx_power_w * 1000.0)
    fspl = free_space_path_loss_db(distance_km, freq_mhz)
    dust = dust_attenuation_db(distance_km, optical_depth)
    return tx_dbm + tx_gain_dbi + rx_gain_dbi - fspl - dust


def max_range_km(tx_power_w, tx_gain_dbi, rx_gain_dbi,
                 freq_mhz, sensitivity_dbm, optical_depth=0.0):
    """Maximum detection range (km) by binary search on link budget."""
    lo, hi = 0.001, 5000.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        rx = received_power_dbm(tx_power_w, tx_gain_dbi, rx_gain_dbi,
                                mid, freq_mhz, optical_depth)
        if rx > sensitivity_dbm:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def range_from_roundtrip_s(roundtrip_s: float) -> float:
    """Distance in metres from round-trip signal time."""
    if roundtrip_s < 0.0:
        return 0.0
    return SPEED_OF_LIGHT_M_S * roundtrip_s / 2.0


def timing_to_range_precision_m(timing_accuracy_ns: float) -> float:
    """Range precision (m) from timing accuracy (ns)."""
    return SPEED_OF_LIGHT_M_S * (timing_accuracy_ns * 1e-9) / 2.0


def geometric_dilution_of_precision(beacon_positions, receiver_pos):
    """2D GDOP.  Returns >= 1.0.  Inf if < 3 beacons or degenerate."""
    n = len(beacon_positions)
    if n < MIN_BEACONS_FOR_FIX:
        return float("inf")
    hx, hy = [], []
    rx, ry = receiver_pos
    for bx, by in beacon_positions:
        dx, dy = bx - rx, by - ry
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return float("inf")
        hx.append(dx / dist)
        hy.append(dy / dist)
    a = sum(x * x for x in hx)
    b = sum(x * y for x, y in zip(hx, hy))
    d = sum(y * y for y in hy)
    det = a * d - b * b
    if det < 1e-12:
        return float("inf")
    trace_inv = (a + d) / det
    if trace_inv < 0.0:
        return float("inf")
    return max(math.sqrt(trace_inv), GDOP_PERFECT)


def position_error_m(range_precision_m: float, gdop: float) -> float:
    """Expected position error (m) = range_precision * GDOP."""
    return abs(range_precision_m) * max(gdop, GDOP_PERFECT)


def trilaterate_2d(beacons):
    """Estimate 2D position from beacon ranges via least-squares.

    Input: list of (x, y, measured_range) for 3+ beacons.
    Returns (x, y) or None if underdetermined.
    """
    if len(beacons) < MIN_BEACONS_FOR_FIX:
        return None
    x1, y1, r1 = beacons[0]
    rows_a, rows_b = [], []
    for i in range(1, len(beacons)):
        xi, yi, ri = beacons[i]
        rows_a.append((2.0 * (xi - x1), 2.0 * (yi - y1)))
        rows_b.append(r1*r1 - ri*ri + xi*xi - x1*x1 + yi*yi - y1*y1)
    n = len(rows_a)
    ata00 = sum(a[0]*a[0] for a in rows_a)
    ata01 = sum(a[0]*a[1] for a in rows_a)
    ata11 = sum(a[1]*a[1] for a in rows_a)
    atb0 = sum(rows_a[i][0]*rows_b[i] for i in range(n))
    atb1 = sum(rows_a[i][1]*rows_b[i] for i in range(n))
    det = ata00 * ata11 - ata01 * ata01
    if abs(det) < 1e-12:
        return None
    x = (ata11 * atb0 - ata01 * atb1) / det
    y = (ata00 * atb1 - ata01 * atb0) / det
    return (x, y)


def battery_capacity_at_temp(nominal_wh, temp_k, health=1.0):
    """Effective battery capacity (Wh) at given temperature."""
    if temp_k >= 273.0:
        frac = 1.0
    else:
        frac = max(BATTERY_MIN_CAPACITY_FRAC,
                    1.0 - BATTERY_TEMP_COEFF * (273.0 - temp_k))
    return nominal_wh * frac * min(max(health, 0.0), 1.0)


def solar_energy_per_sol_wh(panel_w, optical_depth=0.3):
    """Solar energy harvested per sol (Wh).  Beer-Lambert: exp(-tau)."""
    return panel_w * math.exp(-optical_depth) * SOLAR_HOURS_PER_SOL


def tx_energy_per_sol_wh(tx_power_w, ping_interval_s=PING_INTERVAL_S,
                         ping_duration_s=PING_DURATION_S):
    """Transmit energy consumed per sol (Wh)."""
    if ping_interval_s <= 0.0:
        return 0.0
    duty = ping_duration_s / ping_interval_s
    return tx_power_w * duty * SECONDS_PER_SOL / 3600.0


def electronics_energy_per_sol_wh(idle_power_w=0.5):
    """Baseline electronics power consumption per sol (Wh)."""
    return idle_power_w * SECONDS_PER_SOL / 3600.0


@dataclass
class Beacon:
    """State of a single navigation beacon."""
    beacon_id: str = "nav-001"
    x_m: float = 0.0
    y_m: float = 0.0
    mast_height_m: float = DEFAULT_MAST_HEIGHT_M
    tx_power_w: float = DEFAULT_TX_POWER_W
    freq_mhz: float = DEFAULT_TX_FREQ_MHZ
    antenna_gain_dbi: float = DEFAULT_ANTENNA_GAIN_DBI
    battery_wh: float = DEFAULT_BATTERY_WH
    battery_capacity_wh: float = DEFAULT_BATTERY_WH
    solar_panel_w: float = DEFAULT_SOLAR_PANEL_W
    battery_health: float = 1.0
    temp_k: float = MARS_SURFACE_TEMP_K
    active: bool = True
    sol: int = 0

    def to_dict(self):
        return {
            "beacon_id": self.beacon_id, "x_m": self.x_m, "y_m": self.y_m,
            "mast_height_m": self.mast_height_m, "tx_power_w": self.tx_power_w,
            "freq_mhz": self.freq_mhz, "antenna_gain_dbi": self.antenna_gain_dbi,
            "battery_wh": round(self.battery_wh, 4),
            "battery_capacity_wh": self.battery_capacity_wh,
            "solar_panel_w": self.solar_panel_w,
            "battery_health": round(self.battery_health, 6),
            "temp_k": self.temp_k, "active": self.active, "sol": self.sol,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class NetworkState:
    """State of the entire beacon network."""
    beacons: list = field(default_factory=list)
    sol: int = 0
    total_fixes: int = 0
    total_failed_fixes: int = 0
    best_precision_m: float = float("inf")
    worst_precision_m: float = 0.0

    def to_dict(self):
        return {
            "beacons": [b.to_dict() for b in self.beacons],
            "sol": self.sol, "total_fixes": self.total_fixes,
            "total_failed_fixes": self.total_failed_fixes,
            "best_precision_m": round(self.best_precision_m, 4)
                if self.best_precision_m != float("inf") else None,
            "worst_precision_m": round(self.worst_precision_m, 4),
        }

    @classmethod
    def from_dict(cls, d):
        beacons = [Beacon.from_dict(b) for b in d.get("beacons", [])]
        return cls(beacons=beacons, sol=d.get("sol", 0),
                   total_fixes=d.get("total_fixes", 0),
                   total_failed_fixes=d.get("total_failed_fixes", 0),
                   best_precision_m=d.get("best_precision_m") or float("inf"),
                   worst_precision_m=d.get("worst_precision_m", 0.0))


@dataclass
class TickResult:
    """Result of one sol tick."""
    sol: int = 0
    active_beacons: int = 0
    beacons_with_power: int = 0
    network_fix_available: bool = False
    estimated_precision_m: float = float("inf")
    gdop: float = float("inf")
    solar_harvest_wh: float = 0.0
    tx_consumed_wh: float = 0.0
    net_energy_wh: float = 0.0


def create_beacon(beacon_id="nav-001", x_m=0.0, y_m=0.0,
                  mast_height_m=DEFAULT_MAST_HEIGHT_M):
    """Factory: create a new beacon with defaults."""
    return Beacon(beacon_id=beacon_id, x_m=x_m, y_m=y_m,
                  mast_height_m=mast_height_m)


def create_network(beacon_configs=None):
    """Factory: create a beacon network (default: 4-beacon diamond)."""
    if beacon_configs is None:
        beacon_configs = [
            {"beacon_id": "nav-N", "x_m": 0.0, "y_m": 5000.0},
            {"beacon_id": "nav-S", "x_m": 0.0, "y_m": -5000.0},
            {"beacon_id": "nav-E", "x_m": 5000.0, "y_m": 0.0},
            {"beacon_id": "nav-W", "x_m": -5000.0, "y_m": 0.0},
        ]
    return NetworkState(beacons=[create_beacon(**c) for c in beacon_configs])


def tick(state, optical_depth=0.3, temp_k=MARS_SURFACE_TEMP_K,
         receiver_pos=(0.0, 0.0), receiver_height_m=2.0):
    """Advance the beacon network by one sol."""
    state.sol += 1
    result = TickResult(sol=state.sol)
    visible_beacons = []

    for beacon in state.beacons:
        beacon.sol = state.sol
        beacon.temp_k = temp_k
        solar_wh = solar_energy_per_sol_wh(beacon.solar_panel_w, optical_depth)
        tx_wh = tx_energy_per_sol_wh(beacon.tx_power_w)
        elec_wh = electronics_energy_per_sol_wh()
        total_consumed = tx_wh + elec_wh

        if beacon.active:
            beacon.battery_wh -= total_consumed
        beacon.battery_wh += solar_wh

        beacon.battery_health = max(0.5,
            beacon.battery_health - BATTERY_CYCLE_DEGRADATION)
        effective_cap = battery_capacity_at_temp(
            beacon.battery_capacity_wh, temp_k, beacon.battery_health)
        beacon.battery_wh = max(0.0, min(beacon.battery_wh, effective_cap))

        if beacon.battery_wh <= 0.1:
            beacon.active = False
        if not beacon.active and beacon.battery_wh > effective_cap * 0.2:
            beacon.active = True
        if not beacon.active:
            continue

        result.active_beacons += 1
        los = mutual_horizon_m(beacon.mast_height_m, receiver_height_m)
        dx = beacon.x_m - receiver_pos[0]
        dy = beacon.y_m - receiver_pos[1]
        distance_m = math.sqrt(dx * dx + dy * dy)
        if distance_m > los:
            continue
        distance_km = distance_m / 1000.0
        rx_power = received_power_dbm(
            beacon.tx_power_w, beacon.antenna_gain_dbi,
            DEFAULT_ANTENNA_GAIN_DBI, distance_km, beacon.freq_mhz,
            optical_depth)
        if rx_power >= RECEIVER_SENSITIVITY_DBM:
            visible_beacons.append((beacon.x_m, beacon.y_m))
            result.beacons_with_power += 1

    if len(visible_beacons) >= MIN_BEACONS_FOR_FIX:
        gdop = geometric_dilution_of_precision(visible_beacons, receiver_pos)
        range_prec = timing_to_range_precision_m(TIMING_ACCURACY_NS)
        precision = position_error_m(range_prec, gdop)
        result.network_fix_available = True
        result.estimated_precision_m = precision
        result.gdop = gdop
        state.total_fixes += 1
        state.best_precision_m = min(state.best_precision_m, precision)
        state.worst_precision_m = max(state.worst_precision_m, precision)
    else:
        state.total_failed_fixes += 1

    result.solar_harvest_wh = solar_energy_per_sol_wh(
        DEFAULT_SOLAR_PANEL_W, optical_depth)
    result.tx_consumed_wh = (tx_energy_per_sol_wh(DEFAULT_TX_POWER_W)
                             + electronics_energy_per_sol_wh())
    result.net_energy_wh = result.solar_harvest_wh - result.tx_consumed_wh
    return result


def run_simulation(state, sols=100, optical_depth=0.3,
                   temp_k=MARS_SURFACE_TEMP_K):
    """Run the beacon network for multiple sols."""
    return [tick(state, optical_depth, temp_k) for _ in range(sols)]
