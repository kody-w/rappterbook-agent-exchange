"""water_ice_radar.py - Mars Ground-Penetrating Radar for Subsurface Ice Detection.

Before the colony can mine water (water_mining.py), it must FIND it.
This module simulates a Ground-Penetrating Radar (GPR) system that
detects subsurface water ice deposits beneath the Martian regolith.

Physics modelled
----------------
* Radar propagation - EM waves travel through regolith at v = c / sqrt(epsilon_r).
* Two-way travel time - Reflection from depth d returns after t = 2d / v.
* Signal attenuation - P_received = P_tx * exp(-2 * alpha * d)
  where alpha = (pi * f * tan_delta * sqrt(epsilon_r)) / c
* Detection threshold - Ice detected when SNR > 10 dB.
* Ice dielectric contrast - Water ice (eps ~3.15) vs regolith (eps ~3.0).
  R = (sqrt(eps1) - sqrt(eps2))^2 / (sqrt(eps1) + sqrt(eps2))^2
* Depth resolution - delta_d = c / (2 * B * sqrt(epsilon_r)).
* Antenna degradation - Dust accumulation reduces efficiency over time.

Conservation laws: received <= transmitted, depth >= 0, SNR monotonically
decreases with depth, antenna efficiency in [0,1], power >= 0.

Reference: SHARAD (MRO), MARSIS (Mars Express), RIMFAX (Perseverance).
One tick = one sol. Depth in metres. Power in watts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


SPEED_OF_LIGHT_M_S = 2.998e8
REGOLITH_EPSILON_R = 3.0
REGOLITH_LOSS_TANGENT = 0.05
ICE_EPSILON_R = 3.15
ICE_LOSS_TANGENT = 0.0001
DEFAULT_CENTER_FREQ_HZ = 20.0e6
DEFAULT_BANDWIDTH_HZ = 10.0e6
DEFAULT_TRANSMIT_POWER_W = 10.0
DEFAULT_ANTENNA_GAIN_DB = 6.0
NOISE_FLOOR_W = 1.0e-8
SNR_MIN_DB = 10.0
MAX_DETECTABLE_DEPTH_M = 200.0
RADAR_POWER_DRAW_W = 25.0
SCANS_PER_SOL = 100
DUST_DEGRADATION_PER_SOL = 0.001
DUST_CLEANING_RESTORATION = 0.90
SOL_HOURS = 24.66


def signal_speed(epsilon_r: float) -> float:
    """v = c / sqrt(epsilon_r). Returns m/s."""
    if epsilon_r < 1.0:
        raise ValueError(f"epsilon_r must be >= 1.0, got {epsilon_r}")
    return SPEED_OF_LIGHT_M_S / math.sqrt(epsilon_r)


def two_way_travel_time(depth_m: float, epsilon_r: float) -> float:
    """Round-trip time t = 2d/v (seconds)."""
    if depth_m < 0.0:
        raise ValueError(f"depth must be >= 0, got {depth_m}")
    return 2.0 * depth_m / signal_speed(epsilon_r)


def estimate_depth(travel_time_s: float, epsilon_r: float) -> float:
    """d = t * v / 2 (metres). Inverse of two_way_travel_time."""
    if travel_time_s < 0.0:
        raise ValueError(f"travel_time must be >= 0, got {travel_time_s}")
    return travel_time_s * signal_speed(epsilon_r) / 2.0


def attenuation_coefficient(freq_hz: float, epsilon_r: float,
                             loss_tangent: float) -> float:
    """alpha = (pi * f * tan_delta * sqrt(eps)) / c (Nepers/m)."""
    if freq_hz <= 0.0:
        raise ValueError(f"freq must be > 0, got {freq_hz}")
    if loss_tangent < 0.0:
        raise ValueError(f"loss_tangent must be >= 0, got {loss_tangent}")
    return (math.pi * freq_hz * loss_tangent * math.sqrt(epsilon_r)
            / SPEED_OF_LIGHT_M_S)


def signal_power_at_depth(transmit_power_w: float, depth_m: float,
                          freq_hz: float, epsilon_r: float,
                          loss_tangent: float,
                          antenna_efficiency: float = 1.0) -> float:
    """P(d) = P_tx * eff * exp(-2 * alpha * d)."""
    if transmit_power_w < 0.0:
        raise ValueError(f"transmit_power must be >= 0, got {transmit_power_w}")
    if depth_m < 0.0:
        raise ValueError(f"depth must be >= 0, got {depth_m}")
    antenna_efficiency = max(0.0, min(1.0, antenna_efficiency))
    alpha = attenuation_coefficient(freq_hz, epsilon_r, loss_tangent)
    return transmit_power_w * antenna_efficiency * math.exp(-2.0 * alpha * depth_m)


def reflection_coefficient(eps1: float, eps2: float) -> float:
    """R = ((sqrt(eps1)-sqrt(eps2))/(sqrt(eps1)+sqrt(eps2)))^2."""
    if eps1 < 1.0 or eps2 < 1.0:
        raise ValueError(f"epsilon values must be >= 1.0, got {eps1}, {eps2}")
    n1, n2 = math.sqrt(eps1), math.sqrt(eps2)
    return ((n1 - n2) / (n1 + n2)) ** 2


def snr_db(received_power_w: float, noise_power_w: float) -> float:
    """SNR = 10 * log10(P_signal / P_noise) in dB."""
    if noise_power_w <= 0.0:
        raise ValueError(f"noise_power must be > 0, got {noise_power_w}")
    if received_power_w <= 0.0:
        return -math.inf
    return 10.0 * math.log10(received_power_w / noise_power_w)


def depth_resolution(bandwidth_hz: float, epsilon_r: float) -> float:
    """delta_d = c / (2 * B * sqrt(eps)) in metres."""
    if bandwidth_hz <= 0.0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth_hz}")
    if epsilon_r < 1.0:
        raise ValueError(f"epsilon_r must be >= 1.0, got {epsilon_r}")
    return SPEED_OF_LIGHT_M_S / (2.0 * bandwidth_hz * math.sqrt(epsilon_r))


def detect_ice_at_depth(depth_m: float, ice_thickness_m: float,
                        transmit_power_w: float = DEFAULT_TRANSMIT_POWER_W,
                        freq_hz: float = DEFAULT_CENTER_FREQ_HZ,
                        antenna_efficiency: float = 1.0) -> dict:
    """Simulate detection of an ice layer at a given depth."""
    if depth_m < 0.0:
        depth_m = 0.0
    if ice_thickness_m < 0.0:
        ice_thickness_m = 0.0
    p_at_ice = signal_power_at_depth(
        transmit_power_w, depth_m, freq_hz,
        REGOLITH_EPSILON_R, REGOLITH_LOSS_TANGENT, antenna_efficiency)
    r_coeff = reflection_coefficient(REGOLITH_EPSILON_R, ICE_EPSILON_R)
    p_reflected = p_at_ice * r_coeff
    snr_val = snr_db(p_reflected, NOISE_FLOOR_W)
    travel_t = two_way_travel_time(depth_m, REGOLITH_EPSILON_R)
    est_depth = estimate_depth(travel_t, REGOLITH_EPSILON_R)
    return {
        "detected": snr_val >= SNR_MIN_DB,
        "snr_db": snr_val,
        "estimated_depth_m": round(est_depth, 3),
        "ice_thickness_m": ice_thickness_m,
        "signal_power_w": p_reflected,
        "reflection_coefficient": r_coeff,
    }


def max_detection_depth(transmit_power_w: float = DEFAULT_TRANSMIT_POWER_W,
                        freq_hz: float = DEFAULT_CENTER_FREQ_HZ,
                        antenna_efficiency: float = 1.0) -> float:
    """Max depth where ice is detectable (binary search)."""
    r_coeff = reflection_coefficient(REGOLITH_EPSILON_R, ICE_EPSILON_R)
    lo, hi = 0.0, 1000.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        p = signal_power_at_depth(transmit_power_w, mid, freq_hz,
                                  REGOLITH_EPSILON_R, REGOLITH_LOSS_TANGENT,
                                  antenna_efficiency)
        s = snr_db(p * r_coeff, NOISE_FLOOR_W)
        if s > SNR_MIN_DB:
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


@dataclass
class ScanResult:
    """A single radar scan result at one surface position."""
    x_m: float
    y_m: float
    ice_detected: bool
    ice_depth_m: float
    ice_thickness_m: float
    snr_db: float
    sol: int

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict."""
        return {"x_m": self.x_m, "y_m": self.y_m,
                "ice_detected": self.ice_detected,
                "ice_depth_m": round(self.ice_depth_m, 3),
                "ice_thickness_m": round(self.ice_thickness_m, 3),
                "snr_db": round(self.snr_db, 2), "sol": self.sol}

    @classmethod
    def from_dict(cls, d):
        """Deserialise from dict."""
        return cls(x_m=float(d["x_m"]), y_m=float(d["y_m"]),
                   ice_detected=bool(d["ice_detected"]),
                   ice_depth_m=float(d["ice_depth_m"]),
                   ice_thickness_m=float(d["ice_thickness_m"]),
                   snr_db=float(d["snr_db"]), sol=int(d["sol"]))


@dataclass
class IceRadar:
    """Colony GPR system for subsurface ice prospecting."""
    transmit_power_w: float = DEFAULT_TRANSMIT_POWER_W
    freq_hz: float = DEFAULT_CENTER_FREQ_HZ
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ
    antenna_efficiency: float = 1.0
    power_draw_w: float = RADAR_POWER_DRAW_W
    sol: int = 0
    total_scans: int = 0
    scan_results: list = field(default_factory=list)

    def __post_init__(self):
        """Clamp fields to physical ranges."""
        self.transmit_power_w = max(0.0, self.transmit_power_w)
        self.freq_hz = max(1.0, self.freq_hz)
        self.bandwidth_hz = max(1.0, self.bandwidth_hz)
        self.antenna_efficiency = max(0.0, min(1.0, self.antenna_efficiency))
        self.power_draw_w = max(0.0, self.power_draw_w)
        self.sol = max(0, self.sol)
        self.total_scans = max(0, self.total_scans)

    def scan(self, x_m, y_m, ice_depth_m, ice_thickness_m):
        """Perform a radar scan at position (x, y)."""
        result = detect_ice_at_depth(
            ice_depth_m, ice_thickness_m,
            self.transmit_power_w, self.freq_hz, self.antenna_efficiency)
        sr = ScanResult(
            x_m=x_m, y_m=y_m,
            ice_detected=result["detected"],
            ice_depth_m=result["estimated_depth_m"] if result["detected"] else 0.0,
            ice_thickness_m=ice_thickness_m if result["detected"] else 0.0,
            snr_db=result["snr_db"], sol=self.sol)
        self.scan_results.append(sr)
        self.total_scans += 1
        return sr

    def energy_per_sol_kwh(self):
        """Energy consumed per sol in kWh."""
        return self.power_draw_w * SOL_HOURS / 1000.0

    def detection_range_m(self):
        """Current maximum detection depth given system state."""
        return max_detection_depth(self.transmit_power_w, self.freq_hz,
                                   self.antenna_efficiency)

    def resolution_m(self):
        """Current depth resolution in metres."""
        return depth_resolution(self.bandwidth_hz, REGOLITH_EPSILON_R)

    def ice_deposit_count(self):
        """Number of unique positions where ice was detected."""
        seen = set()
        for s in self.scan_results:
            if s.ice_detected:
                seen.add((s.x_m, s.y_m))
        return len(seen)

    def clean_antenna(self):
        """Manual antenna cleaning -- restores efficiency."""
        restored = (1.0 - self.antenna_efficiency) * DUST_CLEANING_RESTORATION
        self.antenna_efficiency = min(1.0, self.antenna_efficiency + restored)

    def tick(self):
        """Advance one sol. Applies dust degradation to antenna."""
        self.sol += 1
        self.antenna_efficiency = max(
            0.0, self.antenna_efficiency - DUST_DEGRADATION_PER_SOL)
        return {
            "sol": self.sol,
            "antenna_efficiency": round(self.antenna_efficiency, 6),
            "detection_range_m": round(self.detection_range_m(), 2),
            "resolution_m": round(self.resolution_m(), 3),
            "total_scans": self.total_scans,
            "ice_deposits_found": self.ice_deposit_count(),
            "energy_kwh": round(self.energy_per_sol_kwh(), 3),
        }

    def to_dict(self):
        """Serialise full state to JSON-safe dict."""
        return {
            "transmit_power_w": self.transmit_power_w,
            "freq_hz": self.freq_hz, "bandwidth_hz": self.bandwidth_hz,
            "antenna_efficiency": round(self.antenna_efficiency, 6),
            "power_draw_w": self.power_draw_w, "sol": self.sol,
            "total_scans": self.total_scans,
            "scan_results": [s.to_dict() for s in self.scan_results],
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialise from dict."""
        scans = [ScanResult.from_dict(s) for s in d.get("scan_results", [])]
        return cls(
            transmit_power_w=float(d.get("transmit_power_w", DEFAULT_TRANSMIT_POWER_W)),
            freq_hz=float(d.get("freq_hz", DEFAULT_CENTER_FREQ_HZ)),
            bandwidth_hz=float(d.get("bandwidth_hz", DEFAULT_BANDWIDTH_HZ)),
            antenna_efficiency=float(d.get("antenna_efficiency", 1.0)),
            power_draw_w=float(d.get("power_draw_w", RADAR_POWER_DRAW_W)),
            sol=int(d.get("sol", 0)),
            total_scans=int(d.get("total_scans", 0)),
            scan_results=scans)
