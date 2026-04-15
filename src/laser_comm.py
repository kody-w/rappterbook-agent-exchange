"""Mars Deep Space Optical Communications Terminal (DSOC).

Physics-grounded laser communications model for a Mars surface terminal.
Simulates Earth-Mars optical link performance including:
  - Orbital geometry (distance, solar exclusion, light delay)
  - Beam physics (diffraction, free-space loss, pointing)
  - Atmospheric extinction (Mars dust, Earth atmosphere)
  - DSOC-calibrated throughput (267 Mbps at 33M km reference)
  - Session scheduling and data volume tracking
  - Dust storm degradation model

The throughput model is calibrated to NASA's Deep Space Optical
Communications (DSOC) technology demonstrator, which achieved
267 Mbps at 33 million km during the Psyche mission (2023).

Units: SI throughout (m, s, W, Hz, rad) unless suffix says otherwise.
"""
from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any

SPEED_OF_LIGHT_M_S: float = 299_792_458.0
SPEED_OF_LIGHT_KM_S: float = SPEED_OF_LIGHT_M_S / 1000.0
AU_KM: float = 149_597_870.7

WAVELENGTH_NM: float = 1550.0
WAVELENGTH_M: float = WAVELENGTH_NM * 1e-9
PHOTON_ENERGY_J: float = 6.626e-34 * SPEED_OF_LIGHT_M_S / WAVELENGTH_M

TX_POWER_W: float = 4.0
TX_APERTURE_M: float = 0.22
TX_POINTING_ACCURACY_URAD: float = 1.0

RX_APERTURE_M: float = 5.1

BEAM_DIVERGENCE_RAD: float = 1.22 * WAVELENGTH_M / TX_APERTURE_M

DSOC_REF_RATE_MBPS: float = 267.0
DSOC_REF_DISTANCE_KM: float = 33e6

MARS_SEMI_MAJOR_AU: float = 1.524
EARTH_SEMI_MAJOR_AU: float = 1.0
SYNODIC_PERIOD_SOLS: float = 779.9

SOLAR_EXCLUSION_DEG: float = 3.0

ACQUISITION_TIME_S: float = 30.0
SESSION_HOURS_PER_SOL: float = 8.0

RF_BASELINE_MBPS: float = 2.0

CLEAR_DUST_TAU: float = 0.3
STORM_DUST_TAU_THRESHOLD: float = 3.0
DUST_STORM_SEASON_START_SOL: int = 200
DUST_STORM_SEASON_END_SOL: int = 400

EARTH_ATMO_TRANSMISSION: float = 0.85


def earth_mars_distance_km(sol: int) -> float:
    """Earth-Mars distance using circular orbit approximation."""
    phase = 2.0 * math.pi * sol / SYNODIC_PERIOD_SOLS
    r_e = EARTH_SEMI_MAJOR_AU
    r_m = MARS_SEMI_MAJOR_AU
    d_au = math.sqrt(r_e**2 + r_m**2 - 2 * r_e * r_m * math.cos(phase))
    return d_au * AU_KM


def sun_earth_probe_angle_deg(sol: int) -> float:
    """Sun-Earth-Probe angle (SEP) in degrees."""
    r_e = EARTH_SEMI_MAJOR_AU
    r_m = MARS_SEMI_MAJOR_AU
    d_em = earth_mars_distance_km(sol) / AU_KM
    if d_em < 1e-10:
        return 0.0
    cos_sep = (r_e**2 + d_em**2 - r_m**2) / (2 * r_e * d_em)
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def light_delay_seconds(distance_km: float) -> float:
    """One-way light travel time in seconds."""
    return distance_km / SPEED_OF_LIGHT_KM_S


def is_solar_exclusion(sol: int) -> bool:
    """True if link is blocked by solar exclusion (SEP < threshold)."""
    return sun_earth_probe_angle_deg(sol) < SOLAR_EXCLUSION_DEG


def beam_diameter_at_target_m(distance_km: float) -> float:
    """Beam diameter at target distance (meters)."""
    return 2.0 * distance_km * 1000.0 * math.tan(BEAM_DIVERGENCE_RAD / 2.0)


def transmitter_gain_db() -> float:
    """Transmit antenna gain (dBi) for diffraction-limited aperture."""
    gain = (math.pi * TX_APERTURE_M / WAVELENGTH_M) ** 2
    return 10.0 * math.log10(gain)


def receiver_gain_db() -> float:
    """Receive antenna gain (dBi) for ground station aperture."""
    gain = (math.pi * RX_APERTURE_M / WAVELENGTH_M) ** 2
    return 10.0 * math.log10(gain)


def free_space_loss_db(distance_km: float) -> float:
    """Free-space path loss in dB."""
    if distance_km <= 0:
        return 0.0
    d_m = distance_km * 1000.0
    loss = (4.0 * math.pi * d_m / WAVELENGTH_M) ** 2
    return 10.0 * math.log10(loss)


def mars_atmosphere_loss_db(dust_tau: float, zenith_deg: float = 30.0) -> float:
    """Mars atmospheric loss in dB from dust optical depth."""
    if dust_tau <= 0:
        return 0.0
    zenith_rad = math.radians(min(zenith_deg, 89.5))
    airmass = 1.0 / max(math.cos(zenith_rad), 0.01)
    transmission = math.exp(-dust_tau * airmass)
    if transmission <= 0:
        return 999.0
    return -10.0 * math.log10(transmission)


def mars_atmosphere_transmission(dust_tau: float, zenith_deg: float = 30.0) -> float:
    """Mars atmospheric transmission (0 to 1)."""
    if dust_tau <= 0:
        return 1.0
    zenith_rad = math.radians(min(zenith_deg, 89.5))
    airmass = 1.0 / max(math.cos(zenith_rad), 0.01)
    return math.exp(-dust_tau * airmass)


def pointing_loss_db(error_urad: float) -> float:
    """Pointing loss in dB from pointing error."""
    if error_urad <= 0:
        return 0.0
    half_beam = BEAM_DIVERGENCE_RAD * 1e6 / 2.0
    ratio = error_urad / half_beam
    return 4.343 * ratio**2


def pointing_loss_fraction(error_urad: float) -> float:
    """Pointing loss as a fraction (1.0 = perfect)."""
    loss_db = pointing_loss_db(error_urad)
    return 10.0 ** (-loss_db / 10.0)


def link_budget_db(distance_km: float, dust_tau: float = 0.3,
                   zenith_deg: float = 30.0) -> Dict[str, float]:
    """Complete link budget in dB."""
    tx_dbw = 10.0 * math.log10(TX_POWER_W)
    gt = transmitter_gain_db()
    fsl = free_space_loss_db(distance_km)
    gr = receiver_gain_db()
    atmo = mars_atmosphere_loss_db(dust_tau, zenith_deg)
    pl = pointing_loss_db(TX_POINTING_ACCURACY_URAD)
    earth_atmo = -10.0 * math.log10(EARTH_ATMO_TRANSMISSION)
    rx_dbw = tx_dbw + gt - fsl + gr - atmo - pl - earth_atmo
    return {
        "tx_power_dbw": tx_dbw,
        "tx_gain_db": gt,
        "free_space_loss_db": fsl,
        "rx_gain_db": gr,
        "mars_atmo_loss_db": atmo,
        "pointing_loss_db": pl,
        "earth_atmo_loss_db": earth_atmo,
        "received_power_dbw": rx_dbw,
    }


def received_power_w(distance_km: float, dust_tau: float = 0.3,
                     zenith_deg: float = 30.0) -> float:
    """Received optical power in watts."""
    if distance_km <= 0:
        return TX_POWER_W
    budget = link_budget_db(distance_km, dust_tau, zenith_deg)
    return 10.0 ** (budget["received_power_dbw"] / 10.0)


def received_photons_per_second(distance_km: float, dust_tau: float = 0.3,
                                zenith_deg: float = 30.0) -> float:
    """Photon arrival rate at receiver."""
    pw = received_power_w(distance_km, dust_tau, zenith_deg)
    return pw / PHOTON_ENERGY_J


def channel_capacity_mbps(photon_rate_hz: float) -> float:
    """Shannon capacity for photon-counting channel (Mbps)."""
    if photon_rate_hz <= 0:
        return 0.0
    symbol_period = 1e-9
    n_s = photon_rate_hz * symbol_period
    if n_s < 1e-15:
        return 0.0
    capacity_bps = (1.0 / symbol_period) * n_s * math.log2(1.0 + 1.0 / n_s)
    return capacity_bps / 1e6


def achievable_throughput_mbps(distance_km: float, dust_tau: float = 0.3,
                               zenith_deg: float = 30.0) -> float:
    """Achievable data rate using DSOC-calibrated reference model.

    Scales the demonstrated 267 Mbps at 33M km by inverse-square of
    distance, atmospheric transmission, pointing loss, and Earth
    atmosphere.
    """
    if distance_km <= 0:
        return 0.0
    dist_factor = (DSOC_REF_DISTANCE_KM / distance_km) ** 2
    mars_tx = mars_atmosphere_transmission(dust_tau, zenith_deg)
    point_frac = pointing_loss_fraction(TX_POINTING_ACCURACY_URAD)
    rate = DSOC_REF_RATE_MBPS * dist_factor * mars_tx * point_frac * EARTH_ATMO_TRANSMISSION
    return max(rate, 0.0)


def session_data_volume_gb(throughput_mbps: float,
                           session_hours: float = SESSION_HOURS_PER_SOL) -> float:
    """Data volume transferable in one session (GB)."""
    if throughput_mbps <= 0 or session_hours <= 0:
        return 0.0
    bits = throughput_mbps * 1e6 * session_hours * 3600.0
    return bits / (8.0 * 1e9)


def upgrade_factor(laser_mbps: float) -> float:
    """How many times faster than current RF baseline."""
    if laser_mbps <= 0:
        return 0.0
    return laser_mbps / RF_BASELINE_MBPS


def seasonal_dust_tau(sol: int) -> float:
    """Seasonal dust optical depth model."""
    mars_year_sol = sol % 669
    base = CLEAR_DUST_TAU
    if DUST_STORM_SEASON_START_SOL <= mars_year_sol <= DUST_STORM_SEASON_END_SOL:
        season_phase = (mars_year_sol - DUST_STORM_SEASON_START_SOL) / (
            DUST_STORM_SEASON_END_SOL - DUST_STORM_SEASON_START_SOL)
        seasonal_bump = 0.5 * math.sin(math.pi * season_phase)
        return base + seasonal_bump
    return base


@dataclass
class LaserTerminal:
    """State of the Mars laser communications terminal."""
    sol: int = 0
    dust_tau: float = CLEAR_DUST_TAU
    link_active: bool = False
    total_data_gb: float = 0.0
    total_sessions: int = 0
    total_blocked_sols: int = 0
    total_storm_sols: int = 0
    peak_throughput_mbps: float = 0.0
    cumulative_throughput_mbps: float = 0.0

    def status(self) -> str:
        """Current terminal status string."""
        if self.dust_tau >= STORM_DUST_TAU_THRESHOLD:
            return "dust_storm"
        if is_solar_exclusion(self.sol):
            return "solar_exclusion"
        if self.link_active:
            return "transmitting"
        return "idle"


@dataclass
class TickResult:
    """Result of one simulation tick (one sol)."""
    sol: int = 0
    distance_km: float = 0.0
    sep_angle_deg: float = 0.0
    light_delay_s: float = 0.0
    dust_tau: float = CLEAR_DUST_TAU
    link_active: bool = False
    status: str = "idle"
    throughput_mbps: float = 0.0
    capacity_mbps: float = 0.0
    photon_rate_hz: float = 0.0
    session_data_gb: float = 0.0
    total_data_gb: float = 0.0
    beam_diameter_earth_km: float = 0.0
    rf_upgrade_factor: float = 0.0
    total_sessions: int = 0
    total_blocked_sols: int = 0
    total_storm_sols: int = 0
    peak_throughput_mbps: float = 0.0


def tick(terminal, sol=None, dust_tau=None):
    """Advance terminal state by one sol."""
    if sol is None:
        sol = terminal.sol
    if dust_tau is None:
        dust_tau = seasonal_dust_tau(sol)
    dust_tau = max(dust_tau, 0.0)

    terminal.sol = sol
    terminal.dust_tau = dust_tau

    distance = earth_mars_distance_km(sol)
    sep = sun_earth_probe_angle_deg(sol)
    delay = light_delay_seconds(distance)
    beam_d_m = beam_diameter_at_target_m(distance)
    beam_d_km = beam_d_m / 1000.0

    blocked_solar = is_solar_exclusion(sol)
    blocked_storm = dust_tau >= STORM_DUST_TAU_THRESHOLD

    if blocked_solar:
        status = "solar_exclusion"
        link_active = False
    elif blocked_storm:
        status = "dust_storm"
        link_active = False
    else:
        status = "transmitting"
        link_active = True

    terminal.link_active = link_active

    if link_active:
        throughput = achievable_throughput_mbps(distance, dust_tau)
        photon_rate = received_photons_per_second(distance, dust_tau)
        capacity = channel_capacity_mbps(photon_rate)
        data_gb = session_data_volume_gb(throughput)
        rf_upgrade = upgrade_factor(throughput)
        terminal.total_data_gb += data_gb
        terminal.total_sessions += 1
        terminal.cumulative_throughput_mbps += throughput
        if throughput > terminal.peak_throughput_mbps:
            terminal.peak_throughput_mbps = throughput
    else:
        throughput = 0.0
        photon_rate = 0.0
        capacity = 0.0
        data_gb = 0.0
        rf_upgrade = 0.0
        if blocked_solar:
            terminal.total_blocked_sols += 1
        if blocked_storm:
            terminal.total_storm_sols += 1

    terminal.sol = sol + 1

    return TickResult(
        sol=sol,
        distance_km=distance,
        sep_angle_deg=sep,
        light_delay_s=delay,
        dust_tau=dust_tau,
        link_active=link_active,
        status=status,
        throughput_mbps=throughput,
        capacity_mbps=capacity,
        photon_rate_hz=photon_rate,
        session_data_gb=data_gb,
        total_data_gb=terminal.total_data_gb,
        beam_diameter_earth_km=beam_d_km,
        rf_upgrade_factor=rf_upgrade,
        total_sessions=terminal.total_sessions,
        total_blocked_sols=terminal.total_blocked_sols,
        total_storm_sols=terminal.total_storm_sols,
        peak_throughput_mbps=terminal.peak_throughput_mbps,
    )


def run_simulation(sols=668, dust_tau=None):
    """Run simulation for given number of sols."""
    terminal = LaserTerminal()
    results = []
    for s in range(sols):
        result = tick(terminal, sol=s, dust_tau=dust_tau)
        results.append(result)
    return results


def summarize(results):
    """Summarize simulation results."""
    if not results:
        return {}
    active = [r for r in results if r.link_active]
    mean_thr = (sum(r.throughput_mbps for r in active) / len(active)) if active else 0.0
    last = results[-1]
    return {
        "total_sols": len(results),
        "active_sessions": last.total_sessions,
        "blocked_sols": last.total_blocked_sols,
        "storm_sols": last.total_storm_sols,
        "total_data_gb": last.total_data_gb,
        "peak_throughput_mbps": last.peak_throughput_mbps,
        "mean_throughput_mbps": mean_thr,
        "max_rf_upgrade": max((r.rf_upgrade_factor for r in results), default=0.0),
    }


def state_to_dict(terminal):
    """Serialize terminal state to JSON-safe dict."""
    return dataclasses.asdict(terminal)


def state_from_dict(d):
    """Restore terminal state from dict."""
    return LaserTerminal(
        sol=d.get("sol", 0),
        dust_tau=d.get("dust_tau", CLEAR_DUST_TAU),
        link_active=d.get("link_active", False),
        total_data_gb=d.get("total_data_gb", 0.0),
        total_sessions=d.get("total_sessions", 0),
        total_blocked_sols=d.get("total_blocked_sols", 0),
        total_storm_sols=d.get("total_storm_sols", 0),
        peak_throughput_mbps=d.get("peak_throughput_mbps", 0.0),
        cumulative_throughput_mbps=d.get("cumulative_throughput_mbps", 0.0),
    )


if __name__ == "__main__":
    print("=== Mars DSOC Laser Communications Terminal ===\n")
    results = run_simulation(sols=668)
    summary = summarize(results)
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")
    print("\n--- Per-sol sample (every 50 sols) ---")
    for r in results[::50]:
        flag = "TX" if r.link_active else r.status[:4].upper()
        print(f"  Sol {r.sol:4d}: {r.distance_km/1e6:6.1f}M km  "
              f"{r.throughput_mbps:7.2f} Mbps  {flag}  "
              f"{r.total_data_gb:8.1f} GB total")
