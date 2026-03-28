"""laser_comm.py -- Mars Deep Space Optical Communications Terminal.

The colony has RF at 2 Mbps (comm_relay.py).  That was fine for
telemetry.  But the colony is growing -- it has medical imaging,
science data, construction blueprints, and 120+ colonists who
want to video-call Earth.  RF cannot scale.

This module models a laser communications terminal based on NASA DSOC
(Deep Space Optical Communications), which demonstrated 267 Mbps from
33 million km in December 2023.  A Mars-class terminal uses a 22 cm
flight laser transmitter, 1550 nm near-IR laser, and pulse-position
modulation detected by superconducting nanowire single-photon detectors.

Physics modelled
----------------
* Laser transmitter: 1550 nm, 4 W optical, 22 cm aperture.
  Beam divergence = 1.22*lambda/D ~ 8.6 urad.
* Free-space link budget: Pr = Pt*Gt*Gr*(lambda/4piR)^2.
* Photon-counting detection: SNSPD, QE ~90%, dark counts ~100 cps.
* Achievable data rate: calibrated against DSOC (267 Mbps at 33M km).
  Rate scales as 1/R^2, modified by atmosphere and pointing losses.
* Mars atmosphere: dust optical depth tau 0.3-4.0.
* Solar exclusion zone: link blocked when SEP < 3 deg.

Conservation laws
-----------------
- Energy: received_power <= transmitted_power
- Information: throughput <= channel capacity
- Photon budget: rx_photons <= tx_photons
- Data volume: monotonically non-decreasing

One tick = one sol.  Power in W, distance in km, rates in Mbps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# -- Physical constants -------------------------------------------------------

SPEED_OF_LIGHT_M_S = 299_792_458.0
SPEED_OF_LIGHT_KM_S = 299_792.458
PLANCK_J_S = 6.626e-34

WAVELENGTH_M = 1.550e-6
WAVELENGTH_NM = 1550.0
PHOTON_ENERGY_J = PLANCK_J_S * SPEED_OF_LIGHT_M_S / WAVELENGTH_M

# -- Transmitter (DSOC-class Mars terminal) -----------------------------------

TX_APERTURE_M = 0.22
TX_POWER_W = 4.0
TX_EFFICIENCY = 0.85
TX_POINTING_ACCURACY_URAD = 1.0
BEAM_DIVERGENCE_RAD = 1.22 * WAVELENGTH_M / TX_APERTURE_M

# -- Receiver (Earth ground station) ------------------------------------------

RX_APERTURE_M = 5.0
RX_EFFICIENCY = 0.50
DETECTOR_QE = 0.90
DARK_COUNT_RATE_CPS = 100.0
EARTH_ATMO_TRANSMISSION = 0.70

# -- DSOC calibration (demonstrated performance) ------------------------------

DSOC_REF_RATE_MBPS = 267.0
DSOC_REF_DISTANCE_KM = 33.0e6
DSOC_MIN_RATE_MBPS = 0.01

# -- Mars atmosphere -----------------------------------------------------------

NOMINAL_DUST_TAU = 0.5
STORM_DUST_TAU = 4.0
CLEAR_DUST_TAU = 0.3
DEFAULT_ZENITH_ANGLE_DEG = 30.0

# -- Orbital geometry ----------------------------------------------------------

EARTH_ORBIT_KM = 149_597_870.7
MARS_ORBIT_KM = 227_939_200.0
SYNODIC_PERIOD_SOLS = 764.0
AU_KM = 149_597_870.7
SOLAR_EXCLUSION_DEG = 3.0

# -- Session -------------------------------------------------------------------

LINK_MARGIN_DB = 3.0
POINTING_LOSS_DB = 1.5
ACQUISITION_TIME_S = 30.0
SESSION_HOURS_PER_SOL = 6.0
SECONDS_PER_SOL = 88_775.0

RF_BASELINE_MBPS = 2.0


# == Orbital geometry =========================================================

def earth_mars_distance_km(sol: int) -> float:
    """Earth-Mars distance (km) from simplified circular-orbit model."""
    phase = (sol % SYNODIC_PERIOD_SOLS) / SYNODIC_PERIOD_SOLS * 2 * math.pi
    angle = phase + math.pi
    d_sq = (EARTH_ORBIT_KM ** 2 + MARS_ORBIT_KM ** 2
            - 2 * EARTH_ORBIT_KM * MARS_ORBIT_KM * math.cos(angle))
    return math.sqrt(max(d_sq, 0.0))


def sun_earth_probe_angle_deg(sol: int) -> float:
    """Sun-Earth-Probe angle (degrees).  ~0 at conjunction, ~180 at opposition."""
    phase = (sol % SYNODIC_PERIOD_SOLS) / SYNODIC_PERIOD_SOLS
    return 180.0 * math.sin(math.pi * phase)


def light_delay_seconds(distance_km: float) -> float:
    """One-way light travel time (seconds)."""
    return distance_km / SPEED_OF_LIGHT_KM_S


def is_solar_exclusion(sol: int) -> bool:
    """True if the Sun blocks the laser line of sight."""
    return sun_earth_probe_angle_deg(sol) < SOLAR_EXCLUSION_DEG


# == Beam physics =============================================================

def beam_diameter_at_target_m(distance_km: float) -> float:
    """Laser beam diameter (m) at the target, diffraction-limited."""
    return 2.0 * BEAM_DIVERGENCE_RAD * distance_km * 1000.0


def transmitter_gain_db() -> float:
    """Transmitter antenna gain (dB)."""
    g = (math.pi * TX_APERTURE_M / WAVELENGTH_M) ** 2
    return 10.0 * math.log10(max(g, 1.0))


def receiver_gain_db() -> float:
    """Receiver antenna gain (dB)."""
    g = (math.pi * RX_APERTURE_M / WAVELENGTH_M) ** 2
    return 10.0 * math.log10(max(g, 1.0))


def free_space_loss_db(distance_km: float) -> float:
    """Free-space path loss (dB).  L = (4*pi*R/lambda)^2."""
    if distance_km <= 0:
        return 0.0
    distance_m = distance_km * 1000.0
    l = (4.0 * math.pi * distance_m / WAVELENGTH_M) ** 2
    return 10.0 * math.log10(max(l, 1.0))


def mars_atmosphere_loss_db(dust_tau: float, zenith_deg: float) -> float:
    """Mars atmospheric extinction (dB)."""
    if dust_tau <= 0:
        return 0.0
    z_rad = math.radians(max(0.0, min(zenith_deg, 89.0)))
    cos_z = max(math.cos(z_rad), 0.01)
    transmission = math.exp(-dust_tau / cos_z)
    return -10.0 * math.log10(max(transmission, 1e-30))


def mars_atmosphere_transmission(dust_tau: float, zenith_deg: float) -> float:
    """Mars atmospheric transmission fraction [0, 1]."""
    if dust_tau <= 0:
        return 1.0
    z_rad = math.radians(max(0.0, min(zenith_deg, 89.0)))
    cos_z = max(math.cos(z_rad), 0.01)
    return math.exp(-dust_tau / cos_z)


def pointing_loss_db(error_urad: float) -> float:
    """Pointing loss (dB) from angular misalignment."""
    if error_urad <= 0:
        return 0.0
    beam_urad = BEAM_DIVERGENCE_RAD * 1e6
    ratio = error_urad / beam_urad
    loss_linear = math.exp(-2.0 * ratio ** 2)
    return -10.0 * math.log10(max(loss_linear, 1e-30))


def pointing_loss_fraction(error_urad: float) -> float:
    """Pointing loss as a fraction [0, 1].  1.0 = no loss."""
    if error_urad <= 0:
        return 1.0
    beam_urad = BEAM_DIVERGENCE_RAD * 1e6
    ratio = error_urad / beam_urad
    return math.exp(-2.0 * ratio ** 2)


# == Link budget ==============================================================

def link_budget_db(distance_km: float, dust_tau: float = NOMINAL_DUST_TAU,
                   zenith_deg: float = DEFAULT_ZENITH_ANGLE_DEG) -> dict:
    """Complete optical link budget (all values in dB or dBW)."""
    tx_power_dbw = 10.0 * math.log10(max(TX_POWER_W, 1e-30))
    tx_eff_db = 10.0 * math.log10(max(TX_EFFICIENCY, 1e-30))
    g_tx = transmitter_gain_db()
    fsl = free_space_loss_db(distance_km)
    g_rx = receiver_gain_db()
    rx_eff_db = 10.0 * math.log10(max(RX_EFFICIENCY, 1e-30))
    earth_atmo_db = -10.0 * math.log10(max(EARTH_ATMO_TRANSMISSION, 1e-30))
    mars_atmo_db = mars_atmosphere_loss_db(dust_tau, zenith_deg)
    pt_loss_db = pointing_loss_db(TX_POINTING_ACCURACY_URAD)

    received_dbw = (tx_power_dbw + tx_eff_db + g_tx
                    - fsl + g_rx + rx_eff_db
                    - earth_atmo_db - mars_atmo_db
                    - pt_loss_db - LINK_MARGIN_DB)

    return {
        "tx_power_dbw": round(tx_power_dbw, 2),
        "tx_efficiency_db": round(tx_eff_db, 2),
        "tx_gain_db": round(g_tx, 2),
        "free_space_loss_db": round(fsl, 2),
        "rx_gain_db": round(g_rx, 2),
        "rx_efficiency_db": round(rx_eff_db, 2),
        "earth_atmo_loss_db": round(earth_atmo_db, 2),
        "mars_atmo_loss_db": round(mars_atmo_db, 2),
        "pointing_loss_db": round(pt_loss_db, 2),
        "link_margin_db": LINK_MARGIN_DB,
        "received_power_dbw": round(received_dbw, 2),
    }


def received_power_w(distance_km: float, dust_tau: float = NOMINAL_DUST_TAU,
                     zenith_deg: float = DEFAULT_ZENITH_ANGLE_DEG) -> float:
    """Received optical power in watts at the Earth ground station."""
    budget = link_budget_db(distance_km, dust_tau, zenith_deg)
    return 10.0 ** (budget["received_power_dbw"] / 10.0)


def received_photons_per_second(distance_km: float,
                                dust_tau: float = NOMINAL_DUST_TAU,
                                zenith_deg: float = DEFAULT_ZENITH_ANGLE_DEG
                                ) -> float:
    """Detected photon rate at Earth station (photons/sec)."""
    pr = received_power_w(distance_km, dust_tau, zenith_deg)
    return pr / PHOTON_ENERGY_J * DETECTOR_QE


# == Achievable data rate (DSOC-calibrated) ===================================

def achievable_throughput_mbps(distance_km: float,
                               dust_tau: float = NOMINAL_DUST_TAU,
                               zenith_deg: float = DEFAULT_ZENITH_ANGLE_DEG
                               ) -> float:
    """End-to-end achievable throughput (Mbps).

    Calibrated against DSOC: 267 Mbps at 33M km (Dec 2023).
    Rate scales as 1/R^2, modified by atmosphere and pointing.
    """
    if distance_km <= 0:
        return 0.0
    base_rate = DSOC_REF_RATE_MBPS * (DSOC_REF_DISTANCE_KM / distance_km) ** 2
    mars_trans = mars_atmosphere_transmission(dust_tau, zenith_deg)
    pt_frac = pointing_loss_fraction(TX_POINTING_ACCURACY_URAD)
    rate = base_rate * mars_trans * pt_frac * EARTH_ATMO_TRANSMISSION
    return max(rate, DSOC_MIN_RATE_MBPS) if rate > DSOC_MIN_RATE_MBPS else 0.0


def channel_capacity_mbps(photon_rate: float) -> float:
    """Information-theoretic capacity (Mbps) from photon rate.

    Uses photon information efficiency (PIE) model: ~1.5 bits/photon.
    """
    if photon_rate <= 0:
        return 0.0
    pie = 1.5
    if photon_rate > 1e9:
        pie = 1.0
    elif photon_rate < 1e4:
        pie = 3.0
    return photon_rate * pie / 1e6


# == Session data volume ======================================================

def session_data_volume_gb(throughput_mbps: float,
                           session_hours: float = SESSION_HOURS_PER_SOL
                           ) -> float:
    """Data transferred in one optical session (gigabytes)."""
    if throughput_mbps <= 0 or session_hours <= 0:
        return 0.0
    effective_seconds = session_hours * 3600.0 - ACQUISITION_TIME_S
    effective_seconds = max(effective_seconds, 0.0)
    bits = throughput_mbps * 1e6 * effective_seconds
    return bits / 8.0 / 1e9


def upgrade_factor(laser_mbps: float) -> float:
    """How many times faster the laser link is vs RF baseline."""
    if laser_mbps <= 0:
        return 0.0
    return laser_mbps / RF_BASELINE_MBPS


# == Dust model ===============================================================

def seasonal_dust_tau(sol: int) -> float:
    """Mars seasonal dust optical depth.  Peaks around Ls~250."""
    mars_year_sols = 668.6
    ls_phase = (sol % mars_year_sols) / mars_year_sols * 2 * math.pi
    dust_season_offset = 250.0 / 360.0 * 2 * math.pi
    seasonal = 0.55 + 0.25 * math.sin(ls_phase - dust_season_offset)
    return max(CLEAR_DUST_TAU, min(seasonal, 1.5))


# == State ====================================================================

@dataclass
class LaserTerminal:
    """Mars optical communications terminal state."""
    sol: int = 0
    distance_km: float = 0.0
    sep_angle_deg: float = 180.0
    dust_tau: float = NOMINAL_DUST_TAU
    zenith_deg: float = DEFAULT_ZENITH_ANGLE_DEG
    link_active: bool = False
    throughput_mbps: float = 0.0
    photon_rate_hz: float = 0.0
    total_data_gb: float = 0.0
    total_sessions: int = 0
    total_blocked_sols: int = 0
    total_storm_sols: int = 0
    peak_throughput_mbps: float = 0.0
    received_power_dbw: float = -999.0
    free_space_loss_db_val: float = 0.0

    def status(self) -> str:
        if is_solar_exclusion(self.sol):
            return "solar_exclusion"
        if self.dust_tau >= STORM_DUST_TAU:
            return "dust_storm"
        if self.dust_tau >= 2.0:
            return "degraded"
        if not self.link_active:
            return "idle"
        return "transmitting"


@dataclass
class TickResult:
    """Output from one sol of laser terminal operation."""
    sol: int
    status: str
    distance_km: float
    light_delay_s: float
    sep_angle_deg: float
    dust_tau: float
    link_active: bool
    throughput_mbps: float
    session_data_gb: float
    photon_rate_hz: float
    received_power_dbw: float
    rf_upgrade_factor: float
    beam_diameter_earth_km: float
    capacity_mbps: float
    total_data_gb: float
    total_sessions: int


# == Tick engine ==============================================================

def tick(terminal, sol=None, dust_tau=None):
    """Advance the laser terminal by one sol."""
    if sol is not None:
        terminal.sol = sol
    current_sol = terminal.sol

    terminal.distance_km = earth_mars_distance_km(current_sol)
    terminal.sep_angle_deg = sun_earth_probe_angle_deg(current_sol)
    delay_s = light_delay_seconds(terminal.distance_km)

    blocked_by_sun = terminal.sep_angle_deg < SOLAR_EXCLUSION_DEG

    if dust_tau is not None:
        terminal.dust_tau = max(0.0, dust_tau)
    else:
        terminal.dust_tau = seasonal_dust_tau(current_sol)

    blocked_by_dust = terminal.dust_tau >= STORM_DUST_TAU

    if blocked_by_sun or blocked_by_dust:
        terminal.link_active = False
        terminal.throughput_mbps = 0.0
        terminal.photon_rate_hz = 0.0
        terminal.received_power_dbw = -999.0
        terminal.free_space_loss_db_val = free_space_loss_db(terminal.distance_km)
        capacity = 0.0
        if blocked_by_sun:
            terminal.total_blocked_sols += 1
        if blocked_by_dust:
            terminal.total_storm_sols += 1
    else:
        budget = link_budget_db(terminal.distance_km, terminal.dust_tau,
                                terminal.zenith_deg)
        terminal.received_power_dbw = budget["received_power_dbw"]
        terminal.free_space_loss_db_val = budget["free_space_loss_db"]
        terminal.photon_rate_hz = received_photons_per_second(
            terminal.distance_km, terminal.dust_tau, terminal.zenith_deg)
        terminal.throughput_mbps = achievable_throughput_mbps(
            terminal.distance_km, terminal.dust_tau, terminal.zenith_deg)
        capacity = channel_capacity_mbps(terminal.photon_rate_hz)
        terminal.link_active = terminal.throughput_mbps > 0

    if terminal.link_active:
        session_gb = session_data_volume_gb(terminal.throughput_mbps)
        terminal.total_data_gb += session_gb
        terminal.total_sessions += 1
        if terminal.throughput_mbps > terminal.peak_throughput_mbps:
            terminal.peak_throughput_mbps = terminal.throughput_mbps
    else:
        session_gb = 0.0

    beam_km = beam_diameter_at_target_m(terminal.distance_km) / 1000.0

    result = TickResult(
        sol=current_sol,
        status=terminal.status(),
        distance_km=round(terminal.distance_km, 1),
        light_delay_s=round(delay_s, 2),
        sep_angle_deg=round(terminal.sep_angle_deg, 2),
        dust_tau=round(terminal.dust_tau, 4),
        link_active=terminal.link_active,
        throughput_mbps=round(terminal.throughput_mbps, 3),
        session_data_gb=round(session_gb, 4),
        photon_rate_hz=round(terminal.photon_rate_hz, 1),
        received_power_dbw=round(terminal.received_power_dbw, 2),
        rf_upgrade_factor=round(upgrade_factor(terminal.throughput_mbps), 1),
        beam_diameter_earth_km=round(beam_km, 1),
        capacity_mbps=round(capacity, 3),
        total_data_gb=round(terminal.total_data_gb, 4),
        total_sessions=terminal.total_sessions,
    )

    terminal.sol += 1
    return result


# == Multi-sol simulation =====================================================

def run_simulation(sols=668, dust_tau=None):
    """Run the laser terminal for *sols* ticks (default = 1 Mars year)."""
    terminal = LaserTerminal()
    results = []
    for s in range(sols):
        results.append(tick(terminal, sol=s, dust_tau=dust_tau))
    return results


def summarize(results):
    """Summarize a simulation run."""
    if not results:
        return {}
    active = [r for r in results if r.link_active]
    throughputs = [r.throughput_mbps for r in active] if active else [0.0]
    return {
        "total_sols": len(results),
        "active_sessions": len(active),
        "blocked_sols": sum(1 for r in results if r.status == "solar_exclusion"),
        "storm_sols": sum(1 for r in results if r.status == "dust_storm"),
        "total_data_gb": round(results[-1].total_data_gb, 2),
        "peak_throughput_mbps": round(max(throughputs), 3),
        "mean_throughput_mbps": round(sum(throughputs) / len(throughputs), 3),
        "min_throughput_mbps": round(min(throughputs), 3),
        "max_rf_upgrade_factor": round(
            max(r.rf_upgrade_factor for r in active) if active else 0.0, 1),
    }


# == Serialisation ============================================================

def state_to_dict(terminal):
    """Serialise terminal state for JSON persistence."""
    return {
        "sol": terminal.sol,
        "distance_km": terminal.distance_km,
        "sep_angle_deg": terminal.sep_angle_deg,
        "dust_tau": terminal.dust_tau,
        "zenith_deg": terminal.zenith_deg,
        "link_active": terminal.link_active,
        "throughput_mbps": terminal.throughput_mbps,
        "photon_rate_hz": terminal.photon_rate_hz,
        "total_data_gb": terminal.total_data_gb,
        "total_sessions": terminal.total_sessions,
        "total_blocked_sols": terminal.total_blocked_sols,
        "total_storm_sols": terminal.total_storm_sols,
        "peak_throughput_mbps": terminal.peak_throughput_mbps,
        "received_power_dbw": terminal.received_power_dbw,
        "free_space_loss_db": terminal.free_space_loss_db_val,
    }


def state_from_dict(d):
    """Deserialise terminal state from a JSON-loaded dict."""
    t = LaserTerminal()
    t.sol = int(d.get("sol", 0))
    t.distance_km = float(d.get("distance_km", 0))
    t.sep_angle_deg = float(d.get("sep_angle_deg", 180))
    t.dust_tau = float(d.get("dust_tau", NOMINAL_DUST_TAU))
    t.zenith_deg = float(d.get("zenith_deg", DEFAULT_ZENITH_ANGLE_DEG))
    t.link_active = bool(d.get("link_active", False))
    t.throughput_mbps = float(d.get("throughput_mbps", 0))
    t.photon_rate_hz = float(d.get("photon_rate_hz", 0))
    t.total_data_gb = float(d.get("total_data_gb", 0))
    t.total_sessions = int(d.get("total_sessions", 0))
    t.total_blocked_sols = int(d.get("total_blocked_sols", 0))
    t.total_storm_sols = int(d.get("total_storm_sols", 0))
    t.peak_throughput_mbps = float(d.get("peak_throughput_mbps", 0))
    t.received_power_dbw = float(d.get("received_power_dbw", -999))
    t.free_space_loss_db_val = float(d.get("free_space_loss_db", 0))
    return t
