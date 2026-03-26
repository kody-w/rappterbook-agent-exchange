"""emergency_beacon.py -- Mars Surface Emergency Locator Beacon.

Every crew member, rover, and remote outpost carries an emergency beacon.
When activated, it broadcasts a coded signal on 401.585625 MHz (UHF Mars
relay frequency) to be picked up by orbiting relay satellites or the base
station.  The colony cannot perform search-and-rescue without this.

One tick = one sol.  Power in watts, range in km, frequency in MHz.

Physics modelled
----------------
* **Free-space path loss (FSPL)** -- signal attenuation over distance.
  Mars atmosphere is thin (~600 Pa) so RF absorption is negligible
  compared to Earth.  FSPL = 20*log10(d) + 20*log10(f) + 32.45 dB
  where d is km and f is MHz.
* **Link budget** -- transmit power + antenna gain - path loss vs
  receiver sensitivity.  If received power > sensitivity, signal is
  detected.  Mars orbiters have ~-130 dBm sensitivity (MRO).
* **Power budget** -- beacon runs on a small LiFePO4 battery with
  optional solar trickle charging.  Battery capacity degrades in Mars
  cold (~-60 C average).  Beacon can operate in low-power (ping every
  60s) or high-power (continuous) mode.
* **Temperature effects** -- LiFePO4 loses ~1% capacity per degree
  below 0 C.  At -60 C, battery retains ~40% of rated capacity.
  Heating pad can mitigate but costs power.
* **Orbital geometry** -- relay satellites (e.g. MRO, MAVEN, TGO)
  have limited visibility windows.  Average Mars orbiter pass lasts
  ~8 minutes every ~2 hours.  Beacon must transmit during windows.
* **Position encoding** -- beacon transmits lat/lon/altitude encoded
  in the signal.  Accuracy limited by GPS-less Mars: ~100m from
  inertial navigation or ~10m with terrain-relative nav.

Conservation laws enforced:
  - Energy consumed <= energy available (battery + solar)
  - Signal strength obeys inverse-square law
  - Battery charge never negative, never exceeds capacity
  - Temperature bounded by physics

Reference:
  - NASA EPIRB: 406 MHz, 5W, 48-hour battery, -20 to +55 C
  - Cospas-Sarsat: 406.025 MHz, detectable from LEO
  - MRO Electra relay: UHF 401.585625 MHz, -130 dBm sensitivity
  - Mars atmosphere: ~600 Pa, minimal RF absorption at UHF
  - LiFePO4: 3.2V nominal, ~170 Wh/kg, excellent cycle life
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any


# -- Physical constants -------------------------------------------------------

SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_J_K = 1.380649e-23

# Mars environment
MARS_SURFACE_TEMP_K = 210.0       # average surface temperature
MARS_SURFACE_PRESSURE_PA = 610.0  # average surface pressure
MARS_RADIUS_KM = 3389.5           # mean radius

# Beacon RF
BEACON_FREQ_MHZ = 401.585625      # UHF Mars relay frequency
BEACON_BANDWIDTH_HZ = 10_000.0    # signal bandwidth

# Default beacon hardware
DEFAULT_TX_POWER_W = 5.0          # transmit power (similar to EPIRB)
DEFAULT_TX_POWER_DBM = 10.0 * math.log10(DEFAULT_TX_POWER_W * 1000)  # ~37 dBm
DEFAULT_ANTENNA_GAIN_DBI = 2.0    # small omnidirectional antenna
DEFAULT_BATTERY_WH = 200.0         # LiFePO4 battery capacity
DEFAULT_SOLAR_W = 2.0             # small solar panel trickle charge

# Receiver (orbital relay)
ORBITAL_SENSITIVITY_DBM = -130.0  # MRO Electra-class sensitivity
ORBITAL_ALTITUDE_KM = 300.0       # typical relay orbit altitude
ORBITAL_PASS_DURATION_S = 480.0   # ~8 minutes per pass
ORBITAL_PASS_INTERVAL_S = 7200.0  # ~2 hours between passes

# Temperature effects on battery
BATTERY_TEMP_COEFF = 0.01         # 1% capacity loss per K below 273K
BATTERY_MIN_CAPACITY_FRAC = 0.20  # absolute floor at extreme cold
BATTERY_HEATER_W = 0.3            # heater power to prevent freezing
BATTERY_HEATER_THRESHOLD_K = 253.0  # activate heater below -20C

# Beacon modes
PING_INTERVAL_S = 60.0            # low-power: one ping per minute
PING_DURATION_S = 0.5             # each ping transmission length
CONTINUOUS_DUTY_CYCLE = 1.0       # high-power: always on
PING_DUTY_CYCLE = PING_DURATION_S / PING_INTERVAL_S  # ~0.83%

# Degradation
BATTERY_CYCLE_DEGRADATION = 0.0002  # capacity loss per full charge cycle
BATTERY_MIN_HEALTH = 0.50          # minimum before replacement needed
ANTENNA_DEGRADATION_PER_SOL = 0.0001  # dust accumulation on antenna

# Sol timing
SECONDS_PER_SOL = 88_775.0        # Mars sol in seconds
HOURS_PER_SOL = 24.66             # Mars sol in Earth hours


# -- RF Physics ---------------------------------------------------------------

def free_space_path_loss_db(distance_km: float, freq_mhz: float) -> float:
    """Free-space path loss in dB.

    FSPL = 20*log10(d_km) + 20*log10(f_MHz) + 32.45
    Returns 0 for zero or negative distance.
    """
    if distance_km <= 0.0 or freq_mhz <= 0.0:
        return 0.0
    return (20.0 * math.log10(distance_km)
            + 20.0 * math.log10(freq_mhz)
            + 32.45)


def received_power_dbm(
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    distance_km: float,
    freq_mhz: float,
) -> float:
    """Received signal power via link budget equation.

    P_rx = P_tx + G_tx + G_rx - FSPL
    Returns -999.0 for zero/negative distance (no signal).
    """
    if distance_km <= 0.0:
        return -999.0
    fspl = free_space_path_loss_db(distance_km, freq_mhz)
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl


def max_range_km(
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    rx_sensitivity_dbm: float,
    freq_mhz: float,
) -> float:
    """Maximum detection range where received power = sensitivity.

    Derived from FSPL equation:
    d = 10^((P_tx + G_tx + G_rx - S_rx - 32.45 - 20*log10(f)) / 20)
    """
    if freq_mhz <= 0.0:
        return 0.0
    link_margin = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - rx_sensitivity_dbm
    exponent = (link_margin - 32.45 - 20.0 * math.log10(freq_mhz)) / 20.0
    return 10.0 ** exponent


def battery_capacity_at_temp(rated_wh: float, temp_k: float) -> float:
    """Effective battery capacity adjusted for temperature.

    LiFePO4 loses ~1% per K below 273K.  Floors at BATTERY_MIN_CAPACITY_FRAC.
    """
    if temp_k >= 273.0:
        return rated_wh
    loss_fraction = min(1.0 - BATTERY_MIN_CAPACITY_FRAC,
                        BATTERY_TEMP_COEFF * (273.0 - temp_k))
    return rated_wh * max(BATTERY_MIN_CAPACITY_FRAC, 1.0 - loss_fraction)


# -- Data structures ----------------------------------------------------------

@dataclass
class BeaconState:
    """State of a Mars emergency locator beacon."""

    # Configuration
    tx_power_w: float = DEFAULT_TX_POWER_W
    antenna_gain_dbi: float = DEFAULT_ANTENNA_GAIN_DBI
    battery_capacity_wh: float = DEFAULT_BATTERY_WH
    solar_panel_w: float = DEFAULT_SOLAR_W

    # Operating state
    is_active: bool = False
    mode: str = "ping"              # "ping" or "continuous"
    battery_charge_wh: float = DEFAULT_BATTERY_WH  # starts full
    temp_k: float = MARS_SURFACE_TEMP_K
    antenna_health: float = 1.0
    battery_health: float = 1.0     # capacity degradation factor

    # Position
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    altitude_m: float = 0.0

    # Tracking
    sol: int = 0
    total_transmissions: int = 0
    total_energy_used_wh: float = 0.0
    total_detections: int = 0
    events: List[str] = field(default_factory=list)

    def tx_power_dbm(self) -> float:
        """Current transmit power in dBm, adjusted for antenna health."""
        base_dbm = 10.0 * math.log10(max(self.tx_power_w * 1000, 0.001))
        return base_dbm + self.antenna_gain_dbi * self.antenna_health

    def effective_capacity_wh(self) -> float:
        """Battery capacity adjusted for temperature and health."""
        temp_cap = battery_capacity_at_temp(self.battery_capacity_wh, self.temp_k)
        return temp_cap * self.battery_health

    def estimated_runtime_hours(self) -> float:
        """Estimated remaining runtime at current mode."""
        if not self.is_active:
            return float("inf")
        duty = PING_DUTY_CYCLE if self.mode == "ping" else CONTINUOUS_DUTY_CYCLE
        power_draw_w = self.tx_power_w * duty + (
            BATTERY_HEATER_W if self.temp_k < BATTERY_HEATER_THRESHOLD_K else 0.0
        )
        if power_draw_w <= 0:
            return float("inf")
        return self.battery_charge_wh / power_draw_w

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "tx_power_w": round(self.tx_power_w, 3),
            "antenna_gain_dbi": round(self.antenna_gain_dbi, 2),
            "battery_capacity_wh": round(self.battery_capacity_wh, 3),
            "solar_panel_w": round(self.solar_panel_w, 3),
            "is_active": self.is_active,
            "mode": self.mode,
            "battery_charge_wh": round(self.battery_charge_wh, 4),
            "temp_k": round(self.temp_k, 2),
            "antenna_health": round(self.antenna_health, 4),
            "battery_health": round(self.battery_health, 4),
            "latitude_deg": round(self.latitude_deg, 6),
            "longitude_deg": round(self.longitude_deg, 6),
            "altitude_m": round(self.altitude_m, 2),
            "sol": self.sol,
            "total_transmissions": self.total_transmissions,
            "total_energy_used_wh": round(self.total_energy_used_wh, 4),
            "total_detections": self.total_detections,
            "events": list(self.events),
        }


# -- Tick engine --------------------------------------------------------------

def tick(
    state: BeaconState,
    activate: bool = False,
    deactivate: bool = False,
    mode: str | None = None,
    ambient_temp_k: float = MARS_SURFACE_TEMP_K,
    solar_available: bool = True,
    orbital_passes: int = 12,
    receiver_sensitivity_dbm: float = ORBITAL_SENSITIVITY_DBM,
    receiver_gain_dbi: float = 3.0,
    receiver_distance_km: float = ORBITAL_ALTITUDE_KM,
) -> Dict[str, Any]:
    """Advance the beacon one sol.

    Parameters
    ----------
    state : BeaconState
        Mutable beacon state.
    activate : bool
        Activate the beacon this sol.
    deactivate : bool
        Deactivate the beacon this sol.
    mode : str or None
        Switch mode to "ping" or "continuous" (None = no change).
    ambient_temp_k : float
        Environmental temperature this sol.
    solar_available : bool
        Whether solar trickle charging is available.
    orbital_passes : int
        Number of relay satellite passes this sol (~12 typical).
    receiver_sensitivity_dbm : float
        Orbital receiver sensitivity.
    receiver_gain_dbi : float
        Orbital receiver antenna gain.
    receiver_distance_km : float
        Distance to orbital receiver.

    Returns
    -------
    dict with sol results.
    """
    state.sol += 1
    state.events = []
    state.temp_k = ambient_temp_k

    # -- Activation / deactivation ----------------------------------------
    if activate and not state.is_active:
        state.is_active = True
        state.events.append("ACTIVATED")
    if deactivate and state.is_active:
        state.is_active = False
        state.events.append("DEACTIVATED")
    if mode in ("ping", "continuous"):
        if mode != state.mode:
            state.events.append(f"MODE_{mode.upper()}")
        state.mode = mode

    # -- Solar charging (even when inactive) ------------------------------
    solar_wh = 0.0
    if solar_available and state.solar_panel_w > 0:
        # Assume ~6 hours effective sunlight per sol on Mars
        solar_wh = state.solar_panel_w * 6.0
        effective_cap = state.effective_capacity_wh()
        state.battery_charge_wh = min(
            effective_cap,
            state.battery_charge_wh + solar_wh,
        )

    # -- If inactive, just track solar and return -------------------------
    if not state.is_active:
        state.antenna_health = max(0.5, state.antenna_health - ANTENNA_DEGRADATION_PER_SOL * 0.1)
        return _make_result(state, 0.0, 0, 0, solar_wh)

    # -- Power consumption ------------------------------------------------
    duty = PING_DUTY_CYCLE if state.mode == "ping" else CONTINUOUS_DUTY_CYCLE
    tx_energy_wh = state.tx_power_w * duty * HOURS_PER_SOL

    heater_energy_wh = 0.0
    if state.temp_k < BATTERY_HEATER_THRESHOLD_K:
        heater_energy_wh = BATTERY_HEATER_W * HOURS_PER_SOL
        state.events.append("HEATER_ON")

    total_energy_wh = tx_energy_wh + heater_energy_wh

    # Check battery
    if total_energy_wh > state.battery_charge_wh:
        # Partial operation — scale transmissions proportionally
        scale = state.battery_charge_wh / total_energy_wh if total_energy_wh > 0 else 0.0
        tx_energy_wh *= scale
        heater_energy_wh *= scale
        total_energy_wh = state.battery_charge_wh
        state.events.append("LOW_BATTERY")

    state.battery_charge_wh = max(0.0, state.battery_charge_wh - total_energy_wh)
    state.total_energy_used_wh += total_energy_wh

    # Auto-shutdown on empty battery
    if state.battery_charge_wh <= 0.0:
        state.is_active = False
        state.events.append("BATTERY_DEAD")

    # -- Transmission count -----------------------------------------------
    if state.mode == "ping":
        transmissions = int(SECONDS_PER_SOL / PING_INTERVAL_S)  # ~1479 pings/sol
    else:
        transmissions = int(SECONDS_PER_SOL)  # continuous = 1 per second (conceptual)

    # Scale by battery available
    if total_energy_wh < tx_energy_wh + heater_energy_wh:
        transmissions = int(transmissions * (total_energy_wh / max(0.001, tx_energy_wh + heater_energy_wh)))

    state.total_transmissions += transmissions

    # -- Detection by orbital relay ---------------------------------------
    tx_dbm = state.tx_power_dbm()
    p_rx = received_power_dbm(tx_dbm, 0.0, receiver_gain_dbi,
                               receiver_distance_km, BEACON_FREQ_MHZ)
    detectable = p_rx >= receiver_sensitivity_dbm
    detections = orbital_passes if detectable else 0
    state.total_detections += detections

    if detectable:
        state.events.append("DETECTED")
    else:
        state.events.append("NOT_DETECTED")

    # -- Degradation ------------------------------------------------------
    state.antenna_health = max(0.5, state.antenna_health - ANTENNA_DEGRADATION_PER_SOL)
    if state.is_active:
        state.battery_health = max(
            BATTERY_MIN_HEALTH,
            state.battery_health - BATTERY_CYCLE_DEGRADATION,
        )

    return _make_result(state, total_energy_wh, transmissions, detections, solar_wh)


def _make_result(
    state: BeaconState,
    energy_wh: float,
    transmissions: int,
    detections: int,
    solar_wh: float,
) -> Dict[str, Any]:
    """Build standardized result dict."""
    return {
        "sol": state.sol,
        "is_active": state.is_active,
        "mode": state.mode,
        "battery_charge_wh": round(state.battery_charge_wh, 4),
        "battery_pct": round(100.0 * state.battery_charge_wh / max(0.01, state.effective_capacity_wh()), 1),
        "energy_used_wh": round(energy_wh, 4),
        "solar_charged_wh": round(solar_wh, 4),
        "transmissions": transmissions,
        "detections": detections,
        "detectable": detections > 0,
        "antenna_health": round(state.antenna_health, 4),
        "battery_health": round(state.battery_health, 4),
        "temp_k": round(state.temp_k, 2),
        "estimated_runtime_hours": round(state.estimated_runtime_hours(), 2),
        "events": list(state.events),
    }


def create_beacon(
    lat: float = 0.0,
    lon: float = 0.0,
    alt: float = 0.0,
) -> BeaconState:
    """Create a new emergency beacon at the given position."""
    return BeaconState(latitude_deg=lat, longitude_deg=lon, altitude_m=alt)
