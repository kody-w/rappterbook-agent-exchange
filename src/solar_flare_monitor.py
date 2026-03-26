"""solar_flare_monitor.py -- Mars Colony Solar Particle Event Early Warning

Mars has no global magnetic field.  When the Sun erupts, energetic protons
(10 MeV-1 GeV) cross the interplanetary void and slam into anything
unshielded on the surface.  A Carrington-class event delivers a lethal
dose in hours.  The colony survives on MINUTES of warning -- enough to
halt EVAs, seal hatches, and shelter crew.

Physics
-------
* Solar proton events (SPE): CMEs accelerate protons to relativistic
  energies.  Fastest (>100 MeV) arrive in ~15 min at 1 AU; Mars at
  ~1.52 AU gets ~23-90 min transit depending on energy.
* X-ray precursor: Flares emit X-rays at light speed (8-12 min to Mars).
  X-ray spike is the first warning.
* Proton fluence to dose: D_rate ~ pfu * S_p / rho.  Calibrated to
  MAVEN/RAD measurements of the Sept 2017 event at Mars.
* NOAA S-scale: S1 (>10 pfu) to S5 (>100k pfu).
* Shelter: 2m regolith or 50cm water reduces SPE protons by >90%.

Reference: Curiosity RAD 0.67 mSv/sol GCR; MAVEN Sept 2017 SPE ~50 mSv;
  Oct 2003 Halloween storms S4 ~1e4 pfu; Carrington 1859 est. S5+.

One tick = one sol.  Fluxes in pfu, doses in mSv, times in minutes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# -- Physical constants -------------------------------------------------------

SPEED_OF_LIGHT_KM_S = 299_792.458
PROTON_MASS_KG = 1.6726e-27
MEV_TO_JOULE = 1.6022e-13
AU_KM = 149_597_870.7
MARS_AU = 1.524
MARS_DISTANCE_KM = MARS_AU * AU_KM

LIGHT_TIME_SUN_MARS_S = MARS_DISTANCE_KM / SPEED_OF_LIGHT_KM_S
LIGHT_TIME_SUN_MARS_MIN = LIGHT_TIME_SUN_MARS_S / 60.0  # ~12.7 min

GCR_BASELINE_MSV_SOL = 0.67   # Curiosity RAD
PFU_TO_MSV_HR = 0.005          # ~5 uSv/hr per pfu, conservative
SHELTER_ATTENUATION = 0.08     # 2m regolith: SPE dose to ~8%


# -- NOAA S-Scale classification ----------------------------------------------

S_SCALE: Dict[str, Dict[str, float]] = {
    "S1": {"min_pfu": 10,      "max_pfu": 100,       "label": "Minor"},
    "S2": {"min_pfu": 100,     "max_pfu": 1_000,     "label": "Moderate"},
    "S3": {"min_pfu": 1_000,   "max_pfu": 10_000,    "label": "Strong"},
    "S4": {"min_pfu": 10_000,  "max_pfu": 100_000,   "label": "Severe"},
    "S5": {"min_pfu": 100_000, "max_pfu": 1_000_000, "label": "Extreme"},
}

ALERT_WATCH_PFU = 5.0
ALERT_WARNING_PFU = 10.0
ALERT_CRITICAL_PFU = 1000.0
ALERT_EMERGENCY_PFU = 10000.0


# -- SPE probability model (events per sol by solar cycle phase) --------------

SPE_RATES_PER_SOL: Dict[str, Dict[str, float]] = {
    "solar_max": {
        "S1": 10.0 / 668.6, "S2": 4.0 / 668.6, "S3": 2.0 / 668.6,
        "S4": 0.3 / 668.6,  "S5": 0.05 / 668.6,
    },
    "solar_min": {
        "S1": 2.0 / 668.6, "S2": 0.8 / 668.6, "S3": 0.3 / 668.6,
        "S4": 0.05 / 668.6, "S5": 0.005 / 668.6,
    },
}


# -- Data structures ----------------------------------------------------------

@dataclass
class SolarEvent:
    """A single solar particle event."""
    sol_detected: int
    peak_pfu: float
    duration_hours: float
    xray_class: str
    s_scale: str
    proton_energy_mev: float
    transit_time_min: float
    total_fluence: float
    dose_unshielded_msv: float
    dose_sheltered_msv: float
    warning_time_min: float


@dataclass
class MonitorState:
    """Solar flare monitoring system state."""
    sol: int = 0
    alert_level: str = "nominal"
    current_pfu: float = 0.0
    background_pfu: float = 0.1
    cumulative_dose_msv: float = 0.0
    sheltered_dose_msv: float = 0.0
    events_detected: int = 0
    active_event: Optional[SolarEvent] = None
    event_history: List[SolarEvent] = field(default_factory=list)
    shelter_active: bool = False
    shelter_activations: int = 0
    false_alarms: int = 0
    power_consumed_kwh: float = 0.0
    sols_at_elevated: int = 0


@dataclass
class SolReport:
    """Per-sol monitoring results."""
    sol: int
    alert_level: str
    current_pfu: float
    dose_msv: float
    cumulative_dose_msv: float
    shelter_active: bool
    event_detected: Optional[SolarEvent]
    warning_time_min: float
    power_kwh: float
    alerts: List[str]


# -- Pure functions -----------------------------------------------------------

def proton_velocity_km_s(energy_mev: float) -> float:
    """Non-relativistic proton velocity from kinetic energy.

    v = sqrt(2E/m_p).  Capped at speed of light.
    """
    if energy_mev <= 0.0:
        return 0.0
    energy_j = energy_mev * MEV_TO_JOULE
    v_m_s = math.sqrt(2.0 * energy_j / PROTON_MASS_KG)
    return min(v_m_s / 1000.0, SPEED_OF_LIGHT_KM_S)


def proton_transit_time_min(energy_mev: float, distance_km: float) -> float:
    """Transit time for protons from Sun to Mars in minutes."""
    if energy_mev <= 0.0:
        return float("inf")
    v_km_s = proton_velocity_km_s(energy_mev)
    if v_km_s <= 0.0:
        return float("inf")
    return distance_km / v_km_s / 60.0


def warning_lead_time_min(proton_energy_mev: float) -> float:
    """Warning time = proton transit - X-ray transit (light speed)."""
    proton_transit = proton_transit_time_min(proton_energy_mev, MARS_DISTANCE_KM)
    return max(0.0, proton_transit - LIGHT_TIME_SUN_MARS_MIN)


def classify_s_scale(peak_pfu: float) -> str:
    """Classify event on NOAA S-scale from peak proton flux."""
    if peak_pfu >= S_SCALE["S5"]["min_pfu"]:
        return "S5"
    if peak_pfu >= S_SCALE["S4"]["min_pfu"]:
        return "S4"
    if peak_pfu >= S_SCALE["S3"]["min_pfu"]:
        return "S3"
    if peak_pfu >= S_SCALE["S2"]["min_pfu"]:
        return "S2"
    if peak_pfu >= S_SCALE["S1"]["min_pfu"]:
        return "S1"
    return "none"


def dose_rate_msv_hr(pfu: float) -> float:
    """Unshielded surface dose rate from proton flux."""
    return max(0.0, pfu * PFU_TO_MSV_HR)


def event_total_dose_msv(peak_pfu: float, duration_hours: float) -> float:
    """Total unshielded dose (triangular flux profile)."""
    return peak_pfu * 0.5 * PFU_TO_MSV_HR * duration_hours


def sheltered_dose_msv(unshielded_msv: float) -> float:
    """Dose received inside storm shelter."""
    return unshielded_msv * SHELTER_ATTENUATION


def alert_from_pfu(pfu: float) -> str:
    """Determine alert level from current proton flux."""
    if pfu >= ALERT_EMERGENCY_PFU:
        return "emergency"
    if pfu >= ALERT_CRITICAL_PFU:
        return "critical"
    if pfu >= ALERT_WARNING_PFU:
        return "warning"
    if pfu >= ALERT_WATCH_PFU:
        return "watch"
    return "nominal"


def should_shelter(alert_level: str) -> bool:
    """Crew should be in storm shelter at critical or above."""
    return alert_level in ("critical", "emergency")


def monitor_power_kwh(sol_hours: float = 24.66) -> float:
    """Power consumption for monitoring instrument suite (~10 W)."""
    return 0.01 * sol_hours


def generate_spe(
    sol: int,
    s_class: str,
    rng: Optional[random.Random] = None,
) -> SolarEvent:
    """Generate a synthetic SPE of given S-class."""
    r = rng or random.Random()
    bounds = S_SCALE[s_class]
    peak_pfu = r.uniform(bounds["min_pfu"], bounds["max_pfu"])

    base_hours = {"S1": 4, "S2": 8, "S3": 16, "S4": 24, "S5": 48}
    dur = base_hours.get(s_class, 8) * r.uniform(0.5, 1.5)

    energy_map = {"S1": 30, "S2": 50, "S3": 80, "S4": 120, "S5": 200}
    energy = energy_map.get(s_class, 50) * r.uniform(0.7, 1.3)

    transit = proton_transit_time_min(energy, MARS_DISTANCE_KM)
    warning = warning_lead_time_min(energy)
    total_dose = event_total_dose_msv(peak_pfu, dur)
    shelt_dose = sheltered_dose_msv(total_dose)
    fluence = peak_pfu * 0.5 * dur * 3600.0

    xray_map = {"S1": "C", "S2": "M", "S3": "M", "S4": "X", "S5": "X"}
    xray = xray_map.get(s_class, "M")

    return SolarEvent(
        sol_detected=sol, peak_pfu=peak_pfu, duration_hours=dur,
        xray_class=xray, s_scale=s_class, proton_energy_mev=energy,
        transit_time_min=transit, total_fluence=fluence,
        dose_unshielded_msv=total_dose, dose_sheltered_msv=shelt_dose,
        warning_time_min=warning,
    )


def check_for_spe(
    sol: int,
    solar_phase: str = "solar_max",
    rng: Optional[random.Random] = None,
) -> Optional[SolarEvent]:
    """Roll for an SPE this sol based on solar cycle phase."""
    r = rng or random.Random()
    rates = SPE_RATES_PER_SOL.get(solar_phase, SPE_RATES_PER_SOL["solar_max"])

    for s_class in ("S5", "S4", "S3", "S2", "S1"):
        rate = rates[s_class]
        if r.random() < rate:
            return generate_spe(sol, s_class, r)
    return None


# -- Tick function ------------------------------------------------------------

def tick_monitor(
    state: MonitorState,
    solar_phase: str = "solar_max",
    rng: Optional[random.Random] = None,
) -> SolReport:
    """Advance the solar flare monitor by one sol."""
    state.sol += 1
    alerts: List[str] = []
    r = rng or random.Random()

    new_event = check_for_spe(state.sol, solar_phase, r)
    event_this_sol: Optional[SolarEvent] = None

    if new_event is not None:
        state.events_detected += 1
        state.active_event = new_event
        state.event_history.append(new_event)
        event_this_sol = new_event
        state.current_pfu = new_event.peak_pfu
        label = S_SCALE[new_event.s_scale]["label"]
        alerts.append(
            "SPE DETECTED: " + new_event.s_scale + " (" + label + "), "
            + "peak " + str(int(new_event.peak_pfu)) + " pfu, "
            + "warning " + str(int(new_event.warning_time_min)) + " min"
        )
    elif state.active_event is not None:
        evt = state.active_event
        sols_into = state.sol - evt.sol_detected
        dur_sols = evt.duration_hours / 24.66
        if sols_into > dur_sols:
            decay = sols_into - dur_sols
            state.current_pfu = (
                evt.peak_pfu * 0.1 * math.exp(-decay * 2.0)
                + state.background_pfu
            )
            if state.current_pfu < state.background_pfu * 1.5:
                state.current_pfu = state.background_pfu
                state.active_event = None
        else:
            frac = sols_into / max(0.01, dur_sols)
            if frac < 0.5:
                state.current_pfu = evt.peak_pfu * (frac / 0.5)
            else:
                state.current_pfu = evt.peak_pfu * (1.0 - (frac - 0.5) / 0.5)
            state.current_pfu = max(state.background_pfu, state.current_pfu)
    else:
        state.current_pfu = state.background_pfu * r.uniform(0.5, 2.0)

    state.alert_level = alert_from_pfu(state.current_pfu)

    need_shelter = should_shelter(state.alert_level)
    if need_shelter and not state.shelter_active:
        state.shelter_active = True
        state.shelter_activations += 1
        alerts.append("SHELTER ACTIVATED -- all crew to storm shelter")
    elif not need_shelter and state.shelter_active:
        state.shelter_active = False
        alerts.append("All clear -- shelter deactivated")

    sol_dose_rate = dose_rate_msv_hr(state.current_pfu)
    sol_hours = 24.66
    sol_dose_unshielded = sol_dose_rate * sol_hours

    if state.shelter_active:
        sol_dose = sheltered_dose_msv(sol_dose_unshielded)
        state.sheltered_dose_msv += sol_dose
    else:
        sol_dose = sol_dose_unshielded
    sol_dose += GCR_BASELINE_MSV_SOL
    state.cumulative_dose_msv += sol_dose

    if state.current_pfu > state.background_pfu * 3.0:
        state.sols_at_elevated += 1

    power = monitor_power_kwh()
    state.power_consumed_kwh += power

    warning = 0.0
    if event_this_sol is not None:
        warning = event_this_sol.warning_time_min

    return SolReport(
        sol=state.sol, alert_level=state.alert_level,
        current_pfu=state.current_pfu, dose_msv=sol_dose,
        cumulative_dose_msv=state.cumulative_dose_msv,
        shelter_active=state.shelter_active,
        event_detected=event_this_sol, warning_time_min=warning,
        power_kwh=power, alerts=alerts,
    )


# -- Factory functions --------------------------------------------------------

def make_monitor() -> MonitorState:
    """Create a fresh solar flare monitoring system."""
    return MonitorState()


def run_monitor(
    sols: int = 365,
    solar_phase: str = "solar_max",
    seed: Optional[int] = None,
) -> List[SolReport]:
    """Run the solar flare monitor for N sols."""
    monitor = make_monitor()
    rng = random.Random(seed)
    reports: List[SolReport] = []

    for _ in range(sols):
        report = tick_monitor(monitor, solar_phase, rng)
        reports.append(report)

    return reports
