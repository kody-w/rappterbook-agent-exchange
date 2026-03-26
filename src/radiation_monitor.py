"""radiation_monitor.py -- Mars Surface Radiation Monitoring and Crew Dosimetry.

Mars has no global magnetic field and an atmosphere only ~0.6 percent as thick
as Earth's.  The surface receives ~0.64 mSv/day from galactic cosmic rays
(Curiosity/RAD, Hassler et al. 2014) -- roughly 100x Earth sea-level dose.

Each tick = 1 sol (~24.66 hours).

Physics: GCR (solar-cycle modulated), SPE (Poisson + Gaussian profile),
exponential shielding attenuation, crew dosimetry with NASA STD-3001 limits,
silicon detector degradation.
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SECONDS_PER_SOL = 88775.0
HOURS_PER_SOL = SECONDS_PER_SOL / 3600.0
GCR_BASELINE_MSV_DAY = 0.64
GCR_SOLAR_MIN_FACTOR = 1.25
GCR_SOLAR_MAX_FACTOR = 0.75
SOLAR_CYCLE_SOLS = 11 * 668.6
SPE_PROBABILITY_PER_SOL_AVG = 0.003
SPE_PROBABILITY_SOLAR_MAX_MULT = 4.0
SPE_DOSE_MIN_MSV = 5.0
SPE_DOSE_MAX_MSV = 500.0
SPE_DURATION_HOURS_MIN = 2.0
SPE_DURATION_HOURS_MAX = 48.0
ATTEN_LENGTH_POLYETHYLENE = 35.0
ATTEN_LENGTH_REGOLITH = 25.0
ATTEN_LENGTH_WATER = 40.0
ATTEN_LENGTH_ALUMINUM = 20.0
DENSITY_POLYETHYLENE = 0.95
DENSITY_REGOLITH = 1.5
DENSITY_WATER = 1.0
DENSITY_ALUMINUM = 2.7
QUALITY_FACTOR_GCR = 3.0
QUALITY_FACTOR_SPE = 1.5
QUALITY_FACTOR_GAMMA = 1.0
CAREER_LIMIT_MSV = 600.0
ANNUAL_LIMIT_MSV = 500.0
THIRTY_DAY_LIMIT_MSV = 250.0
ALERT_FRACTION = 0.80
DETECTOR_POWER_WATTS = 4.0

@dataclass
class CrewDosimetry:
    """Cumulative radiation exposure for one crew member."""
    name: str = "crew-1"
    cumulative_msv: float = 0.0
    thirty_day_msv: float = 0.0
    annual_msv: float = 0.0
    daily_history: List[float] = field(default_factory=list)
    in_shelter: bool = False
    eva_fraction: float = 0.0
    def __post_init__(self):
        self.cumulative_msv = max(0.0, self.cumulative_msv)
        self.thirty_day_msv = max(0.0, self.thirty_day_msv)
        self.annual_msv = max(0.0, self.annual_msv)
        self.eva_fraction = max(0.0, min(1.0, self.eva_fraction))

@dataclass
class SPEEvent:
    """A solar particle event in progress."""
    start_sol: int = 0
    duration_hours: float = 8.0
    peak_dose_rate_msv_hr: float = 10.0
    elapsed_hours: float = 0.0
    total_delivered_msv: float = 0.0
    active: bool = True
    def __post_init__(self):
        self.duration_hours = max(SPE_DURATION_HOURS_MIN, self.duration_hours)
        self.peak_dose_rate_msv_hr = max(0.0, self.peak_dose_rate_msv_hr)
        self.elapsed_hours = max(0.0, self.elapsed_hours)
        self.total_delivered_msv = max(0.0, self.total_delivered_msv)

@dataclass
class RadiationMonitorState:
    """Complete state of the radiation monitoring system."""
    sol: int = 0
    gcr_dose_rate_msv_day: float = GCR_BASELINE_MSV_DAY
    spe_dose_rate_msv_hr: float = 0.0
    total_ambient_msv_day: float = GCR_BASELINE_MSV_DAY
    solar_cycle_phase: float = 0.0
    habitat_shielding_gcm2: float = 20.0
    shelter_shielding_gcm2: float = 60.0
    shielding_material: str = "regolith"
    active_spe: Optional[SPEEvent] = None
    spe_alert: bool = False
    shelter_ordered: bool = False
    crew: List[CrewDosimetry] = field(default_factory=list)
    detector_health: float = 1.0
    detector_powered: bool = True
    power_draw_watts: float = DETECTOR_POWER_WATTS
    station_cumulative_msv: float = 0.0
    total_spe_events: int = 0
    alerts_issued: int = 0
    def __post_init__(self):
        self.detector_health = max(0.0, min(1.0, self.detector_health))
        self.habitat_shielding_gcm2 = max(0.0, self.habitat_shielding_gcm2)
        self.shelter_shielding_gcm2 = max(0.0, self.shelter_shielding_gcm2)

def solar_cycle_phase(sol):
    """Return solar cycle phase (0=min, 0.5=max, 1=min again)."""
    return (sol % SOLAR_CYCLE_SOLS) / SOLAR_CYCLE_SOLS

def gcr_modulation_factor(phase):
    """GCR flux modulation by solar cycle."""
    mod = math.cos(2.0 * math.pi * phase)
    mid = (GCR_SOLAR_MIN_FACTOR + GCR_SOLAR_MAX_FACTOR) / 2.0
    amp = (GCR_SOLAR_MIN_FACTOR - GCR_SOLAR_MAX_FACTOR) / 2.0
    return mid + amp * mod

def gcr_dose_rate(sol):
    """GCR surface dose rate (mSv/day) for a given sol."""
    return GCR_BASELINE_MSV_DAY * gcr_modulation_factor(solar_cycle_phase(sol))

def spe_probability(phase):
    """Probability of an SPE on a given sol."""
    sa = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
    return SPE_PROBABILITY_PER_SOL_AVG * (1.0 + (SPE_PROBABILITY_SOLAR_MAX_MULT - 1.0) * sa)

def generate_spe(sol, rng=None):
    """Generate a random SPE with physically plausible parameters."""
    r = rng or random.Random()
    log_dose = r.gauss(math.log(20.0), 0.8)
    total_dose = max(SPE_DOSE_MIN_MSV, min(SPE_DOSE_MAX_MSV, math.exp(log_dose)))
    duration = r.uniform(SPE_DURATION_HOURS_MIN, SPE_DURATION_HOURS_MAX)
    return SPEEvent(start_sol=sol, duration_hours=duration,
                    peak_dose_rate_msv_hr=1.5 * total_dose / duration)

def spe_dose_rate_at_time(event, hours_into_event):
    """Dose rate (mSv/hr) at a given time during an SPE. Gaussian profile."""
    if not event.active or hours_into_event < 0.0 or hours_into_event > event.duration_hours:
        return 0.0
    mid = event.duration_hours / 2.0
    sig = event.duration_hours / 4.0
    if sig <= 0.0:
        return 0.0
    return event.peak_dose_rate_msv_hr * math.exp(-0.5 * ((hours_into_event - mid) / sig) ** 2)

def shielding_attenuation(thickness_gcm2, material="regolith"):
    """Dose attenuation: I/I0 = exp(-x/lambda)."""
    lam = {"polyethylene": ATTEN_LENGTH_POLYETHYLENE, "regolith": ATTEN_LENGTH_REGOLITH,
           "water": ATTEN_LENGTH_WATER, "aluminum": ATTEN_LENGTH_ALUMINUM}.get(material, ATTEN_LENGTH_REGOLITH)
    return math.exp(-max(0.0, thickness_gcm2) / lam) if lam > 0 else 1.0

def thickness_to_gcm2(thickness_cm, material="regolith"):
    """Convert thickness (cm) to areal density (g/cm2)."""
    d = {"polyethylene": DENSITY_POLYETHYLENE, "regolith": DENSITY_REGOLITH,
         "water": DENSITY_WATER, "aluminum": DENSITY_ALUMINUM}.get(material, DENSITY_REGOLITH)
    return max(0.0, thickness_cm) * d

def effective_dose(absorbed_msv, quality_factor):
    """Effective dose = absorbed * Q."""
    return absorbed_msv * max(0.0, quality_factor)

def detector_degradation(current_health, dose_msv):
    """Silicon detector radiation damage."""
    return max(0.0, min(1.0, current_health - dose_msv * 0.00001))

def check_dose_limits(crew_member):
    """Check against NASA dose limits. Returns alert strings."""
    alerts = []
    c = crew_member
    if c.cumulative_msv >= CAREER_LIMIT_MSV:
        alerts.append("CRITICAL: %s EXCEEDED career limit (%.1f/%.0f mSv)" % (c.name, c.cumulative_msv, CAREER_LIMIT_MSV))
    elif c.cumulative_msv >= CAREER_LIMIT_MSV * ALERT_FRACTION:
        alerts.append("WARNING: %s at %.1f/%.0f mSv career dose" % (c.name, c.cumulative_msv, CAREER_LIMIT_MSV))
    if c.thirty_day_msv >= THIRTY_DAY_LIMIT_MSV:
        alerts.append("CRITICAL: %s EXCEEDED 30-day limit (%.1f/%.0f mSv)" % (c.name, c.thirty_day_msv, THIRTY_DAY_LIMIT_MSV))
    elif c.thirty_day_msv >= THIRTY_DAY_LIMIT_MSV * ALERT_FRACTION:
        alerts.append("WARNING: %s at %.1f/%.0f mSv 30-day dose" % (c.name, c.thirty_day_msv, THIRTY_DAY_LIMIT_MSV))
    if c.annual_msv >= ANNUAL_LIMIT_MSV:
        alerts.append("CRITICAL: %s EXCEEDED annual limit (%.1f/%.0f mSv)" % (c.name, c.annual_msv, ANNUAL_LIMIT_MSV))
    elif c.annual_msv >= ANNUAL_LIMIT_MSV * ALERT_FRACTION:
        alerts.append("WARNING: %s at %.1f/%.0f mSv annual dose" % (c.name, c.annual_msv, ANNUAL_LIMIT_MSV))
    return alerts

def integrate_spe_sol_dose(event):
    """Integrate SPE dose over one sol (mSv)."""
    if not event.active:
        return 0.0
    dt = HOURS_PER_SOL / 100
    return sum(spe_dose_rate_at_time(event, event.elapsed_hours + i * dt) * dt for i in range(100))

@dataclass
class TickResult:
    """Result of one radiation monitoring tick."""
    sol: int = 0
    gcr_dose_msv: float = 0.0
    spe_dose_msv: float = 0.0
    total_unshielded_msv: float = 0.0
    habitat_dose_msv: float = 0.0
    shelter_dose_msv: float = 0.0
    spe_started: bool = False
    spe_ended: bool = False
    shelter_ordered: bool = False
    alerts: List[str] = field(default_factory=list)

def tick_radiation(state, rng=None):
    """Advance the radiation monitor by one sol."""
    r = rng or random.Random()
    result = TickResult(sol=state.sol)
    state.sol += 1
    state.solar_cycle_phase = solar_cycle_phase(state.sol)
    state.gcr_dose_rate_msv_day = gcr_dose_rate(state.sol)
    gcr_daily = state.gcr_dose_rate_msv_day
    result.gcr_dose_msv = gcr_daily

    spe_daily = 0.0
    if state.active_spe and state.active_spe.active:
        spe_daily = integrate_spe_sol_dose(state.active_spe)
        state.active_spe.elapsed_hours += HOURS_PER_SOL
        state.active_spe.total_delivered_msv += spe_daily
        if state.active_spe.elapsed_hours >= state.active_spe.duration_hours:
            state.active_spe.active = False
            result.spe_ended = True
            state.spe_alert = False
            state.shelter_ordered = False
    else:
        if r.random() < spe_probability(state.solar_cycle_phase):
            state.active_spe = generate_spe(state.sol, r)
            state.total_spe_events += 1
            state.spe_alert = True
            result.spe_started = True
            spe_daily = integrate_spe_sol_dose(state.active_spe)
            state.active_spe.elapsed_hours = HOURS_PER_SOL
            state.active_spe.total_delivered_msv = spe_daily

    state.spe_dose_rate_msv_hr = (spe_daily / HOURS_PER_SOL) if spe_daily > 0 else 0.0
    result.spe_dose_msv = spe_daily
    total_unshielded = gcr_daily + spe_daily
    result.total_unshielded_msv = total_unshielded
    state.total_ambient_msv_day = total_unshielded

    hab_atten = shielding_attenuation(state.habitat_shielding_gcm2, state.shielding_material)
    shlt_atten = shielding_attenuation(state.shelter_shielding_gcm2, state.shielding_material)
    habitat_dose = total_unshielded * hab_atten
    shelter_dose = total_unshielded * shlt_atten
    result.habitat_dose_msv = habitat_dose
    result.shelter_dose_msv = shelter_dose

    if state.spe_alert and spe_daily > 5.0:
        state.shelter_ordered = True
        result.shelter_ordered = True

    all_alerts = []
    for crew in state.crew:
        base = shelter_dose if (crew.in_shelter or state.shelter_ordered) else habitat_dose
        dose = ((1.0 - crew.eva_fraction) * base + crew.eva_fraction * total_unshielded
                if crew.eva_fraction > 0.0 and not state.shelter_ordered else base)
        gcr_frac = gcr_daily / total_unshielded if total_unshielded > 0 else 1.0
        avg_qf = gcr_frac * QUALITY_FACTOR_GCR + (1.0 - gcr_frac) * QUALITY_FACTOR_SPE
        eff = effective_dose(dose, avg_qf)
        crew.cumulative_msv += eff
        crew.daily_history.append(eff)
        crew.thirty_day_msv = sum(crew.daily_history[-30:])
        if len(crew.daily_history) > 668:
            crew.daily_history = crew.daily_history[-668:]
        crew.annual_msv = sum(crew.daily_history[-668:])
        all_alerts.extend(check_dose_limits(crew))

    if all_alerts:
        state.alerts_issued += len(all_alerts)
    result.alerts = all_alerts
    state.station_cumulative_msv += habitat_dose
    if state.detector_powered:
        state.detector_health = detector_degradation(state.detector_health, total_unshielded)
    return result

def create_radiation_monitor(scenario="standard", crew_names=None):
    """Create a RadiationMonitorState for a named scenario."""
    configs = {
        "standard": dict(habitat_shielding_gcm2=20.0, shelter_shielding_gcm2=60.0, shielding_material="regolith"),
        "minimal": dict(habitat_shielding_gcm2=5.0, shelter_shielding_gcm2=20.0, shielding_material="aluminum"),
        "bunker": dict(habitat_shielding_gcm2=40.0, shelter_shielding_gcm2=100.0, shielding_material="water"),
    }
    cfg = configs.get(scenario, configs["standard"])
    names = crew_names or ["commander", "engineer", "scientist", "medic"]
    return RadiationMonitorState(crew=[CrewDosimetry(name=n) for n in names], **cfg)
