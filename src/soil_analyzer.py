"""soil_analyzer.py — Mars Regolith Composition Analyzer

X-ray Fluorescence (XRF) spectroscopy for in-situ Mars soil analysis.
Identifies mineral content, perchlorate concentration, water-ice
signatures, and toxic element levels.  Critical for resource extraction
planning (ISRU) and crew safety assessment.

Physics
-------
* **X-ray fluorescence**: Primary X-ray beam excites atoms → characteristic
  secondary (fluorescent) X-rays emitted.  Energy of secondary photon
  identifies the element; intensity ∝ concentration.
* **Beer–Lambert attenuation**: I = I₀ · exp(−μ·ρ·x).  The signal from
  deeper layers is exponentially attenuated, giving a penetration depth
  of ~100 μm for typical Mars regolith at 6–15 keV.
* **Perchlorate detection**: ClO₄⁻ prevalence 0.5–1% by mass (Phoenix,
  Curiosity). Toxic at >0.1% for crew health.  XRF measures Cl Kα
  at 2.62 keV; perchlorate fraction estimated from Cl/O ratio.
* **Water-ice proxy**: Hydrogen cannot be detected by XRF directly.
  Proxy: excess oxygen signal above stoichiometric mineral prediction
  suggests H₂O or OH⁻ in the sample.

Reference missions: Phoenix (2008), Curiosity/APXS (2012–present),
Perseverance/PIXL (2021–present).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


# ── Physical constants ──────────────────────────────────────────────
MARS_GRAVITY = 3.72          # m/s²
REGOLITH_DENSITY = 1500.0    # kg/m³  (bulk, loosely packed)
XRF_TUBE_VOLTAGE = 40.0      # kV  (max accelerating voltage)
XRF_BEAM_CURRENT = 0.1       # mA  (micro-focus tube)

# Element Kα energies (keV) — the fingerprint lines
KA_ENERGIES: Dict[str, float] = {
    "Fe": 6.40,   "Si": 1.74,   "Ca": 3.69,
    "Al": 1.49,   "Mg": 1.25,   "S":  2.31,
    "Cl": 2.62,   "Ti": 4.51,   "K":  3.31,
    "Na": 1.04,   "O":  0.52,   "Mn": 5.90,
    "Cr": 5.41,   "P":  2.01,   "Ni": 7.47,
}

# Typical Mars regolith mass fractions (wt%) — from Curiosity/APXS averages
MARS_BASELINE: Dict[str, float] = {
    "SiO2": 43.0,  "FeO": 18.5,  "Al2O3": 9.4,
    "CaO": 7.2,    "MgO": 8.6,   "SO3": 6.2,
    "TiO2": 1.1,   "Na2O": 2.7,  "K2O": 0.5,
    "Cl": 0.7,     "MnO": 0.4,   "Cr2O3": 0.4,
    "P2O5": 0.9,   "NiO": 0.04,
}

# Perchlorate safety threshold (mass fraction)
PERCHLORATE_SAFE_LIMIT = 0.001   # 0.1 wt%
PERCHLORATE_WARNING = 0.005      # 0.5 wt%

# Attenuation: mass attenuation coefficient approximation (cm²/g)
# Simplified average for Mars basaltic regolith at ~6 keV
MU_REGOLITH_6KEV = 50.0   # cm²/g


# ── State ───────────────────────────────────────────────────────────

@dataclass
class SoilSample:
    """One analyzed soil sample."""
    sample_id: int = 0
    depth_cm: float = 0.0        # sampling depth
    composition: Dict[str, float] = field(default_factory=dict)  # oxide wt%
    perchlorate_pct: float = 0.0  # ClO4⁻ mass %
    water_proxy_pct: float = 0.0  # estimated H2O from excess O signal
    toxic: bool = False
    quality: float = 1.0         # signal quality 0..1

    def __post_init__(self) -> None:
        self.depth_cm = max(0.0, self.depth_cm)
        self.perchlorate_pct = max(0.0, min(100.0, self.perchlorate_pct))
        self.water_proxy_pct = max(0.0, min(100.0, self.water_proxy_pct))
        self.quality = max(0.0, min(1.0, self.quality))


@dataclass
class AnalyzerState:
    """State of the XRF soil analyzer instrument."""
    power_w: float = 0.0           # current power draw (watts)
    tube_voltage_kv: float = 40.0  # X-ray tube voltage
    tube_hours: float = 0.0        # cumulative tube-on hours
    tube_max_hours: float = 5000.0 # tube lifetime
    samples_taken: int = 0
    samples: list = field(default_factory=list)  # list of SoilSample
    calibrated: bool = True
    integration_time_s: float = 30.0  # measurement time per spot
    temperature_c: float = -20.0   # instrument temperature

    def __post_init__(self) -> None:
        self.power_w = max(0.0, self.power_w)
        self.tube_voltage_kv = max(0.0, min(60.0, self.tube_voltage_kv))
        self.tube_hours = max(0.0, self.tube_hours)
        self.integration_time_s = max(1.0, min(600.0, self.integration_time_s))
        self.temperature_c = max(-120.0, min(80.0, self.temperature_c))


@dataclass
class AnalysisResult:
    """Result of one analysis tick."""
    sample: SoilSample | None = None
    signal_counts: float = 0.0
    snr: float = 0.0
    penetration_depth_um: float = 0.0
    tube_degradation: float = 0.0
    power_used_wh: float = 0.0
    warning: str = ""


# ── Pure physics functions ──────────────────────────────────────────

def beer_lambert(intensity_0: float, mu: float, rho_g_cm3: float,
                 depth_cm: float) -> float:
    """Beer-Lambert attenuation: I = I0 * exp(-mu * rho * x).

    Returns transmitted intensity.
    """
    exponent = -mu * rho_g_cm3 * depth_cm
    exponent = max(-500.0, exponent)  # prevent underflow
    return intensity_0 * math.exp(exponent)


def xrf_signal_counts(tube_kv: float, current_ma: float,
                      integration_s: float,
                      element_fraction: float) -> float:
    """Estimate detected XRF counts for an element.

    Simplified model: counts proportional to tube power * time * concentration.
    Real instruments use fundamental parameters; this captures the scaling.
    """
    tube_power = tube_kv * current_ma  # watts
    raw_counts = tube_power * integration_s * element_fraction * 1000.0
    return max(0.0, raw_counts)


def signal_to_noise(counts: float, background: float = 10.0) -> float:
    """SNR = net_counts / sqrt(net_counts + 2*background).

    Standard counting statistics for XRF.
    """
    net = max(0.0, counts - background)
    denominator = math.sqrt(max(1.0, net + 2.0 * background))
    return net / denominator


def penetration_depth_um(energy_kev: float, density_g_cm3: float) -> float:
    """Approximate XRF information depth in micrometers.

    Higher energy → deeper penetration.  Rough scaling:
    depth ≈ (E / 6)^2.5 * 100 / density.
    """
    if energy_kev <= 0 or density_g_cm3 <= 0:
        return 0.0
    e_ratio = energy_kev / 6.0
    depth_cm = (e_ratio ** 2.5) * 100.0 / (MU_REGOLITH_6KEV * density_g_cm3)
    return depth_cm * 1e4  # cm → μm


def perchlorate_from_chlorine(cl_pct: float, o_excess_pct: float) -> float:
    """Estimate perchlorate (ClO4⁻) mass fraction from Cl and excess O.

    If Cl is present with excess O beyond mineral stoichiometry,
    perchlorate is likely.  ClO4 molecular weight ratio: Cl/ClO4 = 35.5/99.5.
    """
    if cl_pct <= 0:
        return 0.0
    # Scale Cl to ClO4 equivalent
    clo4_equivalent = cl_pct * (99.5 / 35.5)
    # Modulate by excess O confidence (0..1 scaling)
    confidence = min(1.0, o_excess_pct / 5.0) if o_excess_pct > 0 else 0.3
    return clo4_equivalent * confidence


def water_proxy(o_measured_pct: float, o_stoichiometric_pct: float) -> float:
    """Estimate water content from excess oxygen signal.

    XRF cannot detect H directly.  Excess O above mineral stoichiometry
    suggests H2O or OH⁻ groups.  H2O/O mass ratio ≈ 18/16 = 1.125.
    """
    excess = o_measured_pct - o_stoichiometric_pct
    if excess <= 0:
        return 0.0
    return excess * 1.125  # scale O excess to H2O equivalent


def assess_toxicity(perchlorate_pct: float, cr_pct: float = 0.0,
                    ni_pct: float = 0.0) -> tuple:
    """Assess soil toxicity for crew safety.

    Returns (toxic: bool, reason: str).
    """
    reasons = []
    if perchlorate_pct > PERCHLORATE_SAFE_LIMIT:
        reasons.append(f"perchlorate {perchlorate_pct:.3f}% > {PERCHLORATE_SAFE_LIMIT:.3f}%")
    if cr_pct > 0.05:
        reasons.append(f"chromium {cr_pct:.3f}% elevated")
    if ni_pct > 0.02:
        reasons.append(f"nickel {ni_pct:.3f}% elevated")
    toxic = len(reasons) > 0
    return toxic, "; ".join(reasons) if reasons else "safe"


def tube_degradation_rate(tube_hours: float, max_hours: float) -> float:
    """X-ray tube degradation factor (0..1, 1 = new).

    Accelerating degradation near end of life.
    """
    if max_hours <= 0:
        return 0.0
    fraction_used = min(1.0, tube_hours / max_hours)
    return 1.0 - fraction_used ** 2  # quadratic degradation curve


def temperature_quality_factor(temp_c: float) -> float:
    """Signal quality factor based on instrument temperature.

    Optimal: -40 to +10°C (detector performs best cold).
    Degrades outside this range.
    """
    if -40.0 <= temp_c <= 10.0:
        return 1.0
    if temp_c < -40.0:
        # Too cold: electronics sluggish
        return max(0.3, 1.0 - ((-40.0 - temp_c) / 80.0))
    # Too hot: detector noise increases
    return max(0.3, 1.0 - ((temp_c - 10.0) / 70.0))


# ── Tick function ───────────────────────────────────────────────────

def tick_analysis(state: AnalyzerState, dt_s: float = 60.0,
                  take_sample: bool = False,
                  sample_depth_cm: float = 0.0) -> AnalysisResult:
    """Advance the soil analyzer by one time step.

    If take_sample is True, performs an XRF measurement at the given depth.
    Otherwise, idles (standby power only).
    """
    result = AnalysisResult()

    # Tube degradation
    tube_factor = tube_degradation_rate(state.tube_hours, state.tube_max_hours)
    result.tube_degradation = 1.0 - tube_factor

    if not take_sample:
        # Standby mode: 5W idle
        state.power_w = 5.0
        result.power_used_wh = 5.0 * dt_s / 3600.0
        return result

    # Tube dead check
    if tube_factor <= 0.01:
        result.warning = "tube_exhausted"
        state.power_w = 5.0
        result.power_used_wh = 5.0 * dt_s / 3600.0
        return result

    # Calibration check
    if not state.calibrated:
        result.warning = "uncalibrated"

    # Active analysis: power draw ~30W
    state.power_w = 30.0
    analysis_time_h = state.integration_time_s / 3600.0
    state.tube_hours += analysis_time_h
    result.power_used_wh = state.power_w * dt_s / 3600.0

    # Temperature quality
    temp_q = temperature_quality_factor(state.temperature_c)

    # Generate composition from Mars baseline + noise
    composition = {}
    for oxide, baseline_pct in MARS_BASELINE.items():
        # Small random-free variation based on depth
        depth_factor = 1.0 + 0.1 * math.sin(sample_depth_cm * 3.14)
        composition[oxide] = baseline_pct * depth_factor

    # Normalize to ~100%
    total = sum(composition.values())
    if total > 0:
        for k in composition:
            composition[k] = composition[k] * 100.0 / total

    # Chlorine → perchlorate estimation
    cl_pct = composition.get("Cl", 0.0)
    # O stoichiometric from SiO2, FeO, etc. (simplified: ~45% of oxide mass)
    o_stoich = sum(v for k, v in composition.items() if "O" in k) * 0.45
    o_measured = o_stoich + sample_depth_cm * 0.1  # deeper = more H2O
    perchlorate = perchlorate_from_chlorine(cl_pct, o_measured - o_stoich)
    water_est = water_proxy(o_measured, o_stoich)

    # Toxicity
    cr_pct = composition.get("Cr2O3", 0.0) * 0.684  # Cr fraction of Cr2O3
    ni_pct = composition.get("NiO", 0.0) * 0.786     # Ni fraction of NiO
    toxic, reason = assess_toxicity(perchlorate, cr_pct, ni_pct)

    # XRF signal for iron (dominant signal)
    fe_fraction = composition.get("FeO", 0.0) / 100.0
    counts = xrf_signal_counts(
        state.tube_voltage_kv, XRF_BEAM_CURRENT,
        state.integration_time_s, fe_fraction
    ) * tube_factor * temp_q

    snr = signal_to_noise(counts)
    depth = penetration_depth_um(KA_ENERGIES["Fe"], REGOLITH_DENSITY / 1000.0)

    # Build sample
    sample = SoilSample(
        sample_id=state.samples_taken + 1,
        depth_cm=sample_depth_cm,
        composition=composition,
        perchlorate_pct=perchlorate,
        water_proxy_pct=water_est,
        toxic=toxic,
        quality=min(1.0, snr / 50.0) * temp_q * tube_factor,
    )

    state.samples_taken += 1
    state.samples.append(sample)

    result.sample = sample
    result.signal_counts = counts
    result.snr = snr
    result.penetration_depth_um = depth
    if toxic:
        result.warning = f"TOXIC: {reason}"

    return result


# ── Factory ─────────────────────────────────────────────────────────

def create_soil_analyzer(scenario: str = "standard") -> AnalyzerState:
    """Create an AnalyzerState for a named scenario.

    Scenarios
    ---------
    standard : Typical rover-mounted XRF (Curiosity-class)
    deep_core : Drill-fed analyzer for subsurface samples
    portable : Handheld unit for EVA crew
    """
    configs = {
        "standard": dict(
            tube_voltage_kv=40.0, tube_max_hours=5000.0,
            integration_time_s=30.0, temperature_c=-20.0,
        ),
        "deep_core": dict(
            tube_voltage_kv=50.0, tube_max_hours=3000.0,
            integration_time_s=60.0, temperature_c=-10.0,
        ),
        "portable": dict(
            tube_voltage_kv=30.0, tube_max_hours=2000.0,
            integration_time_s=15.0, temperature_c=5.0,
        ),
    }
    cfg = configs.get(scenario, configs["standard"])
    return AnalyzerState(**cfg)
