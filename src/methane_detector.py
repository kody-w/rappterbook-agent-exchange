"""methane_detector.py — Mars Atmospheric Methane Detection & Monitoring.

Methane (CH₄) on Mars is one of the deepest mysteries in planetary
science.  NASA's Curiosity rover measured background levels of ~0.41 ppbv
with seasonal spikes up to ~0.65 ppbv at Gale Crater, and transient
plumes exceeding 7 ppbv.  The European ExoMars TGO, orbiting overhead,
sees essentially zero — the "methane paradox."

For a Mars colony, methane matters twice:

1. **Science** — Detecting and characterising methane constrains whether
   Mars has active biology (methanogenic archaea) or geology (serpentinisation,
   Fischer-Tropsch reactions, UV decomposition of meteoritic organics).
   The δ¹³C isotope ratio separates biotic (< −40‰) from abiotic (> −20‰)
   sources.

2. **Safety** — Methane is explosive at 5–15% by volume in air.  Even at
   ppb ambient levels, leaks from Sabatier reactors or propellant storage
   can accumulate in sealed habitats.  Continuous monitoring prevents
   catastrophic ignition.

Physics modelled
----------------
* **Tunable Laser Absorption Spectroscopy (TLAS)** — A near-IR diode
  laser at 3.27 µm scans across the CH₄ ν₃ absorption band.
  Absorbance follows Beer-Lambert law:
    A = ε · c · L
  where ε is molar absorptivity (m²/mol), c is concentration (mol/m³),
  L is optical path length (m).  Sensitivity ∝ path length.

* **Detection limit** — The minimum detectable concentration depends on
  path length, laser power, and detector noise floor (NEP).
  Typical Herriott multi-pass cell: L = 100m effective path in a
  compact volume.  Detection limit ~0.1 ppbv at 100m path.

* **Ambient Mars methane** — Background ~0.41 ppbv with seasonal cycle
  peaking in northern summer.  Modelled as sinusoidal with period
  = 1 Mars year (668.6 sols) plus stochastic plumes.

* **Habitat methane** — Sabatier reactor (CO₂ + 4H₂ → CH₄ + 2H₂O)
  can leak.  Propellant depot stores LCH₄.  Threshold alarm at
  1000 ppmv (0.1%), critical alarm at 10,000 ppmv (1%).

* **Isotope discrimination** — δ¹³C measured by comparing ¹²CH₄ and
  ¹³CH₄ absorption lines (offset by ~0.02 µm).  Precision ~5‰
  per measurement, improves with √N averaging.

* **Power consumption** — Laser + detector + heater + data logger.
  ~15 W continuous, 25 W during active scan.

* **Sensor degradation** — Laser drift, mirror fouling in dusty Mars
  atmosphere.  Detection limit worsens ~0.5% per sol without
  calibration.  Calibration resets degradation.

Conservation laws
-----------------
- Detection limit ≥ physical minimum (noise floor)
- Concentration readings ≥ 0 (cannot detect negative methane)
- Power consumed ≤ power available
- Sensor degradation ∈ [0, 1] (0 = perfect, 1 = blind)
- Isotope δ¹³C values physically bounded (−100‰ to +50‰)
- Alert level monotonically increases with concentration
- Energy per scan ≤ battery capacity

Reference:
  - Webster et al. (2015) Science: Curiosity TLS methane measurements
  - Mumma et al. (2009) Science: Mars methane plumes
  - Giuranna et al. (2019) Nature Geoscience: Mars Express PFS
  - Korablev et al. (2019) Nature: TGO ACS non-detection
  - CH₄ LEL: 5% by volume in air (NFPA)
  - Mars year: 668.6 sols
  - Mars sol: 88,775 seconds

One tick = one sol.  Concentrations in ppbv (ambient) or ppmv (habitat).
Power in watts, energy in Wh.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ── Physical constants ──────────────────────────────────────────────

# Mars atmosphere
MARS_SURFACE_PRESSURE_PA = 610.0          # average surface pressure
MARS_SURFACE_TEMP_K = 210.0               # average surface temperature
MARS_ATM_CO2_FRACTION = 0.9532            # CO₂ dominates Mars air
MARS_YEAR_SOLS = 668.6                     # one Mars year

# Methane on Mars
AMBIENT_CH4_BACKGROUND_PPBV = 0.41        # Curiosity TLS background
AMBIENT_CH4_SEASONAL_AMPLITUDE_PPBV = 0.24  # seasonal swing (±)
AMBIENT_CH4_PLUME_PROBABILITY = 0.005     # chance per sol of a plume
AMBIENT_CH4_PLUME_MIN_PPBV = 2.0          # minimum plume concentration
AMBIENT_CH4_PLUME_MAX_PPBV = 15.0         # maximum plume (Mumma 2009: ~45 ppb theoretical)

# Habitat safety thresholds (ppmv)
HABITAT_CH4_WARNING_PPMV = 1000.0         # 0.1% — first alarm
HABITAT_CH4_CRITICAL_PPMV = 10_000.0      # 1.0% — evacuate
HABITAT_CH4_LEL_PPMV = 50_000.0           # 5.0% — lower explosive limit

# Spectroscopy
LASER_WAVELENGTH_UM = 3.27                # CH₄ ν₃ band centre
CH4_MOLAR_ABSORPTIVITY_M2_MOL = 0.044    # effective ε at 3.27 µm
DEFAULT_PATH_LENGTH_M = 100.0             # Herriott cell effective path
NOISE_FLOOR_PPBV = 0.05                   # detector NEP equivalent

# Isotope analysis
BIOTIC_D13C_THRESHOLD = -40.0             # ‰ — below is biotic signature
ABIOTIC_D13C_MIN = -20.0                  # ‰ — typical abiotic minimum
D13C_PRECISION_PER_MEASUREMENT = 5.0      # ‰ — single measurement uncertainty
D13C_PHYSICAL_MIN = -100.0                # ‰ — absolute physical lower bound
D13C_PHYSICAL_MAX = 50.0                  # ‰ — absolute physical upper bound

# Power
STANDBY_POWER_W = 15.0                    # idle monitoring
SCAN_POWER_W = 25.0                       # active measurement
SECONDS_PER_SOL = 88_775.0                # Mars sol duration
HOURS_PER_SOL = SECONDS_PER_SOL / 3600.0  # ~24.66 hours

# Sensor degradation
DEGRADATION_RATE_PER_SOL = 0.005          # 0.5% detection limit worsening per sol
MAX_DEGRADATION = 0.80                     # 80% — sensor is effectively blind
CALIBRATION_RESET_FACTOR = 0.05           # calibration restores to 5% of max


# ── Pure functions ──────────────────────────────────────────────────

def beer_lambert_absorbance(
    molar_absorptivity: float,
    concentration_mol_m3: float,
    path_length_m: float,
) -> float:
    """Compute absorbance via Beer-Lambert law: A = ε·c·L.

    All inputs must be non-negative.
    """
    if molar_absorptivity < 0 or concentration_mol_m3 < 0 or path_length_m < 0:
        raise ValueError("Beer-Lambert inputs must be non-negative")
    return molar_absorptivity * concentration_mol_m3 * path_length_m


def ppbv_to_mol_m3(ppbv: float, pressure_pa: float, temperature_k: float) -> float:
    """Convert parts-per-billion by volume to mol/m³ using ideal gas law.

    n/V = P / (R·T)  →  concentration = (ppbv × 1e-9) × P / (R·T)
    """
    if temperature_k <= 0:
        raise ValueError("Temperature must be positive")
    if ppbv < 0:
        raise ValueError("Concentration cannot be negative")
    R = 8.314  # J/(mol·K)
    return (ppbv * 1e-9) * pressure_pa / (R * temperature_k)


def detection_limit_ppbv(
    path_length_m: float,
    noise_floor: float = NOISE_FLOOR_PPBV,
    degradation: float = 0.0,
) -> float:
    """Minimum detectable CH₄ concentration in ppbv.

    Detection limit improves with longer path, worsens with degradation.
    degradation ∈ [0, 1]: 0 = perfect sensor, 1 = blind.
    """
    if path_length_m <= 0:
        raise ValueError("Path length must be positive")
    degradation = max(0.0, min(1.0, degradation))
    base_limit = noise_floor * (DEFAULT_PATH_LENGTH_M / path_length_m)
    return base_limit / max(1.0 - degradation, 0.01)


def ambient_methane_ppbv(sol: int) -> float:
    """Seasonal ambient Mars methane (deterministic component).

    Sinusoidal model peaking at northern summer (~sol 0 = Ls 0).
    Returns background + seasonal variation (no stochastic plumes).
    """
    phase = 2.0 * math.pi * sol / MARS_YEAR_SOLS
    seasonal = AMBIENT_CH4_SEASONAL_AMPLITUDE_PPBV * math.sin(phase)
    return max(0.0, AMBIENT_CH4_BACKGROUND_PPBV + seasonal)


def is_plume_event(rng: random.Random | None = None) -> bool:
    """Stochastic check: does a methane plume occur this sol?"""
    r = rng if rng is not None else random
    return r.random() < AMBIENT_CH4_PLUME_PROBABILITY


def plume_concentration_ppbv(rng: random.Random | None = None) -> float:
    """Generate a random plume concentration (ppbv) if a plume occurs."""
    r = rng if rng is not None else random
    return r.uniform(AMBIENT_CH4_PLUME_MIN_PPBV, AMBIENT_CH4_PLUME_MAX_PPBV)


def alert_level(concentration_ppmv: float) -> str:
    """Determine habitat alert level from CH₄ concentration in ppmv.

    Returns one of: 'nominal', 'warning', 'critical', 'explosive'.
    """
    if concentration_ppmv < 0:
        raise ValueError("Concentration cannot be negative")
    if concentration_ppmv >= HABITAT_CH4_LEL_PPMV:
        return "explosive"
    if concentration_ppmv >= HABITAT_CH4_CRITICAL_PPMV:
        return "critical"
    if concentration_ppmv >= HABITAT_CH4_WARNING_PPMV:
        return "warning"
    return "nominal"


def isotope_discrimination(d13c: float) -> str:
    """Classify methane source from δ¹³C isotope ratio.

    Returns 'biotic', 'abiotic', or 'ambiguous'.
    """
    if d13c < D13C_PHYSICAL_MIN or d13c > D13C_PHYSICAL_MAX:
        raise ValueError(f"δ¹³C {d13c}‰ outside physical bounds [{D13C_PHYSICAL_MIN}, {D13C_PHYSICAL_MAX}]")
    if d13c <= BIOTIC_D13C_THRESHOLD:
        return "biotic"
    if d13c >= ABIOTIC_D13C_MIN:
        return "abiotic"
    return "ambiguous"


def isotope_precision(n_measurements: int, single_precision: float = D13C_PRECISION_PER_MEASUREMENT) -> float:
    """Precision of δ¹³C after averaging N measurements (√N improvement)."""
    if n_measurements < 1:
        raise ValueError("Need at least 1 measurement")
    return single_precision / math.sqrt(n_measurements)


def scan_energy_wh(scan_duration_s: float, power_w: float = SCAN_POWER_W) -> float:
    """Energy consumed by one active scan in Wh."""
    if scan_duration_s < 0 or power_w < 0:
        raise ValueError("Duration and power must be non-negative")
    return power_w * scan_duration_s / 3600.0


def standby_energy_wh(duration_s: float, power_w: float = STANDBY_POWER_W) -> float:
    """Energy consumed in standby mode in Wh."""
    if duration_s < 0 or power_w < 0:
        raise ValueError("Duration and power must be non-negative")
    return power_w * duration_s / 3600.0


def daily_energy_wh(n_scans: int, scan_duration_s: float = 300.0) -> float:
    """Total energy for one sol: standby + N active scans.

    Standby fills the time not spent scanning.
    """
    if n_scans < 0:
        raise ValueError("Number of scans must be non-negative")
    total_scan_s = n_scans * scan_duration_s
    remaining_s = max(0.0, SECONDS_PER_SOL - total_scan_s)
    return scan_energy_wh(total_scan_s) + standby_energy_wh(remaining_s)


def apply_degradation(current: float, rate: float = DEGRADATION_RATE_PER_SOL) -> float:
    """Apply one sol of sensor degradation.  Returns new degradation ∈ [0, MAX_DEGRADATION]."""
    return min(MAX_DEGRADATION, current + rate * (1.0 - current))


def calibrate_sensor(current_degradation: float) -> float:
    """Calibration resets sensor degradation to near-zero."""
    return min(current_degradation, CALIBRATION_RESET_FACTOR)


# ── State dataclass ─────────────────────────────────────────────────

@dataclass
class MethaneDetector:
    """Full state of the methane detection system."""

    # Time
    sol: int = 0

    # Sensor configuration
    path_length_m: float = DEFAULT_PATH_LENGTH_M
    scans_per_sol: int = 24                # one scan per hour

    # Sensor health
    degradation: float = 0.0               # 0 = perfect, MAX_DEGRADATION = blind

    # Latest readings
    ambient_ch4_ppbv: float = AMBIENT_CH4_BACKGROUND_PPBV
    habitat_ch4_ppmv: float = 0.0          # habitat internal reading
    last_d13c: float = -30.0               # last isotope measurement
    alert: str = "nominal"

    # Science accumulation
    plume_events_detected: int = 0
    total_measurements: int = 0
    isotope_measurements: int = 0
    d13c_running_mean: float = -30.0
    d13c_running_precision: float = D13C_PRECISION_PER_MEASUREMENT

    # Energy accounting
    total_energy_wh: float = 0.0

    # History (last 10 sols of ambient readings)
    ambient_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dictionary."""
        return {
            "sol": self.sol,
            "path_length_m": self.path_length_m,
            "scans_per_sol": self.scans_per_sol,
            "degradation": round(self.degradation, 6),
            "ambient_ch4_ppbv": round(self.ambient_ch4_ppbv, 4),
            "habitat_ch4_ppmv": round(self.habitat_ch4_ppmv, 4),
            "last_d13c": round(self.last_d13c, 2),
            "alert": self.alert,
            "plume_events_detected": self.plume_events_detected,
            "total_measurements": self.total_measurements,
            "isotope_measurements": self.isotope_measurements,
            "d13c_running_mean": round(self.d13c_running_mean, 2),
            "d13c_running_precision": round(self.d13c_running_precision, 4),
            "total_energy_wh": round(self.total_energy_wh, 2),
            "ambient_history": [round(v, 4) for v in self.ambient_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MethaneDetector":
        """Deserialise from dictionary."""
        return cls(
            sol=data.get("sol", 0),
            path_length_m=data.get("path_length_m", DEFAULT_PATH_LENGTH_M),
            scans_per_sol=data.get("scans_per_sol", 24),
            degradation=data.get("degradation", 0.0),
            ambient_ch4_ppbv=data.get("ambient_ch4_ppbv", AMBIENT_CH4_BACKGROUND_PPBV),
            habitat_ch4_ppmv=data.get("habitat_ch4_ppmv", 0.0),
            last_d13c=data.get("last_d13c", -30.0),
            alert=data.get("alert", "nominal"),
            plume_events_detected=data.get("plume_events_detected", 0),
            total_measurements=data.get("total_measurements", 0),
            isotope_measurements=data.get("isotope_measurements", 0),
            d13c_running_mean=data.get("d13c_running_mean", -30.0),
            d13c_running_precision=data.get("d13c_running_precision", D13C_PRECISION_PER_MEASUREMENT),
            total_energy_wh=data.get("total_energy_wh", 0.0),
            ambient_history=list(data.get("ambient_history", [])),
        )


# ── Tick result ─────────────────────────────────────────────────────

@dataclass
class TickResult:
    """Result of advancing one sol."""

    sol: int
    ambient_ch4_ppbv: float
    habitat_ch4_ppmv: float
    detection_limit_ppbv: float
    plume_detected: bool
    alert: str
    d13c: float
    isotope_class: str
    energy_used_wh: float
    degradation: float


# ── Tick engine ─────────────────────────────────────────────────────

def tick(
    state: MethaneDetector,
    habitat_ch4_ppmv: float = 0.0,
    d13c_sample: float | None = None,
    do_calibration: bool = False,
    rng: random.Random | None = None,
) -> TickResult:
    """Advance the methane detector by one sol.

    Parameters
    ----------
    state : MethaneDetector
        Current detector state (mutated in place).
    habitat_ch4_ppmv : float
        Current habitat CH₄ reading from internal sensors (ppmv).
    d13c_sample : float or None
        If provided, an isotope measurement this sol.
    do_calibration : bool
        If True, perform sensor calibration this sol.
    rng : Random or None
        RNG for reproducible plume stochasticity.
    """
    state.sol += 1

    # ── Calibration (before measurements) ───────────────────────────
    if do_calibration:
        state.degradation = calibrate_sensor(state.degradation)

    # ── Ambient methane ─────────────────────────────────────────────
    seasonal = ambient_methane_ppbv(state.sol)
    plume = False
    plume_addition = 0.0
    if is_plume_event(rng):
        plume_addition = plume_concentration_ppbv(rng)
        plume = True
        state.plume_events_detected += 1

    ambient = seasonal + plume_addition
    det_limit = detection_limit_ppbv(state.path_length_m, degradation=state.degradation)

    # Can we actually detect it?
    measured_ambient = ambient if ambient >= det_limit else 0.0
    state.ambient_ch4_ppbv = measured_ambient

    # ── Habitat reading ─────────────────────────────────────────────
    state.habitat_ch4_ppmv = max(0.0, habitat_ch4_ppmv)
    state.alert = alert_level(state.habitat_ch4_ppmv)

    # ── Isotope measurement ─────────────────────────────────────────
    d13c_val = state.last_d13c
    iso_class = "none"
    if d13c_sample is not None:
        clamped = max(D13C_PHYSICAL_MIN, min(D13C_PHYSICAL_MAX, d13c_sample))
        state.last_d13c = clamped
        d13c_val = clamped
        state.isotope_measurements += 1
        # Running mean update
        n = state.isotope_measurements
        state.d13c_running_mean = (
            state.d13c_running_mean * (n - 1) + clamped
        ) / n
        state.d13c_running_precision = isotope_precision(n)
        iso_class = isotope_discrimination(clamped)

    # ── Energy accounting ───────────────────────────────────────────
    energy = daily_energy_wh(state.scans_per_sol)
    state.total_energy_wh += energy

    # ── Measurement counter ─────────────────────────────────────────
    state.total_measurements += state.scans_per_sol

    # ── Sensor degradation ──────────────────────────────────────────
    state.degradation = apply_degradation(state.degradation)

    # ── History ─────────────────────────────────────────────────────
    state.ambient_history.append(round(ambient, 4))
    if len(state.ambient_history) > 10:
        state.ambient_history = state.ambient_history[-10:]

    return TickResult(
        sol=state.sol,
        ambient_ch4_ppbv=measured_ambient,
        habitat_ch4_ppmv=state.habitat_ch4_ppmv,
        detection_limit_ppbv=det_limit,
        plume_detected=plume,
        alert=state.alert,
        d13c=d13c_val,
        isotope_class=iso_class,
        energy_used_wh=energy,
        degradation=state.degradation,
    )


def run_simulation(n_sols: int, seed: int = 42, **kwargs: Any) -> tuple[MethaneDetector, list[TickResult]]:
    """Run the detector for N sols and return final state + tick history."""
    rng = random.Random(seed)
    state = MethaneDetector(**kwargs)
    results: list[TickResult] = []
    for _ in range(n_sols):
        result = tick(state, rng=rng)
        results.append(result)
    return state, results
