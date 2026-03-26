"""
seismograph.py — Mars seismic monitoring and structural response.

Models marsquake activity based on NASA InSight mission data, tracks
cumulative structural stress on habitat modules, and triggers emergency
protocols for damaging events.

InSight detected 1,319 marsquakes from 2018-2022. Mars is alive beneath
the surface. A colony that ignores seismology is a colony that dies
surprised.

Physical references:
  - InSight SEIS instrument: detected marsquakes M1.0 to M4.7
  - Marsquake frequency: ~2-4 detectable events per sol (InSight catalog)
  - Largest recorded: M4.7 (2022-05-04, S1222a)
  - Two source types: tectonic (crustal stress) and impact (meteorites)
  - Cerberus Fossae: most active seismic region on Mars
  - Mars seismic attenuation: Q factor ~200-400 (lower than Earth)
  - P-wave velocity in Mars crust: ~3.5 km/s (InSight)
  - S-wave velocity in Mars crust: ~2.0 km/s (InSight)
  - Mars has no global magnetic field → no plate tectonics,
      but thermoelastic cooling causes crustal fracturing
  - Gutenberg-Richter law applies: log10(N) = a - b*M
      InSight data: b ≈ 0.75 (fewer large quakes than Earth's b ≈ 1.0)
  - Habitat structural damage threshold: ~M3.5 (Mars-equivalent)
      at typical colony distance from source

One tick = one sol. Magnitudes on Mars Moment Magnitude scale.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants (from InSight SEIS data)
# ---------------------------------------------------------------------------

# Gutenberg-Richter parameters (InSight catalog fit)
GR_A = 1.8                          # log10(N) intercept (events/sol >= M0)
GR_B = 0.75                         # b-value (slope, Mars has fewer large quakes)

# Magnitude boundaries
MIN_DETECTABLE_MAGNITUDE = 1.0       # SEIS detection threshold
MAX_OBSERVED_MAGNITUDE = 5.5         # theoretical max for Mars (no subduction)
STRUCTURAL_DAMAGE_THRESHOLD = 3.5    # habitat feels it at colony distance
EMERGENCY_THRESHOLD = 4.0            # triggers emergency protocol
CATASTROPHIC_THRESHOLD = 5.0         # potential structural failure

# Seismic wave propagation
P_WAVE_VELOCITY_KM_S = 3.5          # compressional wave in Mars crust
S_WAVE_VELOCITY_KM_S = 2.0          # shear wave in Mars crust
SURFACE_WAVE_VELOCITY_KM_S = 1.8    # Rayleigh wave approximation
Q_ATTENUATION = 300.0               # quality factor (energy loss per cycle)

# Event type probabilities (InSight statistics)
TECTONIC_FRACTION = 0.85             # 85% tectonic, 15% impact
IMPACT_FRACTION = 0.15

# Structural stress parameters
STRESS_PER_MAGNITUDE_UNIT = 0.5     # stress points per magnitude above threshold
STRESS_DECAY_PER_SOL = 0.02         # natural stress relaxation per sol
STRESS_CRITICAL = 100.0             # structural failure threshold
STRESS_WARNING = 60.0               # inspection recommended
MAX_STRESS = 100.0

# Distance model (typical colony distance from seismic source)
TYPICAL_SOURCE_DISTANCE_KM = 500.0  # average distance to source
NEAR_FIELD_KM = 50.0                # close enough for severe shaking
FAR_FIELD_KM = 2000.0               # beyond effective range

# Seismograph sensor
SENSOR_NOISE_FLOOR = 0.5            # magnitude below which signal is noise
SENSOR_SAMPLE_RATE_HZ = 20.0        # SEIS sampled at 20 Hz


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarsquakeEvent:
    """A single seismic event on Mars.

    Attributes:
        magnitude: Mars Moment Magnitude
        source_type: 'tectonic' or 'impact'
        distance_km: distance from colony to epicenter
        depth_km: hypocenter depth (Mars quakes are shallow: 5-50 km)
        p_arrival_s: P-wave travel time in seconds
        s_arrival_s: S-wave travel time in seconds
        peak_ground_accel: peak ground acceleration at colony (m/s²)
    """
    magnitude: float
    source_type: str = "tectonic"
    distance_km: float = 500.0
    depth_km: float = 20.0
    p_arrival_s: float = 0.0
    s_arrival_s: float = 0.0
    peak_ground_accel: float = 0.0


@dataclass
class SeismicStation:
    """Colony seismograph station state.

    Attributes:
        operational: whether the station is working
        sensitivity: detection sensitivity [0, 1] (degrades with dust/damage)
        events_detected: cumulative events detected
        largest_event: magnitude of largest event ever detected
        structural_stress: cumulative habitat stress from seismicity [0, 100]
        emergency_active: whether emergency protocol is currently active
        sols_monitored: total sols of observation
    """
    operational: bool = True
    sensitivity: float = 1.0
    events_detected: int = 0
    largest_event: float = 0.0
    structural_stress: float = 0.0
    emergency_active: bool = False
    sols_monitored: int = 0

    def __post_init__(self) -> None:
        self.sensitivity = max(0.0, min(1.0, self.sensitivity))
        self.structural_stress = max(0.0, min(MAX_STRESS, self.structural_stress))


@dataclass
class SolSeismicReport:
    """Seismic activity report for one sol.

    Attributes:
        events: list of detected marsquake events
        max_magnitude: largest event this sol (0.0 if none)
        total_events: number of events detected
        stress_added: structural stress added this sol
        stress_relieved: stress relaxation this sol
        emergency_triggered: whether emergency protocol was triggered
        pga_max: peak ground acceleration of strongest event (m/s²)
    """
    events: list[MarsquakeEvent] = field(default_factory=list)
    max_magnitude: float = 0.0
    total_events: int = 0
    stress_added: float = 0.0
    stress_relieved: float = 0.0
    emergency_triggered: bool = False
    pga_max: float = 0.0


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------

def gutenberg_richter_rate(magnitude: float) -> float:
    """Expected number of events per sol at or above given magnitude.

    Uses Gutenberg-Richter law: log10(N) = a - b*M
    Fitted to InSight catalog statistics.
    """
    log_n = GR_A - GR_B * magnitude
    return math.pow(10.0, log_n)


def wave_travel_time(distance_km: float, velocity_km_s: float) -> float:
    """Travel time for a seismic wave (seconds).

    Simple straight-line approximation (adequate for crustal paths).
    """
    if velocity_km_s <= 0 or distance_km < 0:
        return 0.0
    return distance_km / velocity_km_s


def peak_ground_acceleration(magnitude: float, distance_km: float) -> float:
    """Estimate peak ground acceleration (m/s²) at a given distance.

    Simplified ground motion prediction equation (GMPE) for Mars:
    log10(PGA) = c1 + c2*M - c3*log10(R) - c4*R/Q

    Calibrated so M4.7 at 500 km ≈ 0.001 m/s² (InSight observation).
    """
    if distance_km <= 0 or magnitude <= 0:
        return 0.0

    # Coefficients calibrated to InSight observations
    c1 = -3.5
    c2 = 0.8
    c3 = 1.2
    c4 = 0.001

    r = max(1.0, distance_km)  # avoid log(0)
    log_pga = c1 + c2 * magnitude - c3 * math.log10(r) - c4 * r / Q_ATTENUATION
    return math.pow(10.0, log_pga)


def generate_events(
    rng: random.Random | None = None,
    distance_range: tuple[float, float] = (50.0, 2000.0),
) -> list[MarsquakeEvent]:
    """Generate marsquake events for one sol using statistical model.

    Uses Gutenberg-Richter distribution to determine event count and
    magnitude distribution. Most events are small (M1-M2).

    Args:
        rng: random number generator (for reproducibility)
        distance_range: (min, max) distance to colony in km

    Returns:
        List of MarsquakeEvent for this sol (may be empty).
    """
    if rng is None:
        rng = random.Random()

    events = []

    # Expected events above detection threshold
    expected_rate = gutenberg_richter_rate(MIN_DETECTABLE_MAGNITUDE)
    n_events = rng.poisson(expected_rate) if hasattr(rng, 'poisson') else _poisson_sample(rng, expected_rate)

    for _ in range(n_events):
        # Sample magnitude from truncated GR distribution
        # P(M > m) = 10^(a - b*m) / 10^(a - b*M_min)
        u = rng.random()
        if u <= 0:
            u = 1e-10
        magnitude = MIN_DETECTABLE_MAGNITUDE - math.log10(u) / GR_B
        magnitude = min(magnitude, MAX_OBSERVED_MAGNITUDE)

        # Source type
        source_type = "impact" if rng.random() < IMPACT_FRACTION else "tectonic"

        # Distance (log-uniform in range)
        d_min, d_max = distance_range
        log_d = rng.uniform(math.log10(max(1.0, d_min)), math.log10(max(2.0, d_max)))
        distance = math.pow(10.0, log_d)

        # Depth (Mars quakes are shallow)
        depth = rng.uniform(5.0, 50.0)

        # Wave arrivals
        hypo_dist = math.sqrt(distance ** 2 + depth ** 2)
        p_time = wave_travel_time(hypo_dist, P_WAVE_VELOCITY_KM_S)
        s_time = wave_travel_time(hypo_dist, S_WAVE_VELOCITY_KM_S)

        # Ground acceleration at colony
        pga = peak_ground_acceleration(magnitude, distance)

        events.append(MarsquakeEvent(
            magnitude=round(magnitude, 2),
            source_type=source_type,
            distance_km=round(distance, 1),
            depth_km=round(depth, 1),
            p_arrival_s=round(p_time, 1),
            s_arrival_s=round(s_time, 1),
            peak_ground_accel=pga,
        ))

    return events


def structural_stress_from_event(event: MarsquakeEvent) -> float:
    """Calculate structural stress added by a seismic event.

    Only events above the damage threshold at close range cause stress.
    Stress scales with magnitude and inversely with distance.
    """
    if event.magnitude < STRUCTURAL_DAMAGE_THRESHOLD:
        return 0.0

    excess = event.magnitude - STRUCTURAL_DAMAGE_THRESHOLD
    distance_factor = max(0.1, TYPICAL_SOURCE_DISTANCE_KM / max(1.0, event.distance_km))
    stress = STRESS_PER_MAGNITUDE_UNIT * excess * distance_factor
    return max(0.0, stress)


def relax_stress(station: SeismicStation) -> float:
    """Apply natural structural stress relaxation for one sol.

    Represents material creep, inspection repairs, and natural settling.
    Returns amount of stress relieved.
    """
    relief = min(station.structural_stress, STRESS_DECAY_PER_SOL * station.structural_stress)
    station.structural_stress = max(0.0, station.structural_stress - relief)
    return relief


def check_emergency(station: SeismicStation, max_magnitude: float) -> bool:
    """Determine if emergency protocol should be activated.

    Emergency activates when:
      - A single event exceeds EMERGENCY_THRESHOLD, or
      - Structural stress exceeds WARNING level
    """
    if max_magnitude >= EMERGENCY_THRESHOLD:
        return True
    if station.structural_stress >= STRESS_WARNING:
        return True
    return False


def degrade_sensor(station: SeismicStation, dust_factor: float = 0.0) -> None:
    """Apply sensor degradation for one sol.

    Dust accumulation and thermal cycling reduce sensitivity.

    Args:
        station: seismic station (mutated)
        dust_factor: current dust level [0, 1] (0 = clear, 1 = severe storm)
    """
    base_degradation = 0.0001  # ~0.01%/sol from thermal cycling
    dust_degradation = 0.0005 * max(0.0, min(1.0, dust_factor))
    station.sensitivity = max(
        0.0, station.sensitivity - base_degradation - dust_degradation
    )


def calibrate_sensor(station: SeismicStation, quality: float = 1.0) -> None:
    """Recalibrate the seismograph sensor.

    Args:
        station: seismic station (mutated)
        quality: calibration quality [0, 1]
    """
    quality = max(0.0, min(1.0, quality))
    station.sensitivity = min(1.0, station.sensitivity + 0.05 * quality)


# ---------------------------------------------------------------------------
# Main tick function
# ---------------------------------------------------------------------------

def tick_seismic(
    station: SeismicStation,
    rng: random.Random | None = None,
    dust_factor: float = 0.0,
    distance_range: tuple[float, float] = (50.0, 2000.0),
) -> SolSeismicReport:
    """Advance seismic monitoring by one sol.

    Generates marsquake events, detects those above sensor threshold,
    calculates structural stress, and determines emergency status.

    Args:
        station: seismic station state (mutated)
        rng: random number generator for reproducibility
        dust_factor: dust level affecting sensor [0, 1]
        distance_range: (min, max) distance to seismic sources

    Returns:
        SolSeismicReport with all events and stress changes.
    """
    report = SolSeismicReport()

    if not station.operational:
        station.sols_monitored += 1
        return report

    # Generate all seismic events this sol
    all_events = generate_events(rng, distance_range)

    # Filter to detectable events (sensitivity affects detection threshold)
    effective_threshold = MIN_DETECTABLE_MAGNITUDE + (1.0 - station.sensitivity) * 1.5
    detected = [e for e in all_events if e.magnitude >= effective_threshold]

    report.events = detected
    report.total_events = len(detected)

    # Track stress and find max event
    for event in detected:
        station.events_detected += 1
        if event.magnitude > station.largest_event:
            station.largest_event = event.magnitude
        if event.magnitude > report.max_magnitude:
            report.max_magnitude = event.magnitude
        if event.peak_ground_accel > report.pga_max:
            report.pga_max = event.peak_ground_accel

        stress = structural_stress_from_event(event)
        station.structural_stress = min(MAX_STRESS, station.structural_stress + stress)
        report.stress_added += stress

    # Natural stress relaxation
    report.stress_relieved = relax_stress(station)

    # Emergency check
    report.emergency_triggered = check_emergency(station, report.max_magnitude)
    station.emergency_active = report.emergency_triggered

    # Sensor degradation
    degrade_sensor(station, dust_factor)

    station.sols_monitored += 1
    return report


# ---------------------------------------------------------------------------
# Utility: Poisson sampling without numpy
# ---------------------------------------------------------------------------

def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Sample from Poisson distribution using Knuth's algorithm.

    Pure Python implementation — no numpy dependency.
    """
    if lam <= 0:
        return 0
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p < limit:
            return k - 1
