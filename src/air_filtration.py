"""air_filtration.py — Mars Habitat HEPA/Electrostatic Air Filtration System.

Models particulate removal from habitat atmosphere.  Mars dust is toxic:
iron-oxide particles (1–100 μm) coated with perchlorates (ClO₄⁻), fine
silica, and electrostatically charged from triboelectric effects.

Without filtration, crew inhale ~0.05 mg/m³ respirable dust per sol
during normal operations, rising to 5+ mg/m³ after EVA ingress events.
NASA PEL (Permissible Exposure Limit) for lunar/Mars dust: 0.3 mg/m³
for particles < 10 μm, 0.05 mg/m³ for < 2.5 μm.

Physics modelled
----------------
* **HEPA filtration** — H14-grade (99.995% at MPPS ~0.12 μm).  For Mars
  dust (median ~3 μm), capture efficiency > 99.99%.  Efficiency modelled
  via the single-fiber efficiency equation (diffusion + interception +
  impaction).
* **Electrostatic precipitator (ESP)** — pre-stage for coarse particles
  (> 10 μm).  Uses corona discharge to charge particles, collected on
  grounded plates.  Deutsch–Anderson equation: η = 1 − exp(−w·A/Q).
* **Pressure drop** — clean filter: ΔP = (μ·v·t·α) / (d_f²).  Loaded
  filter: ΔP increases with dust cake via Ergun-type correlation.
  Fan must overcome ΔP — power = Q·ΔP / η_fan.
* **Filter loading** — mass accumulates on filter media.  At capacity,
  filter must be replaced or back-pulsed.  Loading rate = concentration
  × flow rate × capture efficiency.
* **Dust generation events** — EVA airlock cycling injects ~50–200 mg
  of dust per event.  Normal background: habitat off-gassing, skin
  flakes, fabric fibers (~5 mg/sol baseline).
* **Conservation of mass** — dust removed from air = dust on filter.
  Total dust in system (air + filter + settled) is conserved.

Reference:
  - NASA HSRP Dust Toxicity studies (2018–2024)
  - ISS HEPA filter: 99.97% at 0.3 μm, replaced every 2.5 years
  - Apollo 17: ~10 mg/m³ peak cabin dust after EVA
  - Mars DRA 5.0: Habitat volume 200–400 m³, 4–6 crew

One tick = one sol.  Mass in mg, volume in m³, flow in m³/hr, pressure in Pa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


# ── Physical constants ──────────────────────────────────────────────────────

MARS_DUST_MEDIAN_UM = 3.0          # Median particle diameter (μm)
MARS_DUST_DENSITY_KG_M3 = 3930.0   # Iron-oxide dominant (hematite)
AIR_VISCOSITY_PA_S = 1.8e-5        # Dynamic viscosity at ~20°C, ~70 kPa
SECONDS_PER_SOL = 88775.0          # Mars sol duration (s)
HOURS_PER_SOL = SECONDS_PER_SOL / 3600.0  # ~24.66 hr

# ── HEPA filter parameters ──────────────────────────────────────────────────

FIBER_DIAMETER_UM = 2.0            # Glass microfiber diameter
FILTER_SOLIDITY = 0.05             # Volume fraction of fibers (α)
FILTER_THICKNESS_MM = 50.0         # Media thickness (mm)
HEPA_BASE_EFFICIENCY = 0.9999      # At median Mars dust size (>>MPPS)
HEPA_CLEAN_DP_PA = 250.0           # Clean filter pressure drop at design flow

# ── ESP parameters ──────────────────────────────────────────────────────────

ESP_DRIFT_VELOCITY_M_S = 0.05      # Effective particle drift velocity (m/s)
ESP_PLATE_AREA_M2 = 4.0            # Total collection plate area (m²)
ESP_POWER_W = 50.0                 # Corona discharge power (W)

# ── System parameters ──────────────────────────────────────────────────────

DEFAULT_HAB_VOLUME_M3 = 300.0      # Habitat pressurized volume (m³)
DEFAULT_FLOW_M3_HR = 150.0         # Design air flow rate (m³/hr)
FAN_EFFICIENCY = 0.65              # Fan mechanical efficiency
FILTER_CAPACITY_MG = 50000.0       # Total dust capacity before replacement (mg)
BASELINE_DUST_MG_SOL = 5.0         # Normal internal dust generation (mg/sol)
EVA_DUST_MG = 120.0                # Dust per EVA ingress event (mg)
NASA_PEL_MG_M3 = 0.3              # Permissible exposure limit (mg/m³)
CRITICAL_DUST_MG_M3 = 1.0         # Emergency threshold (mg/m³)


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class FilterState:
    """Mutable state of the habitat air filtration system."""
    sol: int = 0
    airborne_dust_mg: float = 15.0       # Dust mass suspended in hab air (mg)
    filter_load_mg: float = 0.0          # Dust accumulated on HEPA filter (mg)
    settled_dust_mg: float = 0.0         # Dust settled on surfaces (mg)
    hab_volume_m3: float = DEFAULT_HAB_VOLUME_M3
    flow_rate_m3_hr: float = DEFAULT_FLOW_M3_HR
    hepa_online: bool = True
    esp_online: bool = True
    filter_replaced_count: int = 0
    disposed_dust_mg: float = 0.0        # Dust removed from system via filter swap
    fan_power_w: float = 0.0             # Current fan power draw (W)
    hepa_efficiency: float = HEPA_BASE_EFFICIENCY
    esp_efficiency: float = 0.0          # Computed per tick
    pressure_drop_pa: float = HEPA_CLEAN_DP_PA
    cumulative_eva_events: int = 0


@dataclass
class FiltrationRecord:
    """Immutable record of one sol's filtration activity."""
    sol: int
    airborne_dust_mg: float
    concentration_mg_m3: float
    filter_load_mg: float
    filter_life_fraction: float
    settled_dust_mg: float
    dust_removed_mg: float
    dust_added_mg: float
    pressure_drop_pa: float
    fan_power_kwh: float
    hepa_online: bool
    esp_online: bool
    within_pel: bool
    eva_events_today: int


# ── Pure functions ──────────────────────────────────────────────────────────

def dust_concentration(airborne_mg: float, volume_m3: float) -> float:
    """Calculate airborne dust concentration (mg/m³)."""
    if volume_m3 <= 0:
        return 0.0
    return max(airborne_mg / volume_m3, 0.0)


def esp_capture_efficiency(
    flow_m3_hr: float,
    drift_velocity: float = ESP_DRIFT_VELOCITY_M_S,
    plate_area: float = ESP_PLATE_AREA_M2,
) -> float:
    """Deutsch–Anderson equation for ESP efficiency.

    η = 1 − exp(−w·A / Q)
    w = drift velocity (m/s), A = plate area (m²), Q = flow (m³/s).
    """
    if flow_m3_hr <= 0:
        return 0.0
    flow_m3_s = flow_m3_hr / 3600.0
    exponent = -(drift_velocity * plate_area) / flow_m3_s
    exponent = max(exponent, -50.0)  # Numerical stability
    return 1.0 - math.exp(exponent)


def hepa_loaded_efficiency(base_efficiency: float, load_fraction: float) -> float:
    """HEPA efficiency increases slightly with loading (dust cake effect).

    A loaded filter catches more because the dust cake itself acts as
    additional filtration media.  Saturates at 99.999%.
    """
    boost = load_fraction * 0.00005  # Tiny improvement
    return min(base_efficiency + boost, 0.99999)


def pressure_drop(
    clean_dp_pa: float,
    load_fraction: float,
) -> float:
    """Filter pressure drop increases with dust loading.

    Empirical: ΔP = ΔP_clean × (1 + k·load_fraction²)
    where k ≈ 3.0 (typical HEPA loading curve).
    """
    k = 3.0
    return clean_dp_pa * (1.0 + k * load_fraction ** 2)


def fan_power_kwh(dp_pa: float, flow_m3_hr: float, efficiency: float = FAN_EFFICIENCY) -> float:
    """Fan electrical power per sol (kWh).

    P = Q·ΔP / η  (fluid power / fan efficiency)
    """
    if efficiency <= 0 or flow_m3_hr <= 0:
        return 0.0
    flow_m3_s = flow_m3_hr / 3600.0
    watts = (flow_m3_s * dp_pa) / efficiency
    kwh = watts * HOURS_PER_SOL / 1000.0
    return max(kwh, 0.0)


def dust_removed_per_sol(
    airborne_mg: float,
    volume_m3: float,
    flow_m3_hr: float,
    hepa_eff: float,
    esp_eff: float,
    hepa_on: bool,
    esp_on: bool,
) -> float:
    """Calculate dust removed from air per sol (mg).

    The air volume passes through the filter system multiple times per sol.
    Turnovers = (flow_rate × hours_per_sol) / volume.
    Each pass removes a fraction.  Result: exponential decay.

    removed = airborne × (1 − (1 − η_combined)^turnovers)
    """
    if volume_m3 <= 0 or flow_m3_hr <= 0 or airborne_mg <= 0:
        return 0.0

    # Combined efficiency (series: ESP then HEPA)
    eta_esp = esp_eff if esp_on else 0.0
    eta_hepa = hepa_eff if hepa_on else 0.0
    # Particles pass ESP first, then HEPA
    eta_combined = 1.0 - (1.0 - eta_esp) * (1.0 - eta_hepa)
    eta_combined = max(min(eta_combined, 0.99999), 0.0)

    turnovers = (flow_m3_hr * HOURS_PER_SOL) / volume_m3
    # Fraction remaining after all turnovers
    fraction_remaining = (1.0 - eta_combined) ** turnovers
    # Clamp for numerical stability
    fraction_remaining = max(fraction_remaining, 0.0)

    removed = airborne_mg * (1.0 - fraction_remaining)
    return max(min(removed, airborne_mg), 0.0)


def natural_settling(airborne_mg: float) -> float:
    """Dust that settles naturally per sol (gravity + electrostatic deposition).

    Coarse particles (>20 μm) settle under Mars gravity (3.72 m/s²).
    Rate: ~2% of airborne mass per sol for the Mars size distribution.
    """
    settling_rate = 0.02
    return airborne_mg * settling_rate


def needs_filter_replacement(load_mg: float, capacity_mg: float = FILTER_CAPACITY_MG) -> bool:
    """Check if HEPA filter needs replacement."""
    return load_mg >= capacity_mg


def replace_filter(state: FilterState) -> None:
    """Replace the HEPA filter — resets load and pressure drop.

    The dust on the old filter is disposed (removed from the system).
    """
    state.disposed_dust_mg += state.filter_load_mg
    state.filter_load_mg = 0.0
    state.pressure_drop_pa = HEPA_CLEAN_DP_PA
    state.hepa_efficiency = HEPA_BASE_EFFICIENCY
    state.filter_replaced_count += 1


# ── Tick function ───────────────────────────────────────────────────────────

def tick_filtration(
    state: FilterState,
    eva_events: int = 0,
    extra_dust_mg: float = 0.0,
) -> FiltrationRecord:
    """Advance filtration system by one sol.

    Args:
        state: Mutable filtration state.
        eva_events: Number of EVA ingress events this sol.
        extra_dust_mg: Additional dust from other sources (mg).

    Returns:
        FiltrationRecord for this sol.
    """
    state.sol += 1

    # ── Track mass before this sol ──
    total_before = (state.airborne_dust_mg + state.filter_load_mg
                    + state.settled_dust_mg + state.disposed_dust_mg)

    # ── Dust generation ──
    dust_added = BASELINE_DUST_MG_SOL + (eva_events * EVA_DUST_MG) + max(extra_dust_mg, 0.0)
    state.airborne_dust_mg += dust_added
    state.cumulative_eva_events += eva_events

    # ── ESP efficiency (computed fresh each tick) ──
    state.esp_efficiency = esp_capture_efficiency(state.flow_rate_m3_hr)

    # ── HEPA efficiency (accounts for loading) ──
    load_frac = state.filter_load_mg / FILTER_CAPACITY_MG if FILTER_CAPACITY_MG > 0 else 0.0
    load_frac = min(load_frac, 1.0)
    state.hepa_efficiency = hepa_loaded_efficiency(HEPA_BASE_EFFICIENCY, load_frac)

    # ── Filter removal ──
    removed = dust_removed_per_sol(
        state.airborne_dust_mg,
        state.hab_volume_m3,
        state.flow_rate_m3_hr,
        state.hepa_efficiency,
        state.esp_efficiency,
        state.hepa_online,
        state.esp_online,
    )
    state.airborne_dust_mg -= removed
    state.filter_load_mg += removed

    # ── Natural settling ──
    settled = natural_settling(state.airborne_dust_mg)
    state.airborne_dust_mg -= settled
    state.settled_dust_mg += settled

    # ── Clamp ──
    state.airborne_dust_mg = max(state.airborne_dust_mg, 0.0)

    # ── Pressure drop & fan power ──
    load_frac = state.filter_load_mg / FILTER_CAPACITY_MG if FILTER_CAPACITY_MG > 0 else 0.0
    load_frac = min(load_frac, 1.0)
    state.pressure_drop_pa = pressure_drop(HEPA_CLEAN_DP_PA, load_frac)
    power = fan_power_kwh(state.pressure_drop_pa, state.flow_rate_m3_hr)
    if state.esp_online:
        power += ESP_POWER_W * HOURS_PER_SOL / 1000.0  # ESP draws constant power
    state.fan_power_w = power * 1000.0 / HOURS_PER_SOL  # Average watts

    # ── Auto-replace if at capacity ──
    if needs_filter_replacement(state.filter_load_mg):
        replace_filter(state)

    # ── Conservation check ──
    total_after = (state.airborne_dust_mg + state.filter_load_mg
                   + state.settled_dust_mg + state.disposed_dust_mg)
    expected = total_before + dust_added
    # Allow tiny floating-point tolerance
    assert abs(total_after - expected) < 1e-6, (
        f"Mass conservation violated: {total_after:.6f} != {expected:.6f}"
    )

    # ── Build record ──
    conc = dust_concentration(state.airborne_dust_mg, state.hab_volume_m3)
    load_frac_final = state.filter_load_mg / FILTER_CAPACITY_MG if FILTER_CAPACITY_MG > 0 else 0.0

    return FiltrationRecord(
        sol=state.sol,
        airborne_dust_mg=state.airborne_dust_mg,
        concentration_mg_m3=conc,
        filter_load_mg=state.filter_load_mg,
        filter_life_fraction=1.0 - min(load_frac_final, 1.0),
        settled_dust_mg=state.settled_dust_mg,
        dust_removed_mg=removed,
        dust_added_mg=dust_added,
        pressure_drop_pa=state.pressure_drop_pa,
        fan_power_kwh=power,
        hepa_online=state.hepa_online,
        esp_online=state.esp_online,
        within_pel=conc <= NASA_PEL_MG_M3,
        eva_events_today=eva_events,
    )


# ── Factory & runner ────────────────────────────────────────────────────────

def make_filtration(
    hab_volume_m3: float = DEFAULT_HAB_VOLUME_M3,
    flow_rate_m3_hr: float = DEFAULT_FLOW_M3_HR,
    initial_dust_mg: float = 15.0,
) -> FilterState:
    """Create a new filtration system with given specs."""
    return FilterState(
        hab_volume_m3=hab_volume_m3,
        flow_rate_m3_hr=flow_rate_m3_hr,
        airborne_dust_mg=initial_dust_mg,
    )


def run_filtration(
    state: FilterState,
    sols: int = 100,
    eva_per_sol: int = 0,
) -> List[FiltrationRecord]:
    """Run filtration system for multiple sols."""
    records: List[FiltrationRecord] = []
    for _ in range(sols):
        record = tick_filtration(state, eva_events=eva_per_sol)
        records.append(record)
    return records
