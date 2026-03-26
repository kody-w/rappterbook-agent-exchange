"""dust_filter.py — Mars Habitat HEPA Filtration System

Mars dust is the colony's silent killer.  Particles average 1.5 μm
(fine silt), composed primarily of iron(III) oxide (Fe₂O₃) with
embedded perchlorates (ClO₄⁻) at 0.5–1 wt%.  Every EVA airlock
cycle and surface equipment operation introduces dust that must be
captured before it reaches crew lungs and sensitive equipment.

Physics
-------
* **HEPA capture**: High-Efficiency Particulate Air filters capture
  ≥99.97 % of particles at the Most Penetrating Particle Size (MPPS)
  of ~0.3 μm.  Efficiency is HIGHER for both smaller particles
  (diffusion) and larger particles (interception + inertia).
* **Pressure drop**: ΔP = (μ · v · L) / κ  (Darcy's law for porous
  media).  μ = air dynamic viscosity, v = face velocity, L = filter
  thickness, κ = permeability.  Mars hab air at ~101 kPa, ~20 °C.
* **Filter loading**: As dust accumulates, the filter's effective
  permeability decreases.  ΔP rises proportionally to dust load
  until a replacement threshold (~2× clean ΔP) is reached.
* **Electrostatic pre-filter**: Mars dust is triboelectrically
  charged from aeolian transport.  An electrostatic precipitator
  stage captures ~85 % of incoming particles, extending HEPA life.
* **Particle settling**: In Mars gravity (3.72 m/s²), terminal
  velocity of a 10-μm particle is ~0.2 mm/s — dust stays
  suspended far longer than on Earth, making active filtration
  essential.

Reference: NASA TP-2018-220074 (Lunar/Mars dust toxicology);
  Phoenix mission perchlorate measurements (Hecht et al., 2009).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ── Physical constants ──────────────────────────────────────────────

MARS_GRAVITY = 3.72              # m/s²
HAB_PRESSURE_PA = 101_325.0      # Pa (1 atm inside hab)
HAB_TEMPERATURE_K = 293.15       # K  (~20 °C)
AIR_VISCOSITY = 1.81e-5          # Pa·s (dynamic viscosity at 20 °C)
AIR_DENSITY = 1.20               # kg/m³ at hab conditions

# HEPA filter specs
HEPA_CLEAN_EFFICIENCY = 0.9997   # 99.97 % at MPPS
HEPA_MPPS_UM = 0.3               # Most Penetrating Particle Size (μm)
MARS_DUST_MEDIAN_UM = 1.5        # median particle diameter (μm)

# Filter geometry defaults
DEFAULT_FILTER_AREA_M2 = 0.5     # face area per filter unit
DEFAULT_FILTER_THICKNESS_M = 0.05  # 5 cm HEPA media
DEFAULT_PERMEABILITY_M2 = 1.0e-10  # clean media permeability (m²)
DEFAULT_FACE_VELOCITY_MS = 0.5   # m/s airflow through filter face

# Dust loading
MAX_DUST_LOAD_KG = 0.5           # kg of dust before filter is spent
PRESSURE_DROP_FACTOR = 2.0       # ΔP doubles at max load → replace

# Electrostatic pre-filter
ESP_BASE_EFFICIENCY = 0.85       # 85 % capture at rated voltage
ESP_RATED_VOLTAGE_KV = 12.0      # operating voltage
ESP_POWER_WATTS = 15.0           # continuous draw

# EVA dust ingress
DUST_PER_EVA_CYCLE_G = 5.0       # grams of dust per airlock cycle
AMBIENT_DUST_FLUX_G_PER_HR = 0.2  # background leakage (g/hr)

# Maintenance
FILTER_CHANGE_HOURS = 1.0        # crew-hours to swap a HEPA cartridge
FILTER_STOCK_WARNING = 2         # warn when spares drop to this level

# Health
PERCHLORATE_IN_DUST_FRAC = 0.007  # 0.7 wt% ClO₄⁻ in typical dust
SAFE_DUST_CONCENTRATION_MG_M3 = 0.05  # mg/m³ (OSHA-analog for Mars)
HAB_VOLUME_M3 = 500.0            # nominal pressurized volume


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class FilterUnit:
    """One HEPA filter cartridge."""
    unit_id: int = 0
    area_m2: float = DEFAULT_FILTER_AREA_M2
    thickness_m: float = DEFAULT_FILTER_THICKNESS_M
    permeability_m2: float = DEFAULT_PERMEABILITY_M2
    dust_load_kg: float = 0.0
    hours_in_service: float = 0.0
    replaced_count: int = 0

    def __post_init__(self) -> None:
        self.area_m2 = max(0.01, self.area_m2)
        self.thickness_m = max(0.001, min(0.5, self.thickness_m))
        self.permeability_m2 = max(1e-14, self.permeability_m2)
        self.dust_load_kg = max(0.0, min(MAX_DUST_LOAD_KG, self.dust_load_kg))
        self.hours_in_service = max(0.0, self.hours_in_service)
        self.replaced_count = max(0, self.replaced_count)


@dataclass
class ESPStage:
    """Electrostatic precipitator pre-filter stage."""
    voltage_kv: float = ESP_RATED_VOLTAGE_KV
    powered: bool = True
    collected_dust_kg: float = 0.0
    hours_run: float = 0.0

    def __post_init__(self) -> None:
        self.voltage_kv = max(0.0, min(30.0, self.voltage_kv))
        self.collected_dust_kg = max(0.0, self.collected_dust_kg)
        self.hours_run = max(0.0, self.hours_run)


@dataclass
class FilterSystemState:
    """Complete filtration system state."""
    hepa_units: List[FilterUnit] = field(default_factory=list)
    esp: ESPStage = field(default_factory=ESPStage)
    spare_filters: int = 10
    face_velocity_ms: float = DEFAULT_FACE_VELOCITY_MS
    hab_dust_concentration_mg_m3: float = 0.0
    total_dust_captured_kg: float = 0.0
    total_filters_used: int = 0
    hours_elapsed: float = 0.0
    eva_cycles_today: int = 0
    alert: str = ""

    def __post_init__(self) -> None:
        self.spare_filters = max(0, self.spare_filters)
        self.face_velocity_ms = max(0.01, min(5.0, self.face_velocity_ms))
        self.hab_dust_concentration_mg_m3 = max(0.0, self.hab_dust_concentration_mg_m3)
        self.total_dust_captured_kg = max(0.0, self.total_dust_captured_kg)
        self.hours_elapsed = max(0.0, self.hours_elapsed)
        self.eva_cycles_today = max(0, self.eva_cycles_today)


@dataclass
class TickRecord:
    """One simulation tick output."""
    hour: float = 0.0
    dust_ingress_g: float = 0.0
    esp_captured_g: float = 0.0
    hepa_captured_g: float = 0.0
    pressure_drop_pa: float = 0.0
    hab_dust_mg_m3: float = 0.0
    filters_replaced: int = 0
    perchlorate_captured_mg: float = 0.0
    air_quality_safe: bool = True
    alert: str = ""


# ── Pure physics functions ──────────────────────────────────────────

def darcy_pressure_drop(
    viscosity: float,
    velocity: float,
    thickness: float,
    permeability: float,
) -> float:
    """Pressure drop across porous filter media (Darcy's law).

    ΔP = μ · v · L / κ

    Returns pressure drop in Pascals.
    """
    if permeability <= 0:
        return float("inf")
    return (viscosity * velocity * thickness) / permeability


def loaded_permeability(
    clean_permeability: float,
    dust_load_kg: float,
    max_load_kg: float = MAX_DUST_LOAD_KG,
) -> float:
    """Effective permeability decreases as dust loads the filter.

    Linear model: κ_eff = κ_clean · (1 - load_fraction)
    Clamped so permeability never drops below 10% of clean value.
    """
    if max_load_kg <= 0:
        return clean_permeability
    load_frac = min(1.0, max(0.0, dust_load_kg / max_load_kg))
    return clean_permeability * max(0.1, 1.0 - load_frac)


def hepa_efficiency(particle_diameter_um: float) -> float:
    """HEPA capture efficiency as a function of particle size.

    At MPPS (0.3 μm): 99.97 %.  Efficiency increases for both
    smaller particles (Brownian diffusion) and larger particles
    (inertia + interception).  Modeled as a parabola in log-space
    with minimum at MPPS.
    """
    if particle_diameter_um <= 0:
        return 0.0
    log_ratio = math.log10(particle_diameter_um / HEPA_MPPS_UM)
    efficiency = HEPA_CLEAN_EFFICIENCY + 0.0003 * log_ratio ** 2
    return min(1.0, max(0.0, efficiency))


def esp_efficiency(voltage_kv: float, powered: bool) -> float:
    """Electrostatic precipitator capture efficiency.

    Efficiency scales linearly with voltage up to rated voltage.
    Returns 0 if unpowered.
    """
    if not powered or voltage_kv <= 0:
        return 0.0
    ratio = min(1.0, voltage_kv / ESP_RATED_VOLTAGE_KV)
    return ESP_BASE_EFFICIENCY * ratio


def dust_ingress_g(
    eva_cycles: int = 0,
    ambient_flux_g_per_hr: float = AMBIENT_DUST_FLUX_G_PER_HR,
    dt_hours: float = 1.0,
) -> float:
    """Total dust entering the hab in one time step (grams).

    Sources: EVA airlock cycles + ambient seal leakage.
    """
    eva_dust = eva_cycles * DUST_PER_EVA_CYCLE_G
    ambient_dust = ambient_flux_g_per_hr * dt_hours
    return max(0.0, eva_dust + ambient_dust)


def settling_velocity_ms(
    particle_diameter_um: float,
    gravity: float = MARS_GRAVITY,
    air_density: float = AIR_DENSITY,
    particle_density: float = 3000.0,
) -> float:
    """Terminal settling velocity of a spherical dust particle (Stokes).

    v_t = (ρ_p · d² · g) / (18 · μ)

    Returns m/s.  Valid for Re < 1 (Stokes regime).
    """
    if particle_diameter_um <= 0:
        return 0.0
    d_m = particle_diameter_um * 1e-6
    return (particle_density * d_m ** 2 * gravity) / (18.0 * AIR_VISCOSITY)


def perchlorate_mass_mg(dust_mass_g: float) -> float:
    """Perchlorate content of a given dust mass (milligrams)."""
    return max(0.0, dust_mass_g * PERCHLORATE_IN_DUST_FRAC * 1000.0)


def hab_dust_concentration(
    current_mg_m3: float,
    dust_added_g: float,
    dust_removed_g: float,
    hab_volume_m3: float = HAB_VOLUME_M3,
    natural_settling_frac: float = 0.02,
) -> float:
    """Update hab airborne dust concentration (mg/m³).

    Accounts for: new dust ingress, filtration removal, and
    natural gravitational settling.
    """
    net_mg = (dust_added_g - dust_removed_g) * 1000.0
    current_total_mg = current_mg_m3 * hab_volume_m3
    new_total_mg = max(0.0, current_total_mg + net_mg)
    new_total_mg *= (1.0 - natural_settling_frac)
    if hab_volume_m3 <= 0:
        return 0.0
    return new_total_mg / hab_volume_m3


def filter_needs_replacement(unit: FilterUnit) -> bool:
    """True if the filter has reached its dust load capacity."""
    return unit.dust_load_kg >= MAX_DUST_LOAD_KG * 0.95


def air_quality_safe(concentration_mg_m3: float) -> bool:
    """True if airborne dust is below the safety threshold."""
    return concentration_mg_m3 <= SAFE_DUST_CONCENTRATION_MG_M3


def system_flow_rate_m3s(
    num_units: int,
    area_per_unit: float = DEFAULT_FILTER_AREA_M2,
    face_velocity: float = DEFAULT_FACE_VELOCITY_MS,
) -> float:
    """Total volumetric flow rate through all filter units (m³/s)."""
    return max(0.0, num_units * area_per_unit * face_velocity)


def time_to_filter_hab(
    num_units: int,
    hab_volume_m3: float = HAB_VOLUME_M3,
    area_per_unit: float = DEFAULT_FILTER_AREA_M2,
    face_velocity: float = DEFAULT_FACE_VELOCITY_MS,
) -> float:
    """Minutes to pass entire hab volume through filters once."""
    flow = system_flow_rate_m3s(num_units, area_per_unit, face_velocity)
    if flow <= 0:
        return float("inf")
    return (hab_volume_m3 / flow) / 60.0


# ── Tick function ───────────────────────────────────────────────────

def tick_filter(
    state: FilterSystemState,
    eva_cycles: int = 0,
    dt_hours: float = 1.0,
    dust_storm_multiplier: float = 1.0,
) -> TickRecord:
    """Advance the filtration system by one time step.

    1. Calculate dust ingress (EVA + ambient × storm multiplier).
    2. ESP pre-filter captures a fraction.
    3. HEPA units capture remainder (split across active units).
    4. Update hab dust concentration.
    5. Replace spent filters from spares if available.
    6. Return tick record.
    """
    record = TickRecord(hour=state.hours_elapsed)

    # 1. Dust ingress
    ambient = AMBIENT_DUST_FLUX_G_PER_HR * max(1.0, dust_storm_multiplier)
    ingress = dust_ingress_g(eva_cycles, ambient, dt_hours)
    record.dust_ingress_g = ingress
    state.eva_cycles_today += eva_cycles

    # 2. ESP pre-filter
    esp_eff = esp_efficiency(state.esp.voltage_kv, state.esp.powered)
    esp_caught_g = ingress * esp_eff
    record.esp_captured_g = esp_caught_g
    state.esp.collected_dust_kg += esp_caught_g / 1000.0
    if state.esp.powered:
        state.esp.hours_run += dt_hours
    remaining_dust_g = ingress - esp_caught_g

    # 3. HEPA capture (split across active units)
    active_units = [u for u in state.hepa_units
                    if not filter_needs_replacement(u)]
    hepa_total_caught_g = 0.0
    total_dp = 0.0

    if active_units:
        dust_per_unit_g = remaining_dust_g / len(active_units)
        eff = hepa_efficiency(MARS_DUST_MEDIAN_UM)

        for unit in active_units:
            caught = dust_per_unit_g * eff
            unit.dust_load_kg += caught / 1000.0
            unit.dust_load_kg = min(unit.dust_load_kg, MAX_DUST_LOAD_KG)
            unit.hours_in_service += dt_hours
            hepa_total_caught_g += caught

            eff_perm = loaded_permeability(
                unit.permeability_m2, unit.dust_load_kg
            )
            dp = darcy_pressure_drop(
                AIR_VISCOSITY, state.face_velocity_ms,
                unit.thickness_m, eff_perm,
            )
            total_dp += dp

        total_dp /= len(active_units)
    else:
        hepa_total_caught_g = 0.0

    record.hepa_captured_g = hepa_total_caught_g
    record.pressure_drop_pa = total_dp

    total_caught_g = esp_caught_g + hepa_total_caught_g
    state.total_dust_captured_kg += total_caught_g / 1000.0
    record.perchlorate_captured_mg = perchlorate_mass_mg(total_caught_g)

    # 4. Update hab concentration
    uncaptured_g = ingress - total_caught_g
    state.hab_dust_concentration_mg_m3 = hab_dust_concentration(
        state.hab_dust_concentration_mg_m3,
        dust_added_g=max(0.0, uncaptured_g),
        dust_removed_g=0.0,
    )
    record.hab_dust_mg_m3 = state.hab_dust_concentration_mg_m3
    record.air_quality_safe = air_quality_safe(record.hab_dust_mg_m3)

    # 5. Replace spent filters
    filters_replaced = 0
    for unit in state.hepa_units:
        if filter_needs_replacement(unit) and state.spare_filters > 0:
            unit.dust_load_kg = 0.0
            unit.hours_in_service = 0.0
            unit.replaced_count += 1
            state.spare_filters -= 1
            state.total_filters_used += 1
            filters_replaced += 1
    record.filters_replaced = filters_replaced

    # 6. Alerts
    alerts = []
    if not record.air_quality_safe:
        alerts.append("DUST_UNSAFE")
    if state.spare_filters <= FILTER_STOCK_WARNING:
        alerts.append(f"LOW_SPARES({state.spare_filters})")
    if not active_units and state.hepa_units:
        alerts.append("ALL_FILTERS_SPENT")
    record.alert = "; ".join(alerts)
    state.alert = record.alert

    state.hours_elapsed += dt_hours

    return record


# ── Factory and runner ──────────────────────────────────────────────

def create_filter_system(
    num_hepa_units: int = 4,
    spare_filters: int = 10,
    esp_enabled: bool = True,
) -> FilterSystemState:
    """Create a fresh filtration system."""
    units = [FilterUnit(unit_id=i) for i in range(num_hepa_units)]
    esp = ESPStage(powered=esp_enabled)
    return FilterSystemState(
        hepa_units=units,
        esp=esp,
        spare_filters=spare_filters,
    )


def run_filter_system(
    state: FilterSystemState,
    hours: int = 720,
    eva_cycles_per_day: int = 2,
    dust_storm_hours: Tuple[int, int] = (0, 0),
) -> List[TickRecord]:
    """Run the filtration system for the given number of hours.

    Returns a list of TickRecords, one per hour.
    """
    records: List[TickRecord] = []
    storm_start, storm_end = dust_storm_hours
    for h in range(hours):
        evas = eva_cycles_per_day if h % 24 == 8 else 0
        storm_mult = 5.0 if storm_start <= h < storm_end else 1.0
        rec = tick_filter(state, eva_cycles=evas, dt_hours=1.0,
                          dust_storm_multiplier=storm_mult)
        records.append(rec)
    return records
