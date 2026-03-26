"""dust_filter.py — Mars Habitat Perchlorate Dust Filtration System.

The colony's immune system.  Every EVA drags ~50 g of Martian regolith
into the habitat through the airlock.  That dust is 0.5–1% perchlorate
(ClO₄⁻) by mass — a thyroid toxin that causes hypothyroidism, organ
damage, and death at chronic exposure above 0.1 mg/m³ airborne.

Without filtration, a 6-crew colony doing 4 EVAs/sol would accumulate
~200 g/sol of toxic dust.  In a 500 m³ habitat, that's 0.4 mg/m³
airborne in ONE SOL — already above the safe limit.

This module models a multi-stage HEPA + electrostatic + activated-carbon
filtration system that scrubs habitat air continuously.

Physics modelled
----------------
* **Particle capture** — HEPA filters capture ≥99.97% of particles ≥0.3 µm.
  Mars dust median size is 3 µm; sub-micron fraction is ~10%.  Combined
  HEPA + electrostatic achieves 99.99% capture above 0.5 µm.
* **Filter loading** — captured mass accumulates on filter media.
  Pressure drop across filter rises linearly with loading (Darcy's law).
  At max loading, flow rate drops → filtration rate degrades.
* **Perchlorate adsorption** — activated carbon bed downstream of HEPA
  adsorbs dissolved/aerosolized perchlorate.  Capacity ~50 mg ClO₄⁻
  per gram of activated carbon (GAC).  Exhaustion follows Langmuir isotherm.
* **Power consumption** — fans push air through filter stack.  Power ∝
  pressure drop.  Clean filter: ~0.2 kW.  Loaded filter: up to 0.8 kW.
* **Air turnover** — habitat volume must be filtered N times per sol
  for safe air quality.  Target: 12 turnovers/sol (every 2 hours).
* **Electrostatic pre-filter** — Mars dust is triboelectrically charged.
  Electrostatic plates capture charged particles with near-zero pressure
  drop.  Removes ~80% of coarse dust before HEPA stage.

Conservation laws enforced
--------------------------
- Mass in (dust ingress) = mass captured + mass remaining airborne
- Filter load always increases (until maintenance resets it)
- Perchlorate mass balance: ingress − captured = airborne
- Power consumption ≥ 0 and scales with pressure drop
- Airborne concentration ≥ 0, never negative

Reference:
  - Mars dust: 3 µm median, basaltic composition, 0.5–1% ClO₄⁻
  - Curiosity DAN/SAM: perchlorate 0.4–1.0 wt% at Gale Crater
  - OSHA perchlorate limit (Earth): 0.1 mg/m³ (no Mars standard yet)
  - HEPA: MIL-STD-282, ≥99.97% at 0.3 µm
  - ISS HEPA: replaced every 2–3 years, ~0.1 kW fan power
  - Activated carbon capacity: 30–80 mg/g depending on adsorbate

One tick = one sol.  Mass in grams, volume in m³, concentration in mg/m³.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── Physical constants ──────────────────────────────────────────────

# Habitat
HABITAT_VOLUME_M3 = 500.0          # pressurized volume

# Dust ingress
DUST_PER_EVA_GRAM = 50.0           # grams of regolith per returning crew
PERCHLORATE_FRACTION = 0.007       # 0.7 wt% ClO₄⁻ in Mars dust (Gale avg)

# Safety thresholds (mg/m³ airborne perchlorate)
PERCHLORATE_SAFE_MG_M3 = 0.1      # below this: safe for indefinite exposure
PERCHLORATE_WARNING_MG_M3 = 0.5   # above this: health effects within weeks
PERCHLORATE_DANGER_MG_M3 = 2.0    # above this: acute thyroid crisis

# HEPA filter
HEPA_CAPTURE_RATE = 0.9997         # fraction of particles ≥ 0.3 µm captured
HEPA_MAX_LOAD_GRAM = 2000.0        # max dust load before filter is spent
HEPA_CLEAN_PRESSURE_DROP_PA = 50.0 # clean filter ΔP
HEPA_LOADED_PRESSURE_DROP_PA = 400.0  # fully loaded ΔP

# Electrostatic pre-filter
ELECTROSTATIC_CAPTURE = 0.80       # coarse particle capture (no HEPA load)
ELECTROSTATIC_POWER_W = 15.0       # corona discharge power

# Activated carbon (perchlorate adsorption)
CARBON_BED_MASS_GRAM = 5000.0      # 5 kg activated carbon
CARBON_CAPACITY_MG_PER_G = 50.0    # mg ClO₄⁻ adsorbed per gram GAC
LANGMUIR_K = 0.02                  # Langmuir affinity constant (m³/mg)

# Fan / airflow
TARGET_TURNOVERS_PER_SOL = 12.0    # habitat air filtered 12×/sol
FAN_EFFICIENCY = 0.65              # fan electrical → pneumatic efficiency
FAN_BASE_POWER_W = 80.0            # minimum fan power (clean filter)

# Maintenance
MAINTENANCE_HEPA_RESTORE = 0.85    # fraction of HEPA load removed
MAINTENANCE_CARBON_RESTORE = 0.70  # fraction of carbon capacity restored

# Dust settling (not all dust stays airborne)
SETTLING_RATE_PER_SOL = 0.15       # fraction of airborne dust that settles


# ── State ───────────────────────────────────────────────────────────

@dataclass
class FilterState:
    """State of the multi-stage dust filtration system."""

    # Filter loading
    hepa_load_gram: float = 0.0         # accumulated dust on HEPA
    carbon_adsorbed_mg: float = 0.0     # perchlorate adsorbed on carbon bed

    # Airborne contamination
    airborne_dust_gram: float = 0.0     # dust suspended in habitat air
    airborne_perchlorate_mg: float = 0.0  # perchlorate in habitat air

    # Cumulative tracking
    total_dust_captured_gram: float = 0.0
    total_perchlorate_captured_mg: float = 0.0
    total_dust_ingress_gram: float = 0.0

    # Operational
    power_draw_w: float = 0.0
    sols_since_hepa_change: int = 0
    sols_since_carbon_change: int = 0
    maintenance_events: int = 0

    # Alerts
    alert_level: str = "nominal"  # nominal / warning / danger / critical

    def __post_init__(self) -> None:
        self.hepa_load_gram = max(0.0, min(HEPA_MAX_LOAD_GRAM, self.hepa_load_gram))
        self.carbon_adsorbed_mg = max(0.0, self.carbon_adsorbed_mg)
        self.airborne_dust_gram = max(0.0, self.airborne_dust_gram)
        self.airborne_perchlorate_mg = max(0.0, self.airborne_perchlorate_mg)
        self.total_dust_captured_gram = max(0.0, self.total_dust_captured_gram)
        self.total_perchlorate_captured_mg = max(0.0, self.total_perchlorate_captured_mg)
        self.total_dust_ingress_gram = max(0.0, self.total_dust_ingress_gram)
        self.power_draw_w = max(0.0, self.power_draw_w)


@dataclass
class FilterTickResult:
    """Result of one filtration tick (one sol)."""
    dust_ingress_gram: float = 0.0
    perchlorate_ingress_mg: float = 0.0
    dust_captured_gram: float = 0.0
    perchlorate_captured_mg: float = 0.0
    airborne_dust_mg_m3: float = 0.0
    airborne_perchlorate_mg_m3: float = 0.0
    pressure_drop_pa: float = 0.0
    power_consumed_wh: float = 0.0
    hepa_life_fraction: float = 1.0
    carbon_life_fraction: float = 1.0
    alert: str = "nominal"


# ── Pure physics functions ──────────────────────────────────────────

def dust_ingress(eva_count: int, crew_per_eva: int = 2,
                 dust_storm_factor: float = 1.0) -> float:
    """Calculate total dust entering habitat in one sol (grams).

    Parameters
    ----------
    eva_count : int
        Number of EVA sorties this sol.
    crew_per_eva : int
        Crew members returning per EVA.
    dust_storm_factor : float
        Multiplier during dust storms (1.0 = normal, up to 5.0).

    Returns
    -------
    float
        Total dust ingress in grams.  Always ≥ 0.
    """
    eva_count = max(0, eva_count)
    crew_per_eva = max(0, crew_per_eva)
    dust_storm_factor = max(0.0, min(5.0, dust_storm_factor))
    return eva_count * crew_per_eva * DUST_PER_EVA_GRAM * dust_storm_factor


def perchlorate_from_dust(dust_gram: float) -> float:
    """Perchlorate mass (mg) in a given mass of Mars dust.

    Uses global PERCHLORATE_FRACTION (0.7 wt%).
    1 g dust × 0.007 = 0.007 g = 7 mg perchlorate.
    """
    return max(0.0, dust_gram * PERCHLORATE_FRACTION * 1000.0)


def pressure_drop_pa(hepa_load_gram: float) -> float:
    """Filter pressure drop (Pa) as function of HEPA dust loading.

    Linear interpolation between clean and fully-loaded pressure drop.
    Darcy's law: ΔP ∝ accumulated cake thickness ∝ mass loaded.
    """
    load_fraction = max(0.0, min(1.0, hepa_load_gram / HEPA_MAX_LOAD_GRAM))
    return (HEPA_CLEAN_PRESSURE_DROP_PA
            + load_fraction * (HEPA_LOADED_PRESSURE_DROP_PA
                               - HEPA_CLEAN_PRESSURE_DROP_PA))


def fan_power_watts(dp_pa: float) -> float:
    """Fan electrical power (W) to push air through filter at given ΔP.

    Total = base power + (Q · ΔP) / η_fan
    Q = HABITAT_VOLUME_M3 * TARGET_TURNOVERS_PER_SOL / 86400  (m³/s)
    Base power covers motor idle, electronics, sensors.
    ΔP component is additional work to push air through filter cake.
    """
    flow_rate_m3_s = (HABITAT_VOLUME_M3 * TARGET_TURNOVERS_PER_SOL
                      / (24.65 * 3600.0))  # Mars sol = 24.65 hours
    dp_pa = max(0.0, dp_pa)
    pneumatic_power = flow_rate_m3_s * dp_pa / max(0.01, FAN_EFFICIENCY)
    return FAN_BASE_POWER_W + pneumatic_power


def electrostatic_capture(dust_gram: float) -> float:
    """Dust captured by electrostatic pre-filter (grams).

    Removes coarse charged particles before they reach HEPA.
    Mars dust is triboelectrically charged → high capture efficiency.
    """
    return max(0.0, dust_gram * ELECTROSTATIC_CAPTURE)


def hepa_capture(dust_gram: float, load_fraction: float) -> float:
    """Dust captured by HEPA filter (grams).

    Capture efficiency degrades slightly as filter loads (cake buildup
    actually INCREASES capture but at the cost of pressure drop).
    At >90% loading, bypass channels form → efficiency drops.

    Parameters
    ----------
    dust_gram : float
        Dust reaching HEPA (after electrostatic pre-filter).
    load_fraction : float
        Current HEPA loading as fraction of max capacity (0..1).
    """
    dust_gram = max(0.0, dust_gram)
    load_fraction = max(0.0, min(1.0, load_fraction))
    if load_fraction < 0.9:
        # Cake actually improves capture slightly
        efficiency = min(1.0, HEPA_CAPTURE_RATE + load_fraction * 0.0003)
    else:
        # Bypass channels degrade capture
        bypass_penalty = (load_fraction - 0.9) * 10.0  # 0..1 over last 10%
        efficiency = HEPA_CAPTURE_RATE * (1.0 - 0.2 * bypass_penalty)
    return dust_gram * max(0.0, min(1.0, efficiency))


def langmuir_adsorption(perchlorate_mg: float, carbon_adsorbed_mg: float,
                        carbon_bed_mass_g: float = CARBON_BED_MASS_GRAM) -> float:
    """Perchlorate adsorbed by activated carbon this sol (mg).

    Langmuir isotherm: q = q_max · K·C / (1 + K·C)
    where q_max = remaining capacity, C = concentration, K = affinity.

    Returns mg of perchlorate adsorbed (always ≤ available + remaining capacity).
    """
    perchlorate_mg = max(0.0, perchlorate_mg)
    carbon_adsorbed_mg = max(0.0, carbon_adsorbed_mg)
    carbon_bed_mass_g = max(0.0, carbon_bed_mass_g)

    max_capacity_mg = carbon_bed_mass_g * CARBON_CAPACITY_MG_PER_G
    remaining_capacity = max(0.0, max_capacity_mg - carbon_adsorbed_mg)

    if remaining_capacity <= 0.0 or perchlorate_mg <= 0.0:
        return 0.0

    # Langmuir: fraction adsorbed depends on concentration and affinity
    concentration_proxy = perchlorate_mg / max(1.0, HABITAT_VOLUME_M3)
    langmuir_fraction = (LANGMUIR_K * concentration_proxy
                         / (1.0 + LANGMUIR_K * concentration_proxy))

    adsorbed = perchlorate_mg * langmuir_fraction
    return min(adsorbed, remaining_capacity, perchlorate_mg)


def airborne_concentration_mg_m3(mass_mg: float,
                                 volume_m3: float = HABITAT_VOLUME_M3) -> float:
    """Convert mass of airborne contaminant (mg) to concentration (mg/m³)."""
    volume_m3 = max(0.01, volume_m3)
    return max(0.0, mass_mg / volume_m3)


def assess_alert(perchlorate_mg_m3: float) -> str:
    """Determine alert level from airborne perchlorate concentration."""
    if perchlorate_mg_m3 >= PERCHLORATE_DANGER_MG_M3:
        return "critical"
    elif perchlorate_mg_m3 >= PERCHLORATE_WARNING_MG_M3:
        return "danger"
    elif perchlorate_mg_m3 >= PERCHLORATE_SAFE_MG_M3:
        return "warning"
    return "nominal"


def hepa_life_remaining(hepa_load_gram: float) -> float:
    """Fraction of HEPA filter life remaining (0..1)."""
    return max(0.0, 1.0 - hepa_load_gram / HEPA_MAX_LOAD_GRAM)


def carbon_life_remaining(carbon_adsorbed_mg: float,
                          carbon_bed_mass_g: float = CARBON_BED_MASS_GRAM) -> float:
    """Fraction of carbon bed capacity remaining (0..1)."""
    max_cap = carbon_bed_mass_g * CARBON_CAPACITY_MG_PER_G
    if max_cap <= 0.0:
        return 0.0
    return max(0.0, 1.0 - carbon_adsorbed_mg / max_cap)


# ── Tick function ───────────────────────────────────────────────────

def tick_filter(state: FilterState, eva_count: int = 2,
                crew_per_eva: int = 2,
                dust_storm_factor: float = 1.0,
                maintenance: bool = False) -> tuple[FilterState, FilterTickResult]:
    """Advance dust filtration system by one sol.

    Parameters
    ----------
    state : FilterState
        Current system state.
    eva_count : int
        Number of EVA sorties this sol.
    crew_per_eva : int
        Crew returning per EVA.
    dust_storm_factor : float
        Dust storm multiplier (1.0 = normal).
    maintenance : bool
        If True, perform filter maintenance this sol.

    Returns
    -------
    tuple[FilterState, FilterTickResult]
        Updated state and tick result.
    """
    result = FilterTickResult()

    # ── Maintenance (if requested) ──────────────────────────────────
    if maintenance:
        old_hepa = state.hepa_load_gram
        state.hepa_load_gram *= (1.0 - MAINTENANCE_HEPA_RESTORE)
        state.hepa_load_gram = max(0.0, state.hepa_load_gram)

        max_carbon_cap = CARBON_BED_MASS_GRAM * CARBON_CAPACITY_MG_PER_G
        restored = state.carbon_adsorbed_mg * MAINTENANCE_CARBON_RESTORE
        state.carbon_adsorbed_mg = max(0.0, state.carbon_adsorbed_mg - restored)

        state.sols_since_hepa_change = 0
        state.sols_since_carbon_change = 0
        state.maintenance_events += 1

    # ── Dust ingress ────────────────────────────────────────────────
    ingress = dust_ingress(eva_count, crew_per_eva, dust_storm_factor)
    perchlorate_in = perchlorate_from_dust(ingress)
    result.dust_ingress_gram = ingress
    result.perchlorate_ingress_mg = perchlorate_in
    state.total_dust_ingress_gram += ingress

    # ── Stage 1: Electrostatic pre-filter ───────────────────────────
    es_captured = electrostatic_capture(ingress)
    dust_to_hepa = ingress - es_captured

    # ── Stage 2: HEPA filter ────────────────────────────────────────
    load_frac = state.hepa_load_gram / HEPA_MAX_LOAD_GRAM
    hepa_caught = hepa_capture(dust_to_hepa, load_frac)
    dust_escaped = dust_to_hepa - hepa_caught

    # Update HEPA loading
    total_physical_captured = es_captured + hepa_caught
    state.hepa_load_gram = min(HEPA_MAX_LOAD_GRAM,
                               state.hepa_load_gram + hepa_caught)

    # ── Stage 3: Carbon bed (perchlorate only) ──────────────────────
    # Perchlorate proportional to dust that escaped filtration
    if ingress > 0:
        escaped_fraction = dust_escaped / ingress
    else:
        escaped_fraction = 0.0

    perchlorate_airborne_new = perchlorate_in * escaped_fraction
    # Also filter perchlorate already in the air
    total_perchlorate_in_air = state.airborne_perchlorate_mg + perchlorate_airborne_new

    carbon_captured = langmuir_adsorption(total_perchlorate_in_air,
                                          state.carbon_adsorbed_mg)
    state.carbon_adsorbed_mg += carbon_captured
    state.total_perchlorate_captured_mg += carbon_captured

    # ── Airborne balance ────────────────────────────────────────────
    # Dust: new escaped + existing - settling
    state.airborne_dust_gram += dust_escaped
    settled = state.airborne_dust_gram * SETTLING_RATE_PER_SOL
    state.airborne_dust_gram = max(0.0, state.airborne_dust_gram - settled)

    # Perchlorate: new escaped + existing - carbon captured - settling
    state.airborne_perchlorate_mg = max(
        0.0,
        total_perchlorate_in_air - carbon_captured
        - total_perchlorate_in_air * SETTLING_RATE_PER_SOL
    )

    # ── Cumulative captures ─────────────────────────────────────────
    state.total_dust_captured_gram += total_physical_captured
    result.dust_captured_gram = total_physical_captured
    result.perchlorate_captured_mg = carbon_captured

    # ── Pressure drop and power ─────────────────────────────────────
    dp = pressure_drop_pa(state.hepa_load_gram)
    fan_w = fan_power_watts(dp)
    total_power_w = fan_w + ELECTROSTATIC_POWER_W
    state.power_draw_w = total_power_w

    result.pressure_drop_pa = dp
    result.power_consumed_wh = total_power_w * 24.65  # watts × hours/sol

    # ── Life tracking ───────────────────────────────────────────────
    result.hepa_life_fraction = hepa_life_remaining(state.hepa_load_gram)
    result.carbon_life_fraction = carbon_life_remaining(state.carbon_adsorbed_mg)

    # ── Alert level ─────────────────────────────────────────────────
    perc_conc = airborne_concentration_mg_m3(state.airborne_perchlorate_mg)
    result.airborne_dust_mg_m3 = airborne_concentration_mg_m3(
        state.airborne_dust_gram * 1000.0)  # g → mg
    result.airborne_perchlorate_mg_m3 = perc_conc
    alert = assess_alert(perc_conc)

    # Override to critical if HEPA is spent
    if result.hepa_life_fraction <= 0.0:
        alert = "critical"

    result.alert = alert
    state.alert_level = alert

    # ── Age tracking ────────────────────────────────────────────────
    state.sols_since_hepa_change += 1
    state.sols_since_carbon_change += 1

    return state, result


# ── Factory ─────────────────────────────────────────────────────────

def create_filter_system() -> FilterState:
    """Create a new dust filtration system in pristine condition."""
    return FilterState()
