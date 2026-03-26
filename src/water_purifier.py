"""water_purifier.py — Mars Colony Water Purification System

Removes perchlorates, heavy metals, and microbial contaminants from
raw Mars water extracted via ice mining or regolith baking.  Three-stage
treatment: ion-exchange resin (perchlorate removal), reverse-osmosis
membrane (dissolved solids), and UV-C sterilization (pathogen kill).

Physics
-------
* **Ion-exchange perchlorate removal**: Strong-base anion resin
  (Type I quaternary amine) selectively binds ClO₄⁻.  Capacity
  ~0.8 eq/L resin.  Breakthrough at ~500 bed volumes (BV) for
  1 mg/L influent.  Regeneration with 1 M NaCl brine restores
  ~90% capacity per cycle.
* **Reverse osmosis**: Transmembrane pressure ΔP drives water across
  a semi-permeable membrane.  Flux J = A·(ΔP − Δπ) where A is the
  membrane permeability (~2.5 L/m²/h/bar for TFC polyamide) and
  Δπ is osmotic pressure difference.  Rejection R = 1 − Cp/Cf.
* **UV-C sterilization**: 254 nm germicidal dose.  Log-reduction =
  dose / D₁₀ where D₁₀ ≈ 7 mJ/cm² for E. coli, ~40 mJ/cm² for
  Cryptosporidium.  Colony standard: 40 mJ/cm² (4-log kill).
* **Perchlorate toxicity**: NASA SMAC (Spacecraft Maximum Allowable
  Concentration) for drinking water = 0.015 mg/L.  Raw Mars water
  from regolith baking can contain 200–2000 mg/L ClO₄⁻.

Reference: NASA TP-2015-218570, JSC Water Recovery System specs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


# ── Physical constants ──────────────────────────────────────────────

PERCHLORATE_SMAC_MG_L = 0.015      # mg/L — NASA drinking water limit
RAW_PERCHLORATE_RANGE = (200.0, 2000.0)  # mg/L in regolith-baked water

RO_PERMEABILITY = 2.5              # L/m²/h/bar (TFC polyamide)
RO_NOMINAL_PRESSURE_BAR = 15.0     # operating pressure
RO_OSMOTIC_COEFF = 0.7             # bar per g/L TDS
RO_REJECTION = 0.97                # salt rejection ratio

IX_CAPACITY_EQ_L = 0.8             # ion-exchange resin capacity
IX_BED_VOLUMES_TO_BREAKTHROUGH = 500
IX_REGEN_RECOVERY = 0.90           # capacity recovery per regen cycle

UV_D10_MJ_CM2 = 7.0               # D10 for E. coli
UV_TARGET_DOSE_MJ_CM2 = 40.0      # 4-log Crypto kill
UV_LAMP_POWER_W = 40.0             # per lamp
UV_LAMP_LIFE_HOURS = 9000.0        # rated lamp lifetime

WATER_DENSITY_KG_L = 1.0           # approximation for dilute solutions
PUMP_EFFICIENCY = 0.75             # RO high-pressure pump efficiency


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class WaterBatch:
    """One batch of water processed through the purifier."""
    batch_id: int = 0
    volume_liters: float = 0.0
    influent_perchlorate_mg_l: float = 0.0
    effluent_perchlorate_mg_l: float = 0.0
    influent_tds_mg_l: float = 0.0
    effluent_tds_mg_l: float = 0.0
    uv_dose_mj_cm2: float = 0.0
    log_reduction: float = 0.0
    potable: bool = False
    energy_wh: float = 0.0

    def __post_init__(self) -> None:
        self.volume_liters = max(0.0, self.volume_liters)
        self.influent_perchlorate_mg_l = max(0.0, self.influent_perchlorate_mg_l)
        self.effluent_perchlorate_mg_l = _clamp(self.effluent_perchlorate_mg_l, 0.0, self.influent_perchlorate_mg_l)
        self.influent_tds_mg_l = max(0.0, self.influent_tds_mg_l)
        self.effluent_tds_mg_l = _clamp(self.effluent_tds_mg_l, 0.0, self.influent_tds_mg_l)
        self.uv_dose_mj_cm2 = max(0.0, self.uv_dose_mj_cm2)
        self.log_reduction = max(0.0, self.log_reduction)
        self.energy_wh = max(0.0, self.energy_wh)


@dataclass
class PurifierState:
    """Full state of the three-stage water purification system."""
    # Ion-exchange stage
    ix_resin_volume_l: float = 50.0       # resin bed volume
    ix_bed_volumes_processed: float = 0.0  # cumulative BV since last regen
    ix_regen_cycles: int = 0
    ix_capacity_fraction: float = 1.0      # 0..1 remaining capacity

    # Reverse-osmosis stage
    ro_membrane_area_m2: float = 10.0
    ro_pressure_bar: float = 15.0
    ro_hours: float = 0.0                  # cumulative membrane hours
    ro_membrane_life_hours: float = 8760.0  # ~1 year
    ro_fouling_factor: float = 1.0         # 1.0 = clean, 0.0 = fully fouled

    # UV-C stage
    uv_lamp_count: int = 2
    uv_lamp_hours: float = 0.0
    uv_lamp_life_hours: float = 9000.0

    # System totals
    total_liters_processed: float = 0.0
    total_batches: int = 0
    total_energy_wh: float = 0.0
    batches: List[WaterBatch] = field(default_factory=list)

    # Temperature affects RO flux and UV output
    water_temperature_c: float = 15.0

    def __post_init__(self) -> None:
        self.ix_resin_volume_l = _clamp(self.ix_resin_volume_l, 1.0, 500.0)
        self.ix_bed_volumes_processed = max(0.0, self.ix_bed_volumes_processed)
        self.ix_regen_cycles = max(0, self.ix_regen_cycles)
        self.ix_capacity_fraction = _clamp(self.ix_capacity_fraction, 0.0, 1.0)

        self.ro_membrane_area_m2 = _clamp(self.ro_membrane_area_m2, 0.1, 200.0)
        self.ro_pressure_bar = _clamp(self.ro_pressure_bar, 1.0, 80.0)
        self.ro_hours = max(0.0, self.ro_hours)
        self.ro_membrane_life_hours = _clamp(self.ro_membrane_life_hours, 100.0, 50000.0)
        self.ro_fouling_factor = _clamp(self.ro_fouling_factor, 0.0, 1.0)

        self.uv_lamp_count = _clamp_int(self.uv_lamp_count, 1, 20)
        self.uv_lamp_hours = max(0.0, self.uv_lamp_hours)
        self.uv_lamp_life_hours = _clamp(self.uv_lamp_life_hours, 100.0, 50000.0)

        self.water_temperature_c = _clamp(self.water_temperature_c, 1.0, 45.0)
        self.total_liters_processed = max(0.0, self.total_liters_processed)
        self.total_batches = max(0, self.total_batches)
        self.total_energy_wh = max(0.0, self.total_energy_wh)


# ── Helpers ─────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ── Ion-exchange physics ────────────────────────────────────────────

def ix_removal_efficiency(bed_volumes_processed: float,
                          capacity_fraction: float) -> float:
    """Perchlorate removal efficiency of ion-exchange resin.

    Returns fraction removed (0..1).  Efficiency drops sharply
    near breakthrough (~500 BV for fresh resin).
    """
    if capacity_fraction <= 0.0:
        return 0.0
    effective_bv_limit = IX_BED_VOLUMES_TO_BREAKTHROUGH * capacity_fraction
    if effective_bv_limit <= 0.0:
        return 0.0
    ratio = bed_volumes_processed / effective_bv_limit
    if ratio < 0.8:
        return 0.9995  # near-complete removal before breakthrough
    elif ratio < 1.0:
        # Steep S-curve breakthrough
        t = (ratio - 0.8) / 0.2
        return 0.9995 * (1.0 - t * t)
    else:
        # Past breakthrough — residual removal only
        return max(0.0, 0.1 * math.exp(-(ratio - 1.0)))


def ix_bed_volumes_used(volume_liters: float,
                        resin_volume_l: float) -> float:
    """Compute bed volumes consumed by a batch."""
    if resin_volume_l <= 0.0:
        return 0.0
    return volume_liters / resin_volume_l


def ix_regen_capacity_after(cycles: int) -> float:
    """Resin capacity fraction after N regeneration cycles.

    Each cycle recovers ~90% of previous capacity.
    """
    return IX_REGEN_RECOVERY ** max(0, cycles)


# ── Reverse-osmosis physics ────────────────────────────────────────

def ro_flux(pressure_bar: float, tds_g_l: float,
            fouling_factor: float, temperature_c: float) -> float:
    """RO permeate flux in L/m²/h.

    J = A · (ΔP − Δπ) · fouling · temp_correction
    Temperature correction: flux increases ~3% per °C above 25°C.
    """
    osmotic_pressure = RO_OSMOTIC_COEFF * tds_g_l
    net_pressure = max(0.0, pressure_bar - osmotic_pressure)
    temp_factor = 1.0 + 0.03 * (temperature_c - 25.0)
    temp_factor = max(0.3, temp_factor)  # don't go below 30%
    return RO_PERMEABILITY * net_pressure * _clamp(fouling_factor, 0.0, 1.0) * temp_factor


def ro_rejection(fouling_factor: float) -> float:
    """TDS rejection ratio, degraded by fouling."""
    base = RO_REJECTION
    return base * _clamp(fouling_factor, 0.0, 1.0)


def ro_energy_per_liter(pressure_bar: float) -> float:
    """Energy to push 1 liter through RO membrane (Wh).

    E = P·V / η  where P in Pa, V in m³, η = pump efficiency.
    1 bar = 100000 Pa, 1 L = 0.001 m³.
    """
    pressure_pa = pressure_bar * 100000.0
    volume_m3 = 0.001
    energy_j = pressure_pa * volume_m3 / PUMP_EFFICIENCY
    return energy_j / 3600.0  # J → Wh


def ro_fouling_rate(hours: float, life_hours: float) -> float:
    """Fouling factor (1=clean, 0=dead) as function of runtime."""
    if life_hours <= 0.0:
        return 0.0
    ratio = hours / life_hours
    if ratio < 0.5:
        return 1.0 - 0.1 * ratio  # slow initial fouling
    else:
        return max(0.0, 1.0 - 0.1 * 0.5 - 0.8 * ((ratio - 0.5) / 0.5) ** 2)


# ── UV-C sterilization physics ─────────────────────────────────────

def uv_dose(lamp_power_w: float, lamp_count: int,
            flow_rate_l_h: float, chamber_area_cm2: float,
            lamp_degradation: float) -> float:
    """UV-C dose delivered in mJ/cm².

    dose = (power × time) / area
    time = chamber_volume / flow_rate
    Simplified: dose = (P × n × degradation) / (flow × area_factor)
    """
    if flow_rate_l_h <= 0.0 or chamber_area_cm2 <= 0.0:
        return 0.0
    total_power_mw = lamp_power_w * lamp_count * 1000.0 * _clamp(lamp_degradation, 0.0, 1.0)
    # Exposure time: assume 1L chamber, time = 1/flow (hours) → seconds
    exposure_s = 3600.0 / flow_rate_l_h
    dose = (total_power_mw * exposure_s) / chamber_area_cm2
    return dose


def uv_log_reduction(dose_mj_cm2: float) -> float:
    """Microbial log-reduction from UV-C dose.

    log_kill = dose / D10  (D10 for E. coli ≈ 7 mJ/cm²).
    Capped at 6-log (practical limit).
    """
    if dose_mj_cm2 <= 0.0:
        return 0.0
    return min(6.0, dose_mj_cm2 / UV_D10_MJ_CM2)


def uv_lamp_degradation(hours: float, life_hours: float) -> float:
    """UV lamp output degradation (1=new, 0=dead).

    Linear fade to 70% at end of rated life, then rapid drop.
    """
    if life_hours <= 0.0:
        return 0.0
    ratio = hours / life_hours
    if ratio <= 1.0:
        return 1.0 - 0.3 * ratio
    else:
        return max(0.0, 0.7 * math.exp(-(ratio - 1.0) * 3.0))


# ── Tick: process one batch ─────────────────────────────────────────

@dataclass
class PurifyResult:
    """Result of one purification tick."""
    batch: WaterBatch = field(default_factory=WaterBatch)
    ix_efficiency: float = 0.0
    ro_flux_l_m2_h: float = 0.0
    ro_rejection_ratio: float = 0.0
    uv_delivered_dose: float = 0.0
    energy_wh: float = 0.0
    ix_near_breakthrough: bool = False
    uv_lamp_weak: bool = False
    ro_membrane_fouled: bool = False
    warning: str = ""


def tick_purify(state: PurifierState,
                dt_hours: float = 1.0,
                raw_volume_liters: float = 0.0,
                raw_perchlorate_mg_l: float = 500.0,
                raw_tds_mg_l: float = 3000.0) -> PurifyResult:
    """Advance the purifier by dt_hours, optionally processing a batch.

    Parameters
    ----------
    state : PurifierState — mutable, updated in place
    dt_hours : simulation time step (hours)
    raw_volume_liters : volume of raw water to process (0 = idle tick)
    raw_perchlorate_mg_l : perchlorate concentration in raw water
    raw_tds_mg_l : total dissolved solids in raw water
    """
    result = PurifyResult()
    dt_hours = max(0.0, dt_hours)
    raw_volume_liters = max(0.0, raw_volume_liters)
    raw_perchlorate_mg_l = max(0.0, raw_perchlorate_mg_l)
    raw_tds_mg_l = max(0.0, raw_tds_mg_l)

    # Age equipment even when idle
    if dt_hours > 0.0:
        state.ro_hours += dt_hours
        state.uv_lamp_hours += dt_hours
        state.ro_fouling_factor = ro_fouling_rate(state.ro_hours, state.ro_membrane_life_hours)

    if raw_volume_liters <= 0.0:
        return result

    # ── Stage 1: Ion exchange (perchlorate removal) ──
    bv_used = ix_bed_volumes_used(raw_volume_liters, state.ix_resin_volume_l)
    state.ix_bed_volumes_processed += bv_used

    ix_eff = ix_removal_efficiency(state.ix_bed_volumes_processed, state.ix_capacity_fraction)
    result.ix_efficiency = ix_eff

    post_ix_perchlorate = raw_perchlorate_mg_l * (1.0 - ix_eff)

    # Check if near breakthrough
    effective_limit = IX_BED_VOLUMES_TO_BREAKTHROUGH * state.ix_capacity_fraction
    if effective_limit > 0 and state.ix_bed_volumes_processed / effective_limit > 0.7:
        result.ix_near_breakthrough = True

    # ── Stage 2: Reverse osmosis (TDS + residual perchlorate) ──
    raw_tds_g_l = raw_tds_mg_l / 1000.0
    flux = ro_flux(state.ro_pressure_bar, raw_tds_g_l,
                   state.ro_fouling_factor, state.water_temperature_c)
    result.ro_flux_l_m2_h = flux

    rejection = ro_rejection(state.ro_fouling_factor)
    result.ro_rejection_ratio = rejection

    # RO removes remaining perchlorate too
    post_ro_perchlorate = post_ix_perchlorate * (1.0 - rejection)
    post_ro_tds = raw_tds_mg_l * (1.0 - rejection)

    # Energy for RO pumping
    ro_energy = ro_energy_per_liter(state.ro_pressure_bar) * raw_volume_liters
    result.energy_wh += ro_energy

    # Check membrane fouling
    if state.ro_fouling_factor < 0.5:
        result.ro_membrane_fouled = True

    # ── Stage 3: UV-C sterilization ──
    lamp_deg = uv_lamp_degradation(state.uv_lamp_hours, state.uv_lamp_life_hours)
    flow_rate = raw_volume_liters / max(0.001, dt_hours)  # L/h
    chamber_area = 500.0  # cm² — fixed chamber geometry
    delivered_dose = uv_dose(UV_LAMP_POWER_W, state.uv_lamp_count,
                             flow_rate, chamber_area, lamp_deg)
    result.uv_delivered_dose = delivered_dose

    log_kill = uv_log_reduction(delivered_dose)

    # UV energy
    uv_energy_wh = UV_LAMP_POWER_W * state.uv_lamp_count * dt_hours
    result.energy_wh += uv_energy_wh

    # Check lamp health
    if lamp_deg < 0.5:
        result.uv_lamp_weak = True

    # ── Build output batch ──
    potable = (post_ro_perchlorate <= PERCHLORATE_SMAC_MG_L
               and log_kill >= 4.0
               and post_ro_tds <= 500.0)

    batch = WaterBatch(
        batch_id=state.total_batches + 1,
        volume_liters=raw_volume_liters,
        influent_perchlorate_mg_l=raw_perchlorate_mg_l,
        effluent_perchlorate_mg_l=post_ro_perchlorate,
        influent_tds_mg_l=raw_tds_mg_l,
        effluent_tds_mg_l=post_ro_tds,
        uv_dose_mj_cm2=delivered_dose,
        log_reduction=log_kill,
        potable=potable,
        energy_wh=result.energy_wh,
    )
    result.batch = batch
    result.energy_wh = batch.energy_wh

    # Update state
    state.total_batches += 1
    state.total_liters_processed += raw_volume_liters
    state.total_energy_wh += result.energy_wh
    state.batches.append(batch)

    # Warnings
    warnings = []
    if result.ix_near_breakthrough:
        warnings.append("IX resin near breakthrough — regenerate soon")
    if result.ro_membrane_fouled:
        warnings.append("RO membrane fouled — replace or clean")
    if result.uv_lamp_weak:
        warnings.append("UV lamp degraded — replace")
    if not potable:
        warnings.append(f"NOT POTABLE: ClO4={post_ro_perchlorate:.3f} mg/L, "
                        f"TDS={post_ro_tds:.0f} mg/L, log_kill={log_kill:.1f}")
    result.warning = "; ".join(warnings)

    return result


def regenerate_ix_resin(state: PurifierState) -> None:
    """Regenerate the ion-exchange resin with brine flush.

    Resets bed volumes counter, increments regen cycle count,
    and recalculates remaining capacity.
    """
    state.ix_regen_cycles += 1
    state.ix_bed_volumes_processed = 0.0
    state.ix_capacity_fraction = ix_regen_capacity_after(state.ix_regen_cycles)


# ── Factory ─────────────────────────────────────────────────────────

def create_water_purifier(scenario: str = "colony") -> PurifierState:
    """Create a PurifierState for a named scenario.

    Scenarios
    ---------
    colony : Full-scale colony water treatment (50L resin, 10m² RO)
    outpost : Small forward outpost (10L resin, 2m² RO)
    emergency : Minimal portable unit (5L resin, 1m² RO, 1 UV lamp)
    """
    configs: Dict[str, dict] = {
        "colony": dict(
            ix_resin_volume_l=50.0,
            ro_membrane_area_m2=10.0,
            ro_pressure_bar=15.0,
            uv_lamp_count=2,
            water_temperature_c=15.0,
        ),
        "outpost": dict(
            ix_resin_volume_l=10.0,
            ro_membrane_area_m2=2.0,
            ro_pressure_bar=12.0,
            uv_lamp_count=2,
            water_temperature_c=10.0,
        ),
        "emergency": dict(
            ix_resin_volume_l=5.0,
            ro_membrane_area_m2=1.0,
            ro_pressure_bar=10.0,
            uv_lamp_count=1,
            water_temperature_c=8.0,
        ),
    }
    cfg = configs.get(scenario, configs["colony"])
    return PurifierState(**cfg)
