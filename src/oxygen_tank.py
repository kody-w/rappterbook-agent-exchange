"""oxygen_tank.py — Mars Colony Cryogenic LOX Storage

Models a cryogenic liquid oxygen (LOX) tank farm for buffering O₂
production (MOXIE / electrolysis) against crew consumption and EVA
demand.  Without storage, production-consumption mismatches kill the
crew during dust storms or equipment downtime.

Physics
-------
* **Cryogenic storage**: LOX at 90.2 K (boiling point at 101 kPa).
  Mars ambient ~210 K.  Heat leak through multilayer insulation (MLI)
  drives boil-off.
* **Boil-off rate**: Q̇ = U·A·ΔT, where U = overall heat transfer
  coefficient (~0.5 W/m²·K for 30-layer MLI), A = tank surface area,
  ΔT = T_ambient - T_LOX.  Boil-off mass rate: ṁ = Q̇ / h_fg,
  where h_fg = 213 kJ/kg (latent heat of vaporization of O₂).
* **Tank pressure**: Boil-off generates gaseous O₂ (GOX) in ullage.
  If not vented or reliquefied, pressure rises.  Relief valve at
  design pressure (typically 300 kPa absolute).
* **Reliquefaction**: Cryocooler can recondense boil-off gas.
  COP ~0.05 at 90 K (Carnot limit ~0.43, real ~12% of Carnot).
  Power: P = Q̇_removed / COP.
* **Tank geometry**: Spherical minimizes surface/volume ratio.
  V = (4/3)πr³, A = 4πr².  LOX density = 1141 kg/m³.
* **Human O₂ consumption**: 0.84 kg/person/sol (NASA HRP).
  EVA: ~1.2 kg/person for 8-hour EVA (higher metabolic rate).
* **Safety margin**: Colony should maintain ≥7 sol reserve at all times.

Reference systems:
  - ISS NORS tanks: 382 kg O₂ at 41.4 MPa (high-pressure gas)
  - Saturn V S-IVB LOX tank: 87,200 kg LOX
  - Mars DRA 5.0: ~3000 kg O₂ for 500-day surface stay (6 crew)
  - This model: 1000 kg LOX capacity (~120 crew-sols for 8 crew)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ── Physical constants ──────────────────────────────────────────────

LOX_BOILING_POINT_K = 90.2      # O₂ boiling point at 101.3 kPa (K)
LOX_DENSITY_KG_M3 = 1141.0     # Liquid O₂ density (kg/m³)
GOX_DENSITY_KG_M3 = 1.429      # Gaseous O₂ at STP (kg/m³)
LATENT_HEAT_KJ_KG = 213.0      # Heat of vaporization of O₂ (kJ/kg)
MARS_AMBIENT_K = 210.0          # Average Mars surface temperature (K)
SECONDS_PER_SOL = 88775.0       # Mars sol (seconds)

# ── MLI insulation ──────────────────────────────────────────────────

MLI_U_COEFF_W_M2_K = 0.5       # Overall U for 30-layer MLI (W/m²·K)

# ── Cryocooler ──────────────────────────────────────────────────────

CRYO_COP = 0.05                 # Coefficient of performance at 90 K
CRYO_MAX_POWER_W = 500.0        # Max cryocooler input power (W)

# ── Tank design ─────────────────────────────────────────────────────

DEFAULT_CAPACITY_KG = 1000.0    # Nameplate LOX capacity (kg)
RELIEF_PRESSURE_KPA = 300.0     # Relief valve set pressure (kPa abs)
OPERATING_PRESSURE_KPA = 150.0  # Normal operating pressure (kPa abs)
MIN_PRESSURE_KPA = 101.3        # Minimum (atmospheric) pressure

# ── Crew consumption ────────────────────────────────────────────────

O2_PER_PERSON_PER_SOL_KG = 0.84   # Baseline metabolic O₂ (kg/sol)
O2_PER_EVA_KG = 1.2               # O₂ for one 8-hour EVA (kg)
RESERVE_SOLS = 7                   # Minimum safety reserve (sols)


# ── State ───────────────────────────────────────────────────────────

@dataclass
class TankState:
    """Mutable state of the LOX tank farm."""
    sol: int = 0
    lox_kg: float = 500.0                  # Current liquid O₂ mass (kg)
    capacity_kg: float = DEFAULT_CAPACITY_KG
    ullage_pressure_kpa: float = OPERATING_PRESSURE_KPA
    boiloff_total_kg: float = 0.0          # Cumulative boil-off (kg)
    vented_total_kg: float = 0.0           # Cumulative vented O₂ (kg)
    reliquefied_total_kg: float = 0.0      # Cumulative reliquefied (kg)
    consumed_total_kg: float = 0.0         # Cumulative crew consumption (kg)
    delivered_total_kg: float = 0.0        # Cumulative O₂ deliveries (kg)
    total_energy_wh: float = 0.0           # Cumulative cryocooler energy (Wh)
    relief_events: int = 0                 # Number of pressure relief vents
    cryo_enabled: bool = True              # Cryocooler on/off
    ambient_temp_k: float = MARS_AMBIENT_K # External temperature
    crew_count: int = 6                    # Number of crew

    def __post_init__(self) -> None:
        self.capacity_kg = max(1.0, self.capacity_kg)
        self.lox_kg = _clamp(self.lox_kg, 0.0, self.capacity_kg)
        self.ullage_pressure_kpa = max(MIN_PRESSURE_KPA, self.ullage_pressure_kpa)
        self.ambient_temp_k = max(100.0, self.ambient_temp_k)
        self.crew_count = max(0, self.crew_count)


@dataclass
class TickResult:
    """Result of one LOX tank tick."""
    boiloff_kg: float = 0.0              # Boil-off this tick (kg)
    heat_leak_w: float = 0.0             # Heat leak rate (W)
    reliquefied_kg: float = 0.0          # Reliquefied this tick (kg)
    cryo_power_w: float = 0.0            # Cryocooler power draw (W)
    energy_used_wh: float = 0.0          # Total energy this tick (Wh)
    consumed_kg: float = 0.0             # Crew consumption this tick (kg)
    vented_kg: float = 0.0               # Pressure-relief vented (kg)
    fill_fraction: float = 0.0           # Tank fill level (0..1)
    reserve_sols: float = 0.0            # Remaining crew-sols of O₂
    pressure_kpa: float = 0.0            # Current ullage pressure
    warning: str = ""


# ── Pure physics functions ──────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def sphere_surface_area(volume_m3: float) -> float:
    """Surface area of a sphere with given volume.

    A = 4π·(3V/4π)^(2/3) = (36π)^(1/3) · V^(2/3)
    """
    if volume_m3 <= 0:
        return 0.0
    r = (3.0 * volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    return 4.0 * math.pi * r * r


def tank_volume_m3(capacity_kg: float, density: float = LOX_DENSITY_KG_M3) -> float:
    """Internal volume of tank for given LOX mass capacity."""
    if density <= 0 or capacity_kg <= 0:
        return 0.0
    return capacity_kg / density


def heat_leak_watts(surface_area_m2: float, t_ambient_k: float,
                    t_lox_k: float = LOX_BOILING_POINT_K,
                    u_coeff: float = MLI_U_COEFF_W_M2_K) -> float:
    """Heat leak through MLI insulation: Q̇ = U·A·ΔT.

    Returns heat flow in watts.  Always non-negative.
    """
    dt = max(0.0, t_ambient_k - t_lox_k)
    return u_coeff * surface_area_m2 * dt


def boiloff_rate_kg_s(heat_leak_w: float,
                      latent_heat_kj_kg: float = LATENT_HEAT_KJ_KG) -> float:
    """Mass boil-off rate: ṁ = Q̇ / h_fg.

    Parameters
    ----------
    heat_leak_w : heat leak in watts (= J/s)
    latent_heat_kj_kg : latent heat in kJ/kg

    Returns
    -------
    Boil-off rate in kg/s.  Always non-negative.
    """
    if latent_heat_kj_kg <= 0:
        return 0.0
    return max(0.0, heat_leak_w / (latent_heat_kj_kg * 1000.0))


def boiloff_mass_kg(heat_leak_w: float, dt_s: float,
                    latent_heat_kj_kg: float = LATENT_HEAT_KJ_KG) -> float:
    """Total boil-off mass over dt_s seconds."""
    rate = boiloff_rate_kg_s(heat_leak_w, latent_heat_kj_kg)
    return rate * max(0.0, dt_s)


def reliquefaction_power_w(heat_removed_w: float,
                           cop: float = CRYO_COP) -> float:
    """Electrical power to reliquify boil-off: P = Q̇/COP.

    Returns electrical power in watts.  Always non-negative.
    """
    if cop <= 0:
        return float('inf')
    return max(0.0, heat_removed_w / cop)


def pressure_from_boiloff(current_pressure_kpa: float,
                          boiloff_kg: float, ullage_volume_m3: float,
                          temperature_k: float = LOX_BOILING_POINT_K) -> float:
    """Estimate pressure rise from boil-off gas in ullage space.

    Uses ideal gas approximation: ΔP = (n·R·T) / V
    where n = boiloff_kg / 0.032 (molar mass of O₂ = 32 g/mol)

    Returns new pressure in kPa.
    """
    if ullage_volume_m3 <= 0 or boiloff_kg <= 0:
        return max(MIN_PRESSURE_KPA, current_pressure_kpa)
    R = 8.314  # J/(mol·K)
    n_moles = boiloff_kg / 0.032
    dp_pa = (n_moles * R * temperature_k) / ullage_volume_m3
    dp_kpa = dp_pa / 1000.0
    return max(MIN_PRESSURE_KPA, current_pressure_kpa + dp_kpa)


def ullage_volume(tank_volume_m3: float, lox_kg: float,
                  lox_density: float = LOX_DENSITY_KG_M3) -> float:
    """Gas ullage space = tank volume - liquid volume."""
    if lox_density <= 0:
        return max(0.001, tank_volume_m3)
    liquid_vol = lox_kg / lox_density
    return max(0.001, tank_volume_m3 - liquid_vol)


def crew_consumption_kg(crew_count: int, eva_count: int = 0,
                        per_person: float = O2_PER_PERSON_PER_SOL_KG,
                        per_eva: float = O2_PER_EVA_KG) -> float:
    """O₂ consumed by crew in one sol."""
    return max(0, crew_count) * per_person + max(0, eva_count) * per_eva


def reserve_sols(lox_kg: float, crew_count: int,
                 per_person: float = O2_PER_PERSON_PER_SOL_KG) -> float:
    """How many sols of O₂ remain at current crew size."""
    daily = max(0, crew_count) * per_person
    if daily <= 0:
        return float('inf')
    return max(0.0, lox_kg / daily)


# ── Tick function ───────────────────────────────────────────────────

def tick_tank(state: TankState, dt_s: float = SECONDS_PER_SOL,
              delivered_kg: float = 0.0, eva_count: int = 0) -> TickResult:
    """Advance the LOX tank by dt_s seconds.

    Parameters
    ----------
    state : TankState (mutated in place)
    dt_s : time step in seconds (default = 1 sol)
    delivered_kg : fresh LOX delivered from production this tick (kg)
    eva_count : number of EVAs consuming extra O₂

    Returns
    -------
    TickResult with performance metrics.
    """
    result = TickResult()
    state.sol += 1

    # 1. Accept delivery (capped at remaining capacity)
    requested = max(0.0, delivered_kg)
    actual_delivered = min(requested, state.capacity_kg - state.lox_kg)
    state.lox_kg += actual_delivered
    state.delivered_total_kg += actual_delivered

    # 2. Tank geometry
    vol_m3 = tank_volume_m3(state.capacity_kg)
    area_m2 = sphere_surface_area(vol_m3)

    # 3. Heat leak and boil-off
    qleak = heat_leak_watts(area_m2, state.ambient_temp_k)
    result.heat_leak_w = qleak
    raw_boiloff = boiloff_mass_kg(qleak, dt_s)

    # 4. Reliquefaction (if cryocooler enabled)
    reliq_kg = 0.0
    cryo_power = 0.0
    if state.cryo_enabled and raw_boiloff > 0:
        cryo_power_needed = reliquefaction_power_w(qleak)
        cryo_power = min(cryo_power_needed, CRYO_MAX_POWER_W)
        if cryo_power_needed > 0:
            frac_reliq = cryo_power / cryo_power_needed
        else:
            frac_reliq = 1.0
        reliq_kg = raw_boiloff * _clamp(frac_reliq, 0.0, 1.0)

    net_boiloff = max(0.0, raw_boiloff - reliq_kg)
    net_boiloff = min(net_boiloff, state.lox_kg)  # can't boil more than exists
    result.boiloff_kg = net_boiloff
    result.reliquefied_kg = reliq_kg
    result.cryo_power_w = cryo_power

    state.lox_kg -= net_boiloff
    state.boiloff_total_kg += net_boiloff
    state.reliquefied_total_kg += reliq_kg

    # 5. Energy
    energy_wh = cryo_power * dt_s / 3600.0
    result.energy_used_wh = energy_wh
    state.total_energy_wh += energy_wh

    # 6. Pressure from net boil-off gas
    ull_vol = ullage_volume(vol_m3, state.lox_kg)
    state.ullage_pressure_kpa = pressure_from_boiloff(
        state.ullage_pressure_kpa, net_boiloff, ull_vol
    )

    # 7. Pressure relief
    vented = 0.0
    if state.ullage_pressure_kpa > RELIEF_PRESSURE_KPA:
        excess_pressure = state.ullage_pressure_kpa - OPERATING_PRESSURE_KPA
        R = 8.314
        n_excess = (excess_pressure * 1000.0 * ull_vol) / (R * LOX_BOILING_POINT_K)
        vented = max(0.0, n_excess * 0.032)
        state.ullage_pressure_kpa = OPERATING_PRESSURE_KPA
        state.relief_events += 1
        state.vented_total_kg += vented
    result.vented_kg = vented

    # 8. Crew consumption
    consumed = crew_consumption_kg(state.crew_count, eva_count)
    consumed = min(consumed, state.lox_kg)
    state.lox_kg -= consumed
    state.consumed_total_kg += consumed
    result.consumed_kg = consumed

    # 9. Metrics
    result.fill_fraction = state.lox_kg / state.capacity_kg if state.capacity_kg > 0 else 0.0
    result.reserve_sols = reserve_sols(state.lox_kg, state.crew_count)
    result.pressure_kpa = state.ullage_pressure_kpa

    # 10. Warnings
    if state.lox_kg <= 0:
        result.warning = "CRITICAL: O₂ tank empty"
    elif result.reserve_sols < RESERVE_SOLS:
        result.warning = f"LOW O₂: {result.reserve_sols:.1f} sols remaining (min {RESERVE_SOLS})"

    return result


# ── Factory ─────────────────────────────────────────────────────────

def create_oxygen_tank(scenario: str = "standard") -> TankState:
    """Create a TankState for a named scenario.

    Scenarios
    ---------
    standard : Main colony LOX farm (1000 kg, 6 crew)
    outpost : Small forward outpost (200 kg, 2 crew)
    emergency : Large reserve tank, cryo disabled (1500 kg, 8 crew)
    """
    configs = {
        "standard": dict(
            capacity_kg=1000.0, lox_kg=500.0, crew_count=6,
        ),
        "outpost": dict(
            capacity_kg=200.0, lox_kg=100.0, crew_count=2,
        ),
        "emergency": dict(
            capacity_kg=1500.0, lox_kg=1200.0, crew_count=8,
            cryo_enabled=False,
        ),
    }
    cfg = configs.get(scenario, configs["standard"])
    return TankState(**cfg)
