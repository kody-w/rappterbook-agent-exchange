"""propellant_depot.py -- Mars Colony Cryogenic Propellant Storage & Transfer

The colony makes fuel.  This module stores it.  Without a depot, the
Sabatier reactor's output boils away between production runs.  Without
correct oxidiser-to-fuel ratio management, engines don't ignite.
Without boiloff tracking, the return-mission propellant budget is a lie.

Physics
-------
* **Liquid methane (LCH4)**: Boiling point 111.7 K, density 422.6 kg/m3,
  latent heat 510 kJ/kg.  Warmer than LOX (90.2 K), so easier to store
  but still cryogenic on Mars (ambient ~210 K).
* **Liquid oxygen (LOX)**: Boiling point 90.2 K, density 1141 kg/m3,
  latent heat 213 kJ/kg.
* **Boiloff model**: Q = U * A * dT.  MLI insulation U ~ 0.1 W/m2/K
  (30-layer, Mars atm degrades from ~0.02 in hard vacuum).
  Boiloff rate = Q / latent_heat.
* **Tank geometry**: Spherical (minimises surface/volume).
  V = (4/3)*pi*r^3, A = 4*pi*r^2.
* **Mixture ratio**: Methalox engines burn at O/F ~ 3.6 by mass.
  Depot must maintain LOX:CH4 ratio for vehicle loading.
* **Reliquefaction**: Cryocooler COP ~ 0.08 at 112 K (CH4),
  ~ 0.05 at 90 K (LOX).  Power = Q_removed / COP.
* **Vehicle loading**: Starship-class return needs ~240 t propellant.
  Colony accumulates over ~500 sols of Sabatier production.

Reference: SpaceX Starship propellant mass 240 t (78% LOX, 22% CH4),
  Zubrin DRA 5.0, Mars ISRU studies (Hintze & Meier 2018).

One tick = one sol.  Mass in kg, energy in kWh, temperature in K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# -- Physical constants -------------------------------------------------------

# Liquid methane
LCH4_BOILING_K = 111.7          # boiling point at 101 kPa (K)
LCH4_DENSITY_KG_M3 = 422.6     # liquid density (kg/m3)
LCH4_LATENT_KJ_KG = 510.0      # latent heat of vaporization (kJ/kg)

# Liquid oxygen
LOX_BOILING_K = 90.2            # boiling point at 101 kPa (K)
LOX_DENSITY_KG_M3 = 1141.0     # liquid density (kg/m3)
LOX_LATENT_KJ_KG = 213.0       # latent heat of vaporization (kJ/kg)

# Mars environment
MARS_AMBIENT_K = 210.0          # average surface temperature (K)
SECONDS_PER_SOL = 88775.0       # one Mars sol (s)

# Insulation (multilayer insulation, 30-layer)
MLI_U_W_M2_K = 0.1             # overall heat transfer coeff (W/m2/K)

# Cryocooler coefficients of performance
CRYO_COP_CH4 = 0.08            # COP at ~112 K
CRYO_COP_LOX = 0.05            # COP at ~90 K

# Engine mixture ratio
MIXTURE_RATIO_OF = 3.6          # O/F mass ratio (LOX/CH4)

# Starship-class mission
STARSHIP_PROPELLANT_KG = 240_000.0    # total propellant for Earth return
STARSHIP_LOX_FRACTION = 0.78          # mass fraction LOX
STARSHIP_CH4_FRACTION = 0.22          # mass fraction CH4

# Safety
MIN_ULLAGE_FRACTION = 0.05     # minimum 5% gas ullage in tanks
MAX_GAS_FRACTION = 0.10        # vent above 10% of capacity in gas


# -- Geometry helpers ---------------------------------------------------------

def sphere_volume_m3(radius_m: float) -> float:
    """Volume of a sphere (m3)."""
    return (4.0 / 3.0) * math.pi * radius_m ** 3


def sphere_surface_m2(radius_m: float) -> float:
    """Surface area of a sphere (m2)."""
    return 4.0 * math.pi * radius_m ** 2


def radius_for_volume(volume_m3: float) -> float:
    """Radius of a sphere with given volume (m)."""
    if volume_m3 <= 0:
        return 0.0
    return (3.0 * volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def tank_surface_for_capacity(capacity_kg: float,
                               density_kg_m3: float) -> float:
    """Surface area of a spherical tank sized for given liquid capacity."""
    if capacity_kg <= 0 or density_kg_m3 <= 0:
        return 0.0
    vol = capacity_kg / density_kg_m3 / (1.0 - MIN_ULLAGE_FRACTION)
    r = radius_for_volume(vol)
    return sphere_surface_m2(r)


# -- Boiloff physics ----------------------------------------------------------

def heat_leak_w(surface_m2: float, u_coeff: float, dt_k: float) -> float:
    """Heat leak through insulation (W)."""
    return max(0.0, surface_m2 * u_coeff * max(0.0, dt_k))


def boiloff_kg_per_sol(heat_leak_watts: float,
                        latent_kj_kg: float) -> float:
    """Mass of cryogen boiled off per sol (kg)."""
    if latent_kj_kg <= 0 or heat_leak_watts <= 0:
        return 0.0
    heat_per_sol_kj = heat_leak_watts * SECONDS_PER_SOL / 1000.0
    return heat_per_sol_kj / latent_kj_kg


def reliquefaction_kwh(boiloff_kg: float, latent_kj_kg: float,
                        cop: float) -> float:
    """Energy to reliquefy boiloff gas (kWh)."""
    if boiloff_kg <= 0 or cop <= 0 or latent_kj_kg <= 0:
        return 0.0
    heat_to_remove_kj = boiloff_kg * latent_kj_kg
    work_kj = heat_to_remove_kj / cop
    return work_kj / 3600.0  # kJ -> kWh


# -- Tank state ---------------------------------------------------------------

@dataclass
class TankState:
    """State of a single cryogenic tank."""
    label: str                      # "ch4" or "lox"
    capacity_kg: float              # max liquid mass
    liquid_kg: float = 0.0         # current liquid mass
    boiloff_gas_kg: float = 0.0    # accumulated boiloff gas
    surface_m2: float = 0.0        # tank surface area (computed)
    boiling_k: float = 0.0         # boiling point of contents
    latent_kj_kg: float = 0.0      # latent heat
    density_kg_m3: float = 0.0     # liquid density
    cryo_cop: float = 0.0          # cryocooler COP

    def fill_fraction(self) -> float:
        """Current fill level as fraction [0, 1]."""
        if self.capacity_kg <= 0:
            return 0.0
        return min(1.0, self.liquid_kg / self.capacity_kg)


@dataclass
class DepotState:
    """State of the full propellant depot."""
    ch4_tank: TankState
    lox_tank: TankState
    sols_running: int = 0

    # Cumulative tracking
    total_ch4_boiloff_kg: float = 0.0
    total_lox_boiloff_kg: float = 0.0
    total_ch4_reliq_kwh: float = 0.0
    total_lox_reliq_kwh: float = 0.0
    total_ch4_received_kg: float = 0.0
    total_lox_received_kg: float = 0.0
    total_ch4_dispensed_kg: float = 0.0
    total_lox_dispensed_kg: float = 0.0

    alert: str = "nominal"


@dataclass
class DepotTickResult:
    """Result of one sol of depot operation."""
    ch4_boiloff_kg: float = 0.0
    lox_boiloff_kg: float = 0.0
    ch4_reliq_kwh: float = 0.0
    lox_reliq_kwh: float = 0.0
    ch4_received_kg: float = 0.0
    lox_received_kg: float = 0.0
    ch4_vented_kg: float = 0.0
    lox_vented_kg: float = 0.0
    total_energy_kwh: float = 0.0

    ch4_fill_fraction: float = 0.0
    lox_fill_fraction: float = 0.0
    mixture_ratio: float = 0.0
    mission_readiness: float = 0.0
    alert: str = "nominal"


# -- Depot operations ---------------------------------------------------------

def compute_mixture_ratio(lox_kg: float, ch4_kg: float) -> float:
    """Current O/F mixture ratio (LOX mass / CH4 mass)."""
    if ch4_kg <= 0:
        return float("inf") if lox_kg > 0 else 0.0
    return lox_kg / ch4_kg


def mission_readiness_fraction(lox_kg: float, ch4_kg: float,
                                target_kg: float = STARSHIP_PROPELLANT_KG
                                ) -> float:
    """Fraction of return-mission propellant accumulated [0, 1].

    Both LOX and CH4 must meet their targets independently.
    """
    if target_kg <= 0:
        return 1.0
    lox_target = target_kg * STARSHIP_LOX_FRACTION
    ch4_target = target_kg * STARSHIP_CH4_FRACTION
    lox_frac = min(1.0, lox_kg / lox_target) if lox_target > 0 else 1.0
    ch4_frac = min(1.0, ch4_kg / ch4_target) if ch4_target > 0 else 1.0
    return min(lox_frac, ch4_frac)


def receive_propellant(tank: TankState, amount_kg: float) -> float:
    """Add propellant to a tank.  Returns amount actually stored."""
    if amount_kg <= 0:
        return 0.0
    space = max(0.0, tank.capacity_kg - tank.liquid_kg)
    stored = min(amount_kg, space)
    tank.liquid_kg += stored
    return stored


def dispense_propellant(tank: TankState, amount_kg: float) -> float:
    """Remove propellant from a tank.  Returns amount dispensed."""
    if amount_kg <= 0:
        return 0.0
    available = max(0.0, tank.liquid_kg)
    dispensed = min(amount_kg, available)
    tank.liquid_kg -= dispensed
    return dispensed


def dispense_matched_pair(depot: DepotState,
                           total_kg: float) -> tuple:
    """Dispense LOX+CH4 at correct mixture ratio for engine use.

    Returns (lox_dispensed, ch4_dispensed).
    """
    if total_kg <= 0:
        return 0.0, 0.0
    ch4_need = total_kg / (1.0 + MIXTURE_RATIO_OF)
    lox_need = total_kg - ch4_need
    ch4_got = dispense_propellant(depot.ch4_tank, ch4_need)
    lox_got = dispense_propellant(depot.lox_tank, lox_need)
    return lox_got, ch4_got


def tick_tank(tank: TankState, ambient_k: float,
               power_for_reliq_kwh: float) -> dict:
    """Advance one tank by one sol.

    Returns dict with boiloff_kg, reliquefied_kg, vented_kg, energy_kwh.
    """
    dt = max(0.0, ambient_k - tank.boiling_k)
    hl_watts = heat_leak_w(tank.surface_m2, MLI_U_W_M2_K, dt)
    raw_boiloff = boiloff_kg_per_sol(hl_watts, tank.latent_kj_kg)
    actual_boiloff = min(raw_boiloff, max(0.0, tank.liquid_kg))

    tank.liquid_kg -= actual_boiloff
    tank.boiloff_gas_kg += actual_boiloff

    # Reliquefaction: use available power to recondense gas
    reliq_energy_per_kg = reliquefaction_kwh(
        1.0, tank.latent_kj_kg, tank.cryo_cop)
    if reliq_energy_per_kg > 0 and power_for_reliq_kwh > 0:
        max_reliq_kg = power_for_reliq_kwh / reliq_energy_per_kg
        reliq_kg = min(max_reliq_kg, tank.boiloff_gas_kg)
        space = max(0.0, tank.capacity_kg - tank.liquid_kg)
        reliq_kg = min(reliq_kg, space)
        tank.liquid_kg += reliq_kg
        tank.boiloff_gas_kg -= reliq_kg
        energy_used = reliq_kg * reliq_energy_per_kg
    else:
        reliq_kg = 0.0
        energy_used = 0.0

    # Vent excess gas
    max_gas_kg = tank.capacity_kg * MAX_GAS_FRACTION
    vented = max(0.0, tank.boiloff_gas_kg - max_gas_kg)
    tank.boiloff_gas_kg -= vented

    return {
        "boiloff_kg": actual_boiloff,
        "reliquefied_kg": reliq_kg,
        "vented_kg": vented,
        "energy_kwh": energy_used,
    }


# -- Main tick ----------------------------------------------------------------

def tick_depot(depot: DepotState,
                ch4_input_kg: float = 0.0,
                lox_input_kg: float = 0.0,
                dispense_kg: float = 0.0,
                power_budget_kwh: float = 20.0,
                ambient_k: float = MARS_AMBIENT_K
                ) -> tuple:
    """Advance the propellant depot by one sol.

    Returns (DepotState, DepotTickResult).
    """
    result = DepotTickResult()

    # Receive propellant
    ch4_stored = receive_propellant(depot.ch4_tank, ch4_input_kg)
    lox_stored = receive_propellant(depot.lox_tank, lox_input_kg)
    depot.total_ch4_received_kg += ch4_stored
    depot.total_lox_received_kg += lox_stored
    result.ch4_received_kg = ch4_stored
    result.lox_received_kg = lox_stored

    # Dispense if requested
    if dispense_kg > 0:
        lox_out, ch4_out = dispense_matched_pair(depot, dispense_kg)
        depot.total_ch4_dispensed_kg += ch4_out
        depot.total_lox_dispensed_kg += lox_out

    # Split power between tanks proportional to heat leak
    ch4_dt = max(0.0, ambient_k - depot.ch4_tank.boiling_k)
    lox_dt = max(0.0, ambient_k - depot.lox_tank.boiling_k)
    total_dt = ch4_dt + lox_dt
    if total_dt > 0:
        ch4_power = power_budget_kwh * (ch4_dt / total_dt)
        lox_power = power_budget_kwh * (lox_dt / total_dt)
    else:
        ch4_power = power_budget_kwh * 0.5
        lox_power = power_budget_kwh * 0.5

    # Tick each tank
    ch4_res = tick_tank(depot.ch4_tank, ambient_k, ch4_power)
    lox_res = tick_tank(depot.lox_tank, ambient_k, lox_power)

    # Update cumulative state
    depot.total_ch4_boiloff_kg += ch4_res["boiloff_kg"]
    depot.total_lox_boiloff_kg += lox_res["boiloff_kg"]
    depot.total_ch4_reliq_kwh += ch4_res["energy_kwh"]
    depot.total_lox_reliq_kwh += lox_res["energy_kwh"]
    depot.sols_running += 1

    # Populate result
    result.ch4_boiloff_kg = ch4_res["boiloff_kg"]
    result.lox_boiloff_kg = lox_res["boiloff_kg"]
    result.ch4_reliq_kwh = ch4_res["energy_kwh"]
    result.lox_reliq_kwh = lox_res["energy_kwh"]
    result.ch4_vented_kg = ch4_res["vented_kg"]
    result.lox_vented_kg = lox_res["vented_kg"]
    result.total_energy_kwh = ch4_res["energy_kwh"] + lox_res["energy_kwh"]
    result.ch4_fill_fraction = depot.ch4_tank.fill_fraction()
    result.lox_fill_fraction = depot.lox_tank.fill_fraction()
    result.mixture_ratio = compute_mixture_ratio(
        depot.lox_tank.liquid_kg, depot.ch4_tank.liquid_kg)
    result.mission_readiness = mission_readiness_fraction(
        depot.lox_tank.liquid_kg, depot.ch4_tank.liquid_kg)

    # Alert assessment
    if (depot.ch4_tank.fill_fraction() > 0.95
            or depot.lox_tank.fill_fraction() > 0.95):
        depot.alert = "warning"
    elif depot.ch4_tank.liquid_kg < 100 and depot.lox_tank.liquid_kg < 100:
        depot.alert = "low"
    else:
        depot.alert = "nominal"
    result.alert = depot.alert

    return depot, result


# -- Factory ------------------------------------------------------------------

def create_depot(ch4_capacity_kg: float = 60_000.0,
                  lox_capacity_kg: float = 200_000.0,
                  ch4_initial_kg: float = 0.0,
                  lox_initial_kg: float = 0.0) -> DepotState:
    """Create a propellant depot with two cryogenic tanks.

    Default capacities sized for Starship return with margin:
      CH4: 60,000 kg (Starship needs ~52,800 kg)
      LOX: 200,000 kg (Starship needs ~187,200 kg)
    """
    ch4_surface = tank_surface_for_capacity(
        ch4_capacity_kg, LCH4_DENSITY_KG_M3)
    lox_surface = tank_surface_for_capacity(
        lox_capacity_kg, LOX_DENSITY_KG_M3)

    ch4_tank = TankState(
        label="ch4",
        capacity_kg=max(0.0, ch4_capacity_kg),
        liquid_kg=max(0.0, min(ch4_initial_kg, ch4_capacity_kg)),
        surface_m2=ch4_surface,
        boiling_k=LCH4_BOILING_K,
        latent_kj_kg=LCH4_LATENT_KJ_KG,
        density_kg_m3=LCH4_DENSITY_KG_M3,
        cryo_cop=CRYO_COP_CH4,
    )
    lox_tank = TankState(
        label="lox",
        capacity_kg=max(0.0, lox_capacity_kg),
        liquid_kg=max(0.0, min(lox_initial_kg, lox_capacity_kg)),
        surface_m2=lox_surface,
        boiling_k=LOX_BOILING_K,
        latent_kj_kg=LOX_LATENT_KJ_KG,
        density_kg_m3=LOX_DENSITY_KG_M3,
        cryo_cop=CRYO_COP_LOX,
    )
    return DepotState(ch4_tank=ch4_tank, lox_tank=lox_tank)


def depot_power_estimate_kwh(depot: DepotState,
                              ambient_k: float = MARS_AMBIENT_K) -> float:
    """Estimate daily power need to fully reliquefy all boiloff."""
    total = 0.0
    for tank in [depot.ch4_tank, depot.lox_tank]:
        dt = max(0.0, ambient_k - tank.boiling_k)
        hl = heat_leak_w(tank.surface_m2, MLI_U_W_M2_K, dt)
        bo = boiloff_kg_per_sol(hl, tank.latent_kj_kg)
        total += reliquefaction_kwh(bo, tank.latent_kj_kg, tank.cryo_cop)
    return total
