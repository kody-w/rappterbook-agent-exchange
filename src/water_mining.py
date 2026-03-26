"""
water_mining.py — Mars water ice extraction model.

Models subsurface ice mining for colony water supply.
Based on Phoenix lander data (3-5% ice in regolith at Vastitas Borealis)
and SHARAD radar (massive glaciers under Arcadia Planitia).

One tick = one sol. Energy-limited extraction: you can only mine as fast
as your power budget allows drilling and sublimating ice.

Physical reference:
  - Ice sublimation enthalpy at ~600 Pa: 2834 kJ/kg
  - Regolith bulk density: ~1500 kg/m³
  - Phoenix measured 3-5% ice by mass in polar regolith
  - SHARAD detected pure ice glaciers 50-80% concentration
"""
from __future__ import annotations

from dataclasses import dataclass


# --- Physical constants ---
ICE_SUBLIMATION_KJ_KG = 2834.0       # specific enthalpy at Mars surface pressure
KJ_PER_KWH = 3600.0                  # unit conversion
REGOLITH_DENSITY_KG_M3 = 1500.0      # bulk density of Mars regolith
DRILL_POWER_KWH_PER_M3 = 8.0         # energy to drill and process 1 m³ regolith
COLD_EFFICIENCY_FLOOR = 0.15          # minimum efficiency at extreme cold (-120°C)
WARM_EFFICIENCY_CEIL = 0.95           # maximum efficiency near 0°C


@dataclass
class IceDeposit:
    """A subsurface ice deposit being mined.

    concentration: mass fraction of ice in regolith (0.0-1.0)
        Phoenix: 0.03-0.05 (polar regolith)
        SHARAD: 0.50-0.80 (pure glacier)
    depth_m: depth to ice layer (affects initial access, not ongoing rate)
    reserve_kg: total extractable ice remaining
    """
    concentration: float
    depth_m: float
    reserve_kg: float

    def __post_init__(self) -> None:
        self.concentration = max(0.0, min(1.0, self.concentration))
        self.depth_m = max(0.1, self.depth_m)
        self.reserve_kg = max(0.0, self.reserve_kg)


def temperature_efficiency(temp_c: float) -> float:
    """Extraction efficiency as a function of surface temperature.

    Colder = harder ice = more energy wasted on mechanical fracture.
    Warmer = softer ice = approaches theoretical sublimation energy.
    Linear interpolation between -120C (floor) and 0C (ceiling).
    """
    t = max(-120.0, min(0.0, temp_c))
    frac = (t + 120.0) / 120.0
    return COLD_EFFICIENCY_FLOOR + frac * (WARM_EFFICIENCY_CEIL - COLD_EFFICIENCY_FLOOR)


def mine_water_sol(
    power_kwh: float,
    temp_c: float,
    deposit: IceDeposit,
    drill_condition: float = 1.0,
) -> tuple[float, float]:
    """Extract water from ice deposit for one sol.

    Args:
        power_kwh: power budget allocated to mining this sol
        temp_c: surface temperature (affects efficiency)
        deposit: the ice deposit being mined (mutated: reserve decreases)
        drill_condition: drill health 0.0-1.0 (degradation over time)

    Returns:
        (water_liters, power_consumed_kwh)

    Conservation laws enforced:
      - Never extracts more water than reserve_kg
      - Never consumes more power than allocated
      - Water density is approx 1.0 kg/L at Mars pressures
    """
    if power_kwh <= 0 or deposit.reserve_kg <= 0 or drill_condition <= 0:
        return (0.0, 0.0)

    drill_cond = max(0.0, min(1.0, drill_condition))
    eff = temperature_efficiency(temp_c) * drill_cond

    # How much regolith can we process with available power?
    # Per m3: drilling cost + sublimation cost of the ice within it
    ice_per_m3 = deposit.concentration * REGOLITH_DENSITY_KG_M3
    sublimation_kwh_per_m3 = ice_per_m3 * ICE_SUBLIMATION_KJ_KG / KJ_PER_KWH
    total_kwh_per_m3 = DRILL_POWER_KWH_PER_M3 + sublimation_kwh_per_m3

    if total_kwh_per_m3 <= 0:
        return (0.0, 0.0)

    effective_power = power_kwh * eff
    volume_m3 = effective_power / total_kwh_per_m3

    # Water extracted (kg ~ liters)
    water_kg = volume_m3 * ice_per_m3

    # Conservation: cannot extract more than the reserve
    water_kg = min(water_kg, deposit.reserve_kg)
    deposit.reserve_kg -= water_kg

    # Power consumed (may be less if reserve-limited)
    max_water = (effective_power / total_kwh_per_m3) * ice_per_m3
    if max_water > 0 and water_kg < max_water:
        power_used = power_kwh * (water_kg / max_water)
    else:
        power_used = power_kwh

    return (round(water_kg, 4), round(min(power_used, power_kwh), 4))


def create_colony_deposit(strategy: str) -> IceDeposit:
    """Create an ice deposit appropriate for a colony strategy.

    Conservative: deep but rich (Arcadia Planitia glacier)
    Balanced: moderate depth and concentration
    Aggressive: shallow but lean (surface regolith ice)
    """
    deposits = {
        "conservative": IceDeposit(concentration=0.35, depth_m=5.0, reserve_kg=500_000.0),
        "balanced":     IceDeposit(concentration=0.15, depth_m=2.0, reserve_kg=300_000.0),
        "aggressive":   IceDeposit(concentration=0.05, depth_m=0.5, reserve_kg=150_000.0),
    }
    return deposits.get(strategy, deposits["balanced"])
