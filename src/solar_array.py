"""
solar_array.py — Solar panel array model for Mars colonies.

Models power generation from photovoltaic arrays on Mars with:
  - Dust accumulation (reduces output, cleaned by wind or crew)
  - Panel degradation (radiation + thermal cycling wear over time)
  - Battery storage (charge/discharge with round-trip efficiency)
  - Temperature derating (GaAs cells lose efficiency in extreme cold)

Physical references:
  - Mars Exploration Rover: dust reduced power 0.28%/sol average
  - InSight lander: 4.6 kWh/sol at start -> 0.5 kWh/sol after 1400 sols
  - GaAs triple-junction cells: ~30% efficiency at AM0, temp coeff -0.2%/C
  - Mars dust devil cleaning events: restored up to 10% power on MER

One tick = one sol. Energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# --- Physical constants ---
PANEL_EFFICIENCY_BASE = 0.30          # GaAs triple-junction at 25C
TEMP_COEFF_PER_C = 0.002             # efficiency drop per C below 25C (cold stress)
TEMP_REF_C = 25.0                     # reference temperature for panel rating
MIN_TEMP_DERATING = 0.60              # floor: panels still work at -120C
DUST_ACCUMULATION_PER_SOL = 0.0028    # fraction of max dust per sol (MER avg)
DUST_STORM_ACCUMULATION = 0.015       # during active dust storm
WIND_CLEANING_CHANCE = 0.02           # chance per sol of a cleaning gust
WIND_CLEANING_AMOUNT = 0.10           # fraction of dust removed by wind
CREW_CLEANING_DUST_REMOVED = 0.90     # manual cleaning removes 90% of dust
DEGRADATION_RATE_PER_SOL = 0.000035   # ~2.3% per Mars year from radiation
BATTERY_ROUND_TRIP_EFF = 0.90         # Li-ion round-trip efficiency
BATTERY_SELF_DISCHARGE_PER_SOL = 0.001  # 0.1%/sol self-discharge
SOL_HOURS = 24.66                     # Mars sol in hours


@dataclass
class SolarArray:
    """A colony's solar panel installation.

    area_m2: total panel area in square meters
    dust_fraction: fraction of panels covered by dust [0, 1]
    degradation: cumulative degradation from radiation/thermal [0, 1]
    """
    area_m2: float
    dust_fraction: float = 0.0
    degradation: float = 0.0

    def __post_init__(self) -> None:
        """Clamp all fields to valid physical ranges."""
        self.area_m2 = max(0.0, self.area_m2)
        self.dust_fraction = max(0.0, min(1.0, self.dust_fraction))
        self.degradation = max(0.0, min(1.0, self.degradation))

    def effective_area(self) -> float:
        """Panel area after dust and degradation losses."""
        return self.area_m2 * (1.0 - self.dust_fraction) * (1.0 - self.degradation)


@dataclass
class Battery:
    """Colony battery bank for power storage.

    capacity_kwh: maximum charge in kWh
    charge_kwh: current charge level in kWh
    """
    capacity_kwh: float
    charge_kwh: float = 0.0

    def __post_init__(self) -> None:
        """Clamp capacity and charge to valid ranges."""
        self.capacity_kwh = max(0.0, self.capacity_kwh)
        self.charge_kwh = max(0.0, min(self.charge_kwh, self.capacity_kwh))

    def headroom(self) -> float:
        """Available capacity for charging (kWh)."""
        return max(0.0, self.capacity_kwh - self.charge_kwh)

    def charge(self, kwh: float) -> float:
        """Store energy. Returns actual kWh stored (after round-trip loss)."""
        if kwh <= 0:
            return 0.0
        storable = kwh * BATTERY_ROUND_TRIP_EFF
        actual = min(storable, self.headroom())
        self.charge_kwh += actual
        return actual

    def discharge(self, kwh_needed: float) -> float:
        """Draw energy. Returns actual kWh delivered."""
        if kwh_needed <= 0:
            return 0.0
        delivered = min(kwh_needed, self.charge_kwh)
        self.charge_kwh -= delivered
        return delivered

    def apply_self_discharge(self) -> float:
        """Apply daily self-discharge. Returns kWh lost."""
        lost = self.charge_kwh * BATTERY_SELF_DISCHARGE_PER_SOL
        self.charge_kwh -= lost
        return lost


def temperature_derating(temp_c: float) -> float:
    """Panel efficiency multiplier due to temperature.

    GaAs cells lose efficiency in extreme cold due to increased series
    resistance and reduced carrier mobility. Linear model with floor.
    """
    delta = temp_c - TEMP_REF_C
    derating = 1.0 + TEMP_COEFF_PER_C * delta
    return max(MIN_TEMP_DERATING, min(1.0, derating))


def solar_power_sol(
    array: SolarArray,
    solar_flux_wm2: float,
    temp_c: float,
) -> float:
    """Calculate solar power generated in one sol (kWh).

    area_m2 * flux_W/m2 = W total. * efficiency = W electrical.
    * sol_hours = Wh. / 1000 = kWh.
    """
    if solar_flux_wm2 <= 0:
        return 0.0
    eff_area = array.effective_area()
    temp_mult = temperature_derating(temp_c)
    watts = eff_area * solar_flux_wm2 * PANEL_EFFICIENCY_BASE * temp_mult
    kwh = watts * SOL_HOURS / 1000.0
    return round(kwh, 4)


def accumulate_dust(
    array: SolarArray,
    in_storm: bool,
    rng_roll: float,
) -> dict:
    """Apply one sol of dust accumulation and possible wind cleaning.

    Args:
        array: the solar array (mutated in place)
        in_storm: whether a dust storm is active
        rng_roll: random float [0, 1) for wind cleaning check

    Returns:
        dict with dust_before, dust_after, wind_cleaned (bool)
    """
    dust_before = array.dust_fraction

    rate = DUST_STORM_ACCUMULATION if in_storm else DUST_ACCUMULATION_PER_SOL
    array.dust_fraction = min(1.0, array.dust_fraction + rate)

    wind_cleaned = False
    if not in_storm and rng_roll < WIND_CLEANING_CHANCE:
        cleaned = array.dust_fraction * WIND_CLEANING_AMOUNT
        array.dust_fraction = max(0.0, array.dust_fraction - cleaned)
        wind_cleaned = True

    return {
        "dust_before": round(dust_before, 6),
        "dust_after": round(array.dust_fraction, 6),
        "wind_cleaned": wind_cleaned,
    }


def crew_clean_panels(array: SolarArray) -> float:
    """Crew manually cleans solar panels. Returns dust removed."""
    removed = array.dust_fraction * CREW_CLEANING_DUST_REMOVED
    array.dust_fraction -= removed
    array.dust_fraction = max(0.0, array.dust_fraction)
    return round(removed, 6)


def degrade_panels(array: SolarArray, radiation_msv: float) -> float:
    """Apply radiation and thermal degradation for one sol.

    Higher radiation accelerates degradation. Returns degradation delta.
    """
    rad_factor = max(1.0, radiation_msv / 0.67)
    delta = DEGRADATION_RATE_PER_SOL * rad_factor
    array.degradation = min(1.0, array.degradation + delta)
    return round(delta, 8)


def tick_power_system(
    array: SolarArray,
    battery: Battery,
    solar_flux_wm2: float,
    temp_c: float,
    radiation_msv: float,
    demand_kwh: float,
    in_storm: bool,
    rng_roll: float,
    nuclear_kwh: float = 0.0,
) -> dict:
    """Advance power system by one sol.

    1. Degrade panels (radiation)
    2. Accumulate/clean dust
    3. Generate solar power
    4. Add nuclear baseline
    5. Battery self-discharge
    6. Meet demand: generation first, then battery
    7. Surplus to battery

    Returns snapshot dict with all power metrics.
    """
    deg_delta = degrade_panels(array, radiation_msv)
    dust_info = accumulate_dust(array, in_storm, rng_roll)
    solar_kwh = solar_power_sol(array, solar_flux_wm2, temp_c)
    total_gen = solar_kwh + nuclear_kwh
    bat_lost = battery.apply_self_discharge()

    deficit = max(0.0, demand_kwh - total_gen)
    surplus = max(0.0, total_gen - demand_kwh)
    bat_drawn = 0.0
    if deficit > 0:
        bat_drawn = battery.discharge(deficit)

    bat_stored = 0.0
    if surplus > 0:
        bat_stored = battery.charge(surplus)

    delivered = min(demand_kwh, total_gen + bat_drawn)

    return {
        "solar_kwh": round(solar_kwh, 4),
        "nuclear_kwh": round(nuclear_kwh, 4),
        "total_generated_kwh": round(total_gen, 4),
        "demand_kwh": round(demand_kwh, 4),
        "delivered_kwh": round(delivered, 4),
        "deficit_kwh": round(max(0.0, demand_kwh - delivered), 4),
        "battery_drawn_kwh": round(bat_drawn, 4),
        "battery_stored_kwh": round(bat_stored, 4),
        "battery_charge_kwh": round(battery.charge_kwh, 4),
        "battery_self_discharge_kwh": round(bat_lost, 4),
        "dust_fraction": round(array.dust_fraction, 6),
        "degradation": round(array.degradation, 6),
        "degradation_delta": round(deg_delta, 8),
        "wind_cleaned": dust_info["wind_cleaned"],
        "panel_effective_area_m2": round(array.effective_area(), 4),
    }
