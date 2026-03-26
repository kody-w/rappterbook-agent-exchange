"""condensation_trap.py -- Mars Atmospheric Water Vapor Condensation Harvester.

Mars atmosphere contains trace water vapor: ~210 ppm on average, rising
to ~300 ppm near the poles in summer and falling to ~50 ppm in winter.
At ~610 Pa surface pressure, the frost point is around 195-200 K.
A condensation trap cools ambient Martian air below the frost point
using Peltier thermoelectric coolers (TECs), depositing water ice on
a cold plate.  Periodically the ice is melted and collected.

Simplest possible ISRU for water -- no drilling, no subsurface access.

Physics: Clausius-Clapeyron saturation pressure, forced-convection mass
transfer, Peltier COP (fraction of Carnot), dust degradation.

Conservation laws: mass in = mass out, COP <= Carnot, yield >= 0,
dust in [0,1], energy >= 0.

One tick = one sol.  Mass in grams, power in watts, temperature in K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# -- Physical constants -------------------------------------------------------

WATER_SUBLIMATION_ENTHALPY_J_KG = 2.83e6
WATER_FUSION_ENTHALPY_J_KG = 3.34e5
WATER_SPECIFIC_GAS_CONSTANT_J_KGK = 461.5
WATER_SPECIFIC_HEAT_J_KGK = 4186.0
ICE_SPECIFIC_HEAT_J_KGK = 2090.0

T_REF_K = 273.16
P_REF_PA = 611.657

MARS_SURFACE_PRESSURE_PA = 610.0
MARS_SURFACE_TEMP_K = 210.0
MARS_H2O_MIXING_RATIO_DEFAULT = 210e-6
MARS_H2O_MIXING_RATIO_MIN = 50e-6
MARS_H2O_MIXING_RATIO_MAX = 300e-6

TEC_EFFICIENCY_FRACTION = 0.35
TEC_MIN_TEMP_DELTA_K = 5.0
TEC_MAX_DELTA_T_K = 80.0
TEC_IDLE_POWER_W = 2.0

DEFAULT_PLATE_AREA_M2 = 2.0
DEFAULT_FAN_POWER_W = 5.0
DEFAULT_WIND_SPEED_M_S = 3.0
MASS_TRANSFER_COEFF_BASE = 0.002

DUST_ACCUMULATION_PER_SOL = 0.003
DUST_CLEANING_POWER_W = 10.0
DUST_CLEANING_EFFICIENCY = 0.90

HOURS_PER_SOL = 24.66
SECONDS_PER_SOL = HOURS_PER_SOL * 3600.0
COLLECTION_DUTY_CYCLE = 0.50
EFFECTIVE_COLLECTION_SECONDS = SECONDS_PER_SOL * COLLECTION_DUTY_CYCLE


# -- Pure physics functions ---------------------------------------------------

def saturation_pressure_ice(temperature_k: float) -> float:
    """Saturation vapor pressure of water over ice via Clausius-Clapeyron."""
    if temperature_k <= 0.0:
        return 0.0
    if temperature_k >= T_REF_K:
        return P_REF_PA
    exponent = -(WATER_SUBLIMATION_ENTHALPY_J_KG / WATER_SPECIFIC_GAS_CONSTANT_J_KGK) * (
        1.0 / temperature_k - 1.0 / T_REF_K
    )
    return P_REF_PA * math.exp(exponent)


def frost_point(partial_pressure_pa: float) -> float:
    """Frost point temperature for a given water vapor partial pressure."""
    if partial_pressure_pa <= 0.0:
        return 0.0
    if partial_pressure_pa >= P_REF_PA:
        return T_REF_K
    ln_ratio = math.log(partial_pressure_pa / P_REF_PA)
    inv_t = (1.0 / T_REF_K) - (
        WATER_SPECIFIC_GAS_CONSTANT_J_KGK / WATER_SUBLIMATION_ENTHALPY_J_KG
    ) * ln_ratio
    if inv_t <= 0.0:
        return T_REF_K
    return 1.0 / inv_t


def water_partial_pressure(atm_pressure_pa: float, mixing_ratio: float) -> float:
    """Partial pressure of water vapor in the Martian atmosphere."""
    return max(0.0, atm_pressure_pa * mixing_ratio)


def carnot_cop(t_cold_k: float, t_hot_k: float) -> float:
    """Ideal Carnot COP for a refrigeration cycle."""
    delta = t_hot_k - t_cold_k
    if delta <= 0.0:
        return float("inf")
    return t_cold_k / delta


def tec_cop(t_cold_k: float, t_hot_k: float) -> float:
    """Actual TEC COP = fraction * Carnot COP."""
    c_cop = carnot_cop(t_cold_k, t_hot_k)
    if math.isinf(c_cop):
        return c_cop
    return max(0.01, TEC_EFFICIENCY_FRACTION * c_cop)


def condensation_rate_kg_s(
    plate_area_m2: float,
    wind_speed_m_s: float,
    p_h2o_pa: float,
    p_sat_plate_pa: float,
    temperature_k: float,
    dust_fraction: float = 0.0,
) -> float:
    """Mass flux of water depositing on the cold plate (kg/s)."""
    if p_h2o_pa <= p_sat_plate_pa:
        return 0.0
    effective_area = plate_area_m2 * (1.0 - dust_fraction)
    if effective_area <= 0.0:
        return 0.0
    wind_factor = wind_speed_m_s / DEFAULT_WIND_SPEED_M_S
    h_m = MASS_TRANSFER_COEFF_BASE * max(0.1, wind_factor)
    pressure_diff = p_h2o_pa - p_sat_plate_pa
    rate = effective_area * h_m * pressure_diff / (
        WATER_SPECIFIC_GAS_CONSTANT_J_KGK * temperature_k
    )
    return max(0.0, rate)


def cooling_power_w(cond_rate_kg_s: float) -> float:
    """Heat load from condensation: Q = dm/dt * L_sublimation."""
    return max(0.0, cond_rate_kg_s * WATER_SUBLIMATION_ENTHALPY_J_KG)


def tec_input_power_w(cooling_load_w: float, cop: float) -> float:
    """Electrical power required by the TEC: P_in = Q_c / COP."""
    if cop <= 0.0:
        return float("inf")
    return max(0.0, cooling_load_w / cop)


def melt_energy_j(ice_mass_kg: float, ice_temp_k: float) -> float:
    """Energy to melt ice and warm to 275 K."""
    if ice_mass_kg <= 0.0:
        return 0.0
    target_k = 275.0
    warm_ice = ice_mass_kg * ICE_SPECIFIC_HEAT_J_KGK * max(0.0, T_REF_K - ice_temp_k)
    melt = ice_mass_kg * WATER_FUSION_ENTHALPY_J_KG
    warm_water = ice_mass_kg * WATER_SPECIFIC_HEAT_J_KGK * (target_k - T_REF_K)
    return warm_ice + melt + warm_water


def daily_yield_grams(
    plate_area_m2: float,
    wind_speed_m_s: float,
    atm_pressure_pa: float,
    ambient_temp_k: float,
    plate_temp_k: float,
    mixing_ratio: float,
    dust_fraction: float = 0.0,
) -> float:
    """Estimated water yield per sol in grams."""
    p_h2o = water_partial_pressure(atm_pressure_pa, mixing_ratio)
    p_sat = saturation_pressure_ice(plate_temp_k)
    rate = condensation_rate_kg_s(
        plate_area_m2, wind_speed_m_s, p_h2o, p_sat, ambient_temp_k, dust_fraction
    )
    return rate * EFFECTIVE_COLLECTION_SECONDS * 1000.0


# -- State dataclass ----------------------------------------------------------

@dataclass
class CondensationTrap:
    """State of a Mars atmospheric water condensation harvester."""

    plate_area_m2: float = DEFAULT_PLATE_AREA_M2
    fan_power_w: float = DEFAULT_FAN_POWER_W
    ambient_temp_k: float = MARS_SURFACE_TEMP_K
    atm_pressure_pa: float = MARS_SURFACE_PRESSURE_PA
    mixing_ratio: float = MARS_H2O_MIXING_RATIO_DEFAULT
    wind_speed_m_s: float = DEFAULT_WIND_SPEED_M_S
    plate_temp_k: float = 0.0
    dust_fraction: float = 0.0
    ice_collected_g: float = 0.0
    water_extracted_g: float = 0.0
    total_water_g: float = 0.0
    total_energy_wh: float = 0.0
    sol: int = 0
    peak_yield_g: float = 0.0
    total_sols_operated: int = 0

    def __post_init__(self) -> None:
        """Initialize plate temperature and clamp values."""
        if self.plate_temp_k <= 0.0:
            self.plate_temp_k = max(80.0, self.ambient_temp_k - 30.0)
        self.dust_fraction = max(0.0, min(1.0, self.dust_fraction))
        self.ice_collected_g = max(0.0, self.ice_collected_g)
        self.water_extracted_g = max(0.0, self.water_extracted_g)
        self.total_water_g = max(0.0, self.total_water_g)
        self.total_energy_wh = max(0.0, self.total_energy_wh)
        self.plate_area_m2 = max(0.01, self.plate_area_m2)

    def to_dict(self) -> dict:
        """Serialize state to a dictionary."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict) -> "CondensationTrap":
        """Deserialize from a dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TickResult:
    """Output of a single sol tick."""

    yield_g: float = 0.0
    tec_power_w: float = 0.0
    fan_power_w: float = 0.0
    total_power_w: float = 0.0
    energy_wh: float = 0.0
    plate_temp_k: float = 0.0
    frost_point_k: float = 0.0
    cop: float = 0.0
    condensation_rate_mg_s: float = 0.0
    dust_after: float = 0.0
    ice_collected_g: float = 0.0
    water_extracted_g: float = 0.0
    melted_this_sol: bool = False
    melt_energy_wh: float = 0.0


# -- Tick engine --------------------------------------------------------------

def tick(state: CondensationTrap, clean_dust: bool = False, melt_ice: bool = False) -> TickResult:
    """Advance the condensation trap by one sol."""
    result = TickResult()

    if clean_dust:
        removed = state.dust_fraction * DUST_CLEANING_EFFICIENCY
        state.dust_fraction = max(0.0, state.dust_fraction - removed)

    p_h2o = water_partial_pressure(state.atm_pressure_pa, state.mixing_ratio)
    result.frost_point_k = frost_point(p_h2o)
    result.plate_temp_k = state.plate_temp_k
    p_sat_plate = saturation_pressure_ice(state.plate_temp_k)

    rate = condensation_rate_kg_s(
        state.plate_area_m2, state.wind_speed_m_s,
        p_h2o, p_sat_plate, state.ambient_temp_k, state.dust_fraction,
    )
    result.condensation_rate_mg_s = rate * 1e6
    sol_yield_g = rate * EFFECTIVE_COLLECTION_SECONDS * 1000.0
    result.yield_g = sol_yield_g
    state.ice_collected_g += sol_yield_g

    q_cooling = cooling_power_w(rate)
    cop = tec_cop(state.plate_temp_k, state.ambient_temp_k)
    result.cop = cop
    tec_power = tec_input_power_w(q_cooling, cop) + TEC_IDLE_POWER_W
    result.tec_power_w = tec_power
    result.fan_power_w = state.fan_power_w
    cleaning_power = DUST_CLEANING_POWER_W * (1.0 / HOURS_PER_SOL) if clean_dust else 0.0
    result.total_power_w = tec_power + state.fan_power_w + cleaning_power
    result.energy_wh = result.total_power_w * HOURS_PER_SOL * COLLECTION_DUTY_CYCLE
    state.total_energy_wh += result.energy_wh

    if melt_ice and state.ice_collected_g > 0.0:
        ice_kg = state.ice_collected_g / 1000.0
        melt_j = melt_energy_j(ice_kg, state.plate_temp_k)
        result.melt_energy_wh = melt_j / 3600.0
        state.total_energy_wh += result.melt_energy_wh
        result.energy_wh += result.melt_energy_wh
        state.water_extracted_g += state.ice_collected_g
        state.total_water_g += state.ice_collected_g
        state.ice_collected_g = 0.0
        result.melted_this_sol = True

    result.ice_collected_g = state.ice_collected_g
    result.water_extracted_g = state.water_extracted_g
    state.dust_fraction = min(1.0, state.dust_fraction + DUST_ACCUMULATION_PER_SOL)
    result.dust_after = state.dust_fraction
    state.sol += 1
    state.total_sols_operated += 1
    if sol_yield_g > state.peak_yield_g:
        state.peak_yield_g = sol_yield_g
    return result


def run_simulation(state: CondensationTrap, sols: int = 100, melt_interval: int = 10) -> list:
    """Run the trap for multiple sols with periodic melt and cleaning."""
    results = []
    for s in range(sols):
        clean = (s > 0 and s % 30 == 0)
        melt = (s > 0 and s % melt_interval == 0)
        results.append(tick(state, clean_dust=clean, melt_ice=melt))
    return results
