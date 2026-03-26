"""water_heater.py — Mars Colony Water Heating & Thermal Storage

Mars averages −60 °C.  Every drop of water starts as ice.  Before the
colony can drink, wash, irrigate, or run industrial chemistry, it needs
hot water — and somewhere to keep it hot when the reactor cycles or
the sun sets.

Physics
-------
* Specific heat of water:  4.186 kJ/(kg·K)
* Heat of fusion (ice→water): 334 kJ/kg
* Mars ambient temperature: −60 °C mean (seasonal ±50 °C)
* Heat loss: Newton's law of cooling.  Tank insulation R-value
  determines steady-state loss rate.
* Energy input: waste heat from nuclear reactor (primary) or
  electric resistance heating from solar/battery (backup).
* Thermal stratification: hot water rises, cold sinks.  We model
  two layers (hot top, cold bottom) with a mixing coefficient.

Design
------
* Insulated tank: 2000 L capacity, R-30 vacuum-panel insulation.
* Primary heat: reactor coolant loop (up to 50 kW thermal).
* Backup heat: electric resistance element (5 kW).
* Demand profile: crew hygiene, greenhouse, medical, industrial.
* Safety: overheat protection at 95 °C, freeze protection at 2 °C.

One tick = one sol.  Energy in kWh, temperatures in °C, volumes in litres.

Reference: ISS Water Recovery System processes ~9 L/crew/day.
  Mars Design Reference Architecture 5.0 estimates 25 L/crew/day
  with recycling.  We model 20 L/crew/day nominal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# -- Physical constants -------------------------------------------------------

SPECIFIC_HEAT_WATER_KJ_KG_K = 4.186     # kJ/(kg·K)
HEAT_OF_FUSION_KJ_KG = 334.0            # kJ/kg  (ice → water at 0°C)
WATER_DENSITY_KG_L = 1.0                # kg/L  (close enough for potable water)
HOURS_PER_SOL = 24.62                    # Mars sol in Earth hours
SECONDS_PER_SOL = HOURS_PER_SOL * 3600.0
KJ_PER_KWH = 3600.0                     # 1 kWh = 3600 kJ

# Mars environment
MARS_AMBIENT_MEAN_C = -60.0
MARS_AMBIENT_AMPLITUDE_C = 50.0          # seasonal swing

# Tank design parameters
DEFAULT_TANK_CAPACITY_L = 2000.0         # litres
DEFAULT_INSULATION_R = 30.0              # R-value (m²·K/W equivalent)
DEFAULT_TANK_SURFACE_M2 = 8.0            # ~2m diameter sphere ≈ 12.6 m², tank is cylindrical ~8 m²
DEFAULT_REACTOR_HEAT_KW = 50.0           # max thermal input from reactor coolant
DEFAULT_ELECTRIC_HEAT_KW = 5.0           # backup electric resistance heater
DEFAULT_TARGET_TEMP_C = 60.0             # target hot water temperature
OVERHEAT_LIMIT_C = 95.0                  # safety shutoff
FREEZE_PROTECT_C = 2.0                   # activate heating below this

# Crew demand
WATER_PER_CREW_L_SOL = 20.0             # litres/crew/sol (DRA 5.0 estimate)
DEMAND_TEMP_C = 40.0                     # water delivered at this temp for use

# Thermal stratification
MIXING_COEFFICIENT = 0.15                # fraction of layers that mix per sol
MIN_TEMP_C = -40.0                       # below this, ice damage to tank


# -- Pure functions -----------------------------------------------------------

def heat_energy_kwh(mass_kg: float, delta_t_c: float) -> float:
    """Energy to heat mass_kg of water by delta_t_c degrees.

    Returns kWh (always non-negative; caller decides sign).
    """
    kj = mass_kg * SPECIFIC_HEAT_WATER_KJ_KG_K * abs(delta_t_c)
    return kj / KJ_PER_KWH


def ice_melt_energy_kwh(mass_kg: float) -> float:
    """Energy to melt mass_kg of ice at 0°C to water at 0°C."""
    return mass_kg * HEAT_OF_FUSION_KJ_KG / KJ_PER_KWH


def heat_loss_kwh(tank_temp_c: float, ambient_c: float,
                  surface_m2: float, r_value: float) -> float:
    """Heat lost through insulation per sol (Newton's law of cooling).

    U = 1/R  (thermal transmittance, W/(m²·K))
    Q = U * A * ΔT * t  (energy in Joules, convert to kWh)
    """
    if tank_temp_c <= ambient_c:
        return 0.0
    u = 1.0 / max(r_value, 0.1)
    delta_t = tank_temp_c - ambient_c
    watts = u * surface_m2 * delta_t
    kwh = watts * HOURS_PER_SOL / 1000.0
    return kwh


def thermal_stratification(hot_c: float, cold_c: float,
                           mix_coeff: float) -> Tuple[float, float]:
    """Model two-layer stratification with partial mixing.

    Returns (new_hot, new_cold) after one sol of mixing.
    """
    mix = min(max(mix_coeff, 0.0), 1.0)
    avg = (hot_c + cold_c) / 2.0
    new_hot = hot_c - mix * (hot_c - avg)
    new_cold = cold_c + mix * (avg - cold_c)
    return round(new_hot, 4), round(new_cold, 4)


def demand_volume_l(crew_count: int, extra_industrial_l: float = 0.0) -> float:
    """Total hot water demand per sol."""
    return crew_count * WATER_PER_CREW_L_SOL + extra_industrial_l


def demand_energy_kwh(volume_l: float, supply_temp_c: float,
                      demand_temp_c: float) -> float:
    """Energy to heat demand volume from supply_temp to demand_temp.

    If supply is already hot enough, returns 0.
    """
    if supply_temp_c >= demand_temp_c:
        return 0.0
    mass_kg = volume_l * WATER_DENSITY_KG_L
    return heat_energy_kwh(mass_kg, demand_temp_c - supply_temp_c)


def heater_output_kwh(reactor_kw: float, electric_kw: float,
                      reactor_available: bool) -> float:
    """Total heating capacity available per sol."""
    thermal = reactor_kw if reactor_available else 0.0
    return (thermal + electric_kw) * HOURS_PER_SOL


def seasonal_ambient_c(sol: int, mean_c: float = MARS_AMBIENT_MEAN_C,
                       amplitude_c: float = MARS_AMBIENT_AMPLITUDE_C) -> float:
    """Approximate ambient temperature for a given sol."""
    sols_per_year = 668.6
    phase = (sol / sols_per_year) * 2.0 * math.pi
    return mean_c + amplitude_c * math.sin(phase)


def efficiency_factor(tank_temp_c: float) -> float:
    """Heating efficiency drops as tank approaches boiling.

    Models real heat exchanger performance degradation.
    Returns 0.0-1.0.
    """
    if tank_temp_c >= OVERHEAT_LIMIT_C:
        return 0.0
    if tank_temp_c <= 0.0:
        return 1.0
    return max(0.1, 1.0 - (tank_temp_c / OVERHEAT_LIMIT_C) ** 2)


# -- State machine ------------------------------------------------------------

@dataclass
class HeaterState:
    """Mutable state of the water heating system."""
    sol: int = 0
    tank_volume_l: float = DEFAULT_TANK_CAPACITY_L
    hot_layer_temp_c: float = 10.0
    cold_layer_temp_c: float = 5.0
    tank_capacity_l: float = DEFAULT_TANK_CAPACITY_L
    insulation_r: float = DEFAULT_INSULATION_R
    surface_m2: float = DEFAULT_TANK_SURFACE_M2
    reactor_heat_kw: float = DEFAULT_REACTOR_HEAT_KW
    electric_heat_kw: float = DEFAULT_ELECTRIC_HEAT_KW
    target_temp_c: float = DEFAULT_TARGET_TEMP_C
    cumulative_energy_kwh: float = 0.0
    cumulative_loss_kwh: float = 0.0
    cumulative_demand_kwh: float = 0.0
    heating_active: bool = True
    freeze_alarm: bool = False
    overheat_alarm: bool = False
    sols_frozen: int = 0
    history: List[Dict] = field(default_factory=list)


def make_heater(tank_capacity_l: float = DEFAULT_TANK_CAPACITY_L,
                initial_temp_c: float = 10.0,
                insulation_r: float = DEFAULT_INSULATION_R,
                reactor_kw: float = DEFAULT_REACTOR_HEAT_KW,
                electric_kw: float = DEFAULT_ELECTRIC_HEAT_KW) -> HeaterState:
    """Create a new water heater with specified parameters."""
    return HeaterState(
        tank_capacity_l=tank_capacity_l,
        tank_volume_l=tank_capacity_l,
        hot_layer_temp_c=initial_temp_c,
        cold_layer_temp_c=initial_temp_c - 5.0,
        insulation_r=insulation_r,
        reactor_heat_kw=reactor_kw,
        electric_heat_kw=electric_kw,
    )


@dataclass
class SolReport:
    """Report for one sol of water heater operation."""
    sol: int
    hot_temp_c: float
    cold_temp_c: float
    avg_temp_c: float
    ambient_c: float
    heat_loss_kwh: float
    heat_input_kwh: float
    demand_kwh: float
    net_energy_kwh: float
    freeze_alarm: bool
    overheat_alarm: bool
    efficiency: float


def tick_heater(state: HeaterState, crew_count: int = 6,
                reactor_available: bool = True,
                extra_demand_l: float = 0.0,
                ambient_override: Optional[float] = None) -> SolReport:
    """Advance the water heater by one sol.

    This is the core mutation: reads state at time T, produces T+1.
    """
    state.sol += 1
    sol = state.sol

    # Environment
    ambient = ambient_override if ambient_override is not None else seasonal_ambient_c(sol)

    # --- Heat loss through insulation ---
    avg_temp = (state.hot_layer_temp_c + state.cold_layer_temp_c) / 2.0
    loss = heat_loss_kwh(avg_temp, ambient, state.surface_m2, state.insulation_r)
    state.cumulative_loss_kwh += loss

    # --- Demand ---
    vol = demand_volume_l(crew_count, extra_demand_l)
    d_energy = demand_energy_kwh(vol, state.cold_layer_temp_c, DEMAND_TEMP_C)
    state.cumulative_demand_kwh += d_energy

    # --- Available heating ---
    max_input = heater_output_kwh(state.reactor_heat_kw, state.electric_heat_kw,
                                   reactor_available)
    eff = efficiency_factor(avg_temp)
    effective_input = max_input * eff

    # Energy balance: input - loss - demand
    net = effective_input - loss - d_energy
    state.cumulative_energy_kwh += max(0.0, effective_input)

    # --- Temperature update ---
    tank_mass_kg = state.tank_volume_l * WATER_DENSITY_KG_L
    half_mass = tank_mass_kg / 2.0

    if half_mass > 0:
        # Net energy heats the hot layer primarily
        delta_hot = (net * KJ_PER_KWH) / (half_mass * SPECIFIC_HEAT_WATER_KJ_KG_K)
        state.hot_layer_temp_c += delta_hot * 0.7
        state.cold_layer_temp_c += delta_hot * 0.3

    # Stratification mixing
    state.hot_layer_temp_c, state.cold_layer_temp_c = thermal_stratification(
        state.hot_layer_temp_c, state.cold_layer_temp_c, MIXING_COEFFICIENT
    )

    # --- Safety limits ---
    state.overheat_alarm = state.hot_layer_temp_c >= OVERHEAT_LIMIT_C
    if state.overheat_alarm:
        state.hot_layer_temp_c = OVERHEAT_LIMIT_C - 1.0
        state.heating_active = False

    state.freeze_alarm = state.cold_layer_temp_c <= FREEZE_PROTECT_C
    if state.freeze_alarm:
        state.heating_active = True
        state.sols_frozen += 1

    # Clamp temperatures to physical bounds
    state.hot_layer_temp_c = max(MIN_TEMP_C, min(OVERHEAT_LIMIT_C, state.hot_layer_temp_c))
    state.cold_layer_temp_c = max(MIN_TEMP_C, min(state.hot_layer_temp_c, state.cold_layer_temp_c))

    new_avg = (state.hot_layer_temp_c + state.cold_layer_temp_c) / 2.0

    report = SolReport(
        sol=sol,
        hot_temp_c=round(state.hot_layer_temp_c, 2),
        cold_temp_c=round(state.cold_layer_temp_c, 2),
        avg_temp_c=round(new_avg, 2),
        ambient_c=round(ambient, 2),
        heat_loss_kwh=round(loss, 4),
        heat_input_kwh=round(effective_input, 4),
        demand_kwh=round(d_energy, 4),
        net_energy_kwh=round(net, 4),
        freeze_alarm=state.freeze_alarm,
        overheat_alarm=state.overheat_alarm,
        efficiency=round(eff, 4),
    )

    state.history.append({
        "sol": sol,
        "hot_c": report.hot_temp_c,
        "cold_c": report.cold_temp_c,
        "avg_c": report.avg_temp_c,
        "loss_kwh": report.heat_loss_kwh,
        "input_kwh": report.heat_input_kwh,
    })

    return report


def run_heater(sols: int = 365, crew: int = 6, seed: int = 42,
               reactor_available: bool = True) -> Tuple[HeaterState, List[SolReport]]:
    """Run the heater for N sols. Returns final state and all reports."""
    state = make_heater()
    reports: List[SolReport] = []
    for _ in range(sols):
        r = tick_heater(state, crew_count=crew, reactor_available=reactor_available)
        reports.append(r)
    return state, reports


def heater_summary(state: HeaterState, reports: List[SolReport]) -> Dict:
    """Produce a summary dict for the heater run."""
    if not reports:
        return {"error": "no reports"}
    temps = [r.avg_temp_c for r in reports]
    losses = [r.heat_loss_kwh for r in reports]
    return {
        "sols": state.sol,
        "final_hot_c": state.hot_layer_temp_c,
        "final_cold_c": state.cold_layer_temp_c,
        "min_avg_c": min(temps),
        "max_avg_c": max(temps),
        "mean_avg_c": round(sum(temps) / len(temps), 2),
        "total_energy_kwh": round(state.cumulative_energy_kwh, 2),
        "total_loss_kwh": round(state.cumulative_loss_kwh, 2),
        "total_demand_kwh": round(state.cumulative_demand_kwh, 2),
        "sols_frozen": state.sols_frozen,
        "freeze_events": sum(1 for r in reports if r.freeze_alarm),
        "overheat_events": sum(1 for r in reports if r.overheat_alarm),
        "mean_loss_kwh_sol": round(sum(losses) / len(losses), 4),
    }
