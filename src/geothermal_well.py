"""geothermal_well.py — Mars Geothermal Energy Extraction via Binary Cycle ORC.

Deep beneath the Martian surface, residual heat from the planet's
formation and radiogenic decay warms the basaltic crust.  While Mars
has lost most of its internal heat engine (no active plate tectonics),
volcanic provinces like Tharsis and Elysium Mons retain elevated
geothermal gradients — perhaps 15–20 K/km vs the planetary average of
~5 K/km.  A 3 km deep well into Tharsis basalt finds rock at ~255 K:
cold by Earth geothermal standards, but enough to drive a Binary Cycle
Organic Rankine Cycle (ORC) using a low-boiling-point working fluid
such as isobutane.

Physics modelled
----------------
* **Geothermal gradient** — T(depth) = T_surface + gradient × depth_km.
  Mars average ~5 K/km; Tharsis volcanic province ~15–20 K/km.
  Surface temperature ~210 K (−63°C).

* **Closed-loop well** — No aquifer on Mars.  Cold working fluid is
  injected down, absorbs heat from borehole rock, and returns hot.
  Heat exchange efficiency 60–80% (fluid never reaches full rock temp).

* **Binary Cycle ORC** — Low-temperature heat source drives an organic
  working fluid (isobutane, R245fa).  Two thermodynamic limits:
  - Carnot efficiency: η_max = 1 − T_cold / T_hot
  - Practical ORC: ~40–50% of Carnot (friction, irreversibilities)
  - Net power = thermal_power × ORC_efficiency − parasitic_loads

* **Thermal drawdown** — Rock cools over time as heat is extracted
  faster than conductive replenishment from below.  Modelled as
  exponential decay of the rock-to-surface delta toward an equilibrium
  where extraction ≈ conduction.  Rate depends on extraction vs rock
  thermal conductivity (~2.0 W/m·K for basalt).

* **Pump parasitic loads** — Circulating the working fluid requires
  pump energy ≈ flow_rate × pressure_drop / pump_efficiency.
  Typically 5–15% of gross thermal power.

* **Scaling / fouling** — Mineral deposits slowly degrade the heat
  exchanger efficiency.  Modelled as a scaling factor that decreases
  ~0.01% per sol.

Conservation laws
-----------------
- Electrical output ≤ thermal input (Second Law)
- Rock temperature ≥ surface temperature (heat flows downhill)
- All power values ≥ 0
- Scaling factor ∈ [0, 1]
- ORC efficiency ≤ Carnot efficiency
- Rock temperature monotonically decreasing under drawdown

Reference:
  - Mars surface temperature: ~210 K (varies 130–308 K, polar to equator)
  - Earth avg geothermal gradient: ~25 K/km
  - Mars avg geothermal gradient: ~5 K/km (Plesa et al. 2016)
  - Tharsis region gradient: ~15–20 K/km (Thiriet et al. 2018)
  - Basalt thermal conductivity: 1.5–2.5 W/(m·K)
  - Isobutane Cp: ~2400 J/(kg·K) near critical point
  - Mars sol: 24 h 39 m 35 s ≈ 24.66 hours

One tick = one sol.  Power in kW, energy in kWh, temperature in K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Physical constants ──────────────────────────────────────────────

# Mars surface
MARS_SURFACE_TEMP_K = 210.0              # average surface temperature

# Geothermal
GEOTHERMAL_GRADIENT_K_PER_KM = 15.0      # Tharsis region default
EARTH_GRADIENT_K_PER_KM = 25.0           # for reference

# Rock properties (basalt)
ROCK_THERMAL_CONDUCTIVITY_W_MK = 2.0     # W/(m·K)
ROCK_HEAT_CAPACITY_J_KGK = 840.0         # J/(kg·K)
ROCK_DENSITY_KG_M3 = 2900.0              # kg/m³

# Working fluid (isobutane-like)
FLUID_SPECIFIC_HEAT_J_KGK = 2400.0       # J/(kg·K)

# Well geometry
DEFAULT_WELL_DEPTH_M = 3000.0            # 3 km
WELL_BORE_RADIUS_M = 0.15               # borehole radius

# Flow and efficiency defaults
DEFAULT_FLOW_RATE_KG_S = 5.0             # working fluid mass flow
DEFAULT_HEAT_EXCHANGE_EFFICIENCY = 0.75  # 75% heat transfer
ORC_CARNOT_FRACTION = 0.45              # practical ORC ≈ 45% of Carnot
PUMP_PARASITIC_FRACTION = 0.10          # pump uses ~10% of gross thermal

# Degradation rates
SCALING_DEGRADATION_PER_SOL = 0.0001    # 0.01% per sol
THERMAL_DRAWDOWN_FRACTION_PER_SOL = 0.00005  # rock cools 0.005%/sol

# Time
HOURS_PER_SOL = 24.66                   # Mars sol length
SECONDS_PER_SOL = HOURS_PER_SOL * 3600.0

# Pump modelling
PUMP_EFFICIENCY = 0.75                  # mechanical pump efficiency
PRESSURE_DROP_PA = 500_000.0            # 5 bar pressure drop in loop

# Minimum operational threshold
MIN_THERMAL_POWER_KW = 0.01            # below this, well is exhausted


# ── Pure physics functions ──────────────────────────────────────────

def rock_temperature_at_depth(depth_m: float,
                              gradient_k_per_km: float = GEOTHERMAL_GRADIENT_K_PER_KM,
                              surface_temp_k: float = MARS_SURFACE_TEMP_K) -> float:
    """Temperature of rock at a given depth.

    T(depth) = T_surface + gradient × depth_km

    Parameters
    ----------
    depth_m : float
        Well depth in metres.
    gradient_k_per_km : float
        Geothermal gradient in K/km.
    surface_temp_k : float
        Surface temperature in K.

    Returns
    -------
    float
        Rock temperature in K (never below surface temp).
    """
    depth_m = max(0.0, depth_m)
    gradient_k_per_km = max(0.0, gradient_k_per_km)
    surface_temp_k = max(0.0, surface_temp_k)
    depth_km = depth_m / 1000.0
    return surface_temp_k + gradient_k_per_km * depth_km


def fluid_outlet_temperature(rock_temp_k: float,
                             surface_temp_k: float,
                             heat_exchange_eff: float,
                             scaling_factor: float = 1.0) -> float:
    """Temperature of working fluid exiting the well.

    The fluid absorbs a fraction of the rock-to-surface delta,
    degraded by heat exchanger scaling.

    Parameters
    ----------
    rock_temp_k : float
        Rock temperature at well bottom (K).
    surface_temp_k : float
        Injection temperature of cold fluid (≈ surface temp, K).
    heat_exchange_eff : float
        Base heat exchange efficiency [0, 1].
    scaling_factor : float
        Scaling/fouling factor [0, 1], degrades over time.

    Returns
    -------
    float
        Fluid outlet temperature in K.
    """
    heat_exchange_eff = max(0.0, min(1.0, heat_exchange_eff))
    scaling_factor = max(0.0, min(1.0, scaling_factor))
    delta = max(0.0, rock_temp_k - surface_temp_k)
    return surface_temp_k + delta * heat_exchange_eff * scaling_factor


def carnot_efficiency(t_hot_k: float, t_cold_k: float) -> float:
    """Maximum theoretical (Carnot) efficiency.

    η_carnot = 1 − T_cold / T_hot

    Returns 0 if T_hot ≤ T_cold or T_hot ≤ 0.
    """
    if t_hot_k <= 0.0 or t_hot_k <= t_cold_k:
        return 0.0
    return 1.0 - t_cold_k / t_hot_k


def orc_efficiency(t_hot_k: float, t_cold_k: float,
                   carnot_fraction: float = ORC_CARNOT_FRACTION) -> float:
    """Practical ORC efficiency as a fraction of Carnot.

    η_orc = carnot_fraction × η_carnot
    """
    carnot_fraction = max(0.0, min(1.0, carnot_fraction))
    return carnot_fraction * carnot_efficiency(t_hot_k, t_cold_k)


def thermal_power_kw(flow_rate_kg_s: float,
                     fluid_cp_j_kgk: float,
                     t_hot_k: float,
                     t_cold_k: float) -> float:
    """Thermal power extracted from the well (kW).

    Q = ṁ × Cp × ΔT  (converted from W to kW)
    """
    flow_rate_kg_s = max(0.0, flow_rate_kg_s)
    delta_t = max(0.0, t_hot_k - t_cold_k)
    return flow_rate_kg_s * fluid_cp_j_kgk * delta_t / 1000.0


def electrical_power_kw(thermal_kw: float,
                        efficiency: float) -> float:
    """Gross electrical power from ORC turbine (kW).

    P_elec = P_thermal × η_orc
    """
    return max(0.0, thermal_kw) * max(0.0, min(1.0, efficiency))


def pump_power_kw(flow_rate_kg_s: float,
                  fluid_density_kg_m3: float = 550.0,
                  pressure_drop_pa: float = PRESSURE_DROP_PA,
                  pump_eff: float = PUMP_EFFICIENCY) -> float:
    """Parasitic pump power to circulate working fluid (kW).

    P_pump = (ṁ / ρ) × ΔP / η_pump  (converted from W to kW)

    Parameters
    ----------
    flow_rate_kg_s : float
        Mass flow rate of working fluid (kg/s).
    fluid_density_kg_m3 : float
        Density of working fluid (~550 kg/m³ for isobutane).
    pressure_drop_pa : float
        Loop pressure drop (Pa).
    pump_eff : float
        Pump mechanical efficiency.
    """
    flow_rate_kg_s = max(0.0, flow_rate_kg_s)
    fluid_density_kg_m3 = max(1.0, fluid_density_kg_m3)
    pressure_drop_pa = max(0.0, pressure_drop_pa)
    pump_eff = max(0.01, min(1.0, pump_eff))
    volume_flow_m3_s = flow_rate_kg_s / fluid_density_kg_m3
    return volume_flow_m3_s * pressure_drop_pa / pump_eff / 1000.0


def net_power_kw(gross_electrical_kw: float, pump_kw: float) -> float:
    """Net electrical power after subtracting parasitic pump load.

    Never goes below 0 — if pump exceeds generation, system shuts down.
    """
    return max(0.0, gross_electrical_kw - pump_kw)


def thermal_drawdown(current_rock_temp_k: float,
                     surface_temp_k: float,
                     drawdown_fraction: float = THERMAL_DRAWDOWN_FRACTION_PER_SOL) -> float:
    """Apply thermal drawdown to rock temperature.

    Rock cools exponentially toward surface temperature as heat is
    extracted faster than conductive replenishment.

    T_new = T_surface + (T_rock − T_surface) × (1 − drawdown_fraction)

    Returns
    -------
    float
        New rock temperature (always ≥ surface temp).
    """
    drawdown_fraction = max(0.0, min(1.0, drawdown_fraction))
    delta = max(0.0, current_rock_temp_k - surface_temp_k)
    new_temp = surface_temp_k + delta * (1.0 - drawdown_fraction)
    return max(surface_temp_k, new_temp)


def apply_scaling_degradation(current_scaling: float,
                              degradation_per_sol: float = SCALING_DEGRADATION_PER_SOL) -> float:
    """Degrade heat exchanger scaling factor by one sol.

    scaling_new = scaling_old × (1 − degradation_rate)
    Clamped to [0, 1].
    """
    degradation_per_sol = max(0.0, min(1.0, degradation_per_sol))
    new_scaling = current_scaling * (1.0 - degradation_per_sol)
    return max(0.0, min(1.0, new_scaling))


def thermal_energy_kwh(power_kw: float,
                       hours: float = HOURS_PER_SOL) -> float:
    """Convert power (kW) over a duration (hours) to energy (kWh)."""
    return max(0.0, power_kw) * max(0.0, hours)


# ── State ───────────────────────────────────────────────────────────

@dataclass
class GeothermalWell:
    """State of a Mars geothermal well and ORC power plant."""

    # Well configuration
    well_depth_m: float = DEFAULT_WELL_DEPTH_M
    gradient_k_per_km: float = GEOTHERMAL_GRADIENT_K_PER_KM
    surface_temp_k: float = MARS_SURFACE_TEMP_K
    flow_rate_kg_s: float = DEFAULT_FLOW_RATE_KG_S

    # Rock and fluid temperatures
    rock_temp_at_depth_k: float = 0.0
    fluid_outlet_temp_k: float = 0.0

    # Efficiencies and scaling
    heat_exchange_efficiency: float = DEFAULT_HEAT_EXCHANGE_EFFICIENCY
    orc_efficiency_fraction: float = 0.0
    scaling_factor: float = 1.0

    # Power outputs (instantaneous, kW)
    thermal_power_kw: float = 0.0
    gross_electrical_power_kw: float = 0.0
    pump_power_kw: float = 0.0
    net_power_kw: float = 0.0

    # Cumulative energy (kWh)
    cumulative_heat_extracted_kwh: float = 0.0
    cumulative_electrical_kwh: float = 0.0

    # Simulation
    sol: int = 0
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialise computed fields from configuration."""
        self.well_depth_m = max(0.0, self.well_depth_m)
        self.gradient_k_per_km = max(0.0, self.gradient_k_per_km)
        self.surface_temp_k = max(0.0, self.surface_temp_k)
        self.flow_rate_kg_s = max(0.0, self.flow_rate_kg_s)
        self.heat_exchange_efficiency = max(0.0, min(1.0, self.heat_exchange_efficiency))
        self.scaling_factor = max(0.0, min(1.0, self.scaling_factor))
        if self.rock_temp_at_depth_k == 0.0:
            self.rock_temp_at_depth_k = rock_temperature_at_depth(
                self.well_depth_m, self.gradient_k_per_km, self.surface_temp_k
            )
        self.rock_temp_at_depth_k = max(self.surface_temp_k, self.rock_temp_at_depth_k)

    def to_dict(self) -> dict[str, Any]:
        """Serialise state to a plain dictionary."""
        return {
            "well_depth_m": self.well_depth_m,
            "gradient_k_per_km": self.gradient_k_per_km,
            "surface_temp_k": self.surface_temp_k,
            "flow_rate_kg_s": self.flow_rate_kg_s,
            "rock_temp_at_depth_k": self.rock_temp_at_depth_k,
            "fluid_outlet_temp_k": self.fluid_outlet_temp_k,
            "heat_exchange_efficiency": self.heat_exchange_efficiency,
            "orc_efficiency_fraction": self.orc_efficiency_fraction,
            "scaling_factor": self.scaling_factor,
            "thermal_power_kw": self.thermal_power_kw,
            "gross_electrical_power_kw": self.gross_electrical_power_kw,
            "pump_power_kw": self.pump_power_kw,
            "net_power_kw": self.net_power_kw,
            "cumulative_heat_extracted_kwh": self.cumulative_heat_extracted_kwh,
            "cumulative_electrical_kwh": self.cumulative_electrical_kwh,
            "sol": self.sol,
            "events": list(self.events),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeothermalWell:
        """Reconstruct state from a dictionary."""
        return cls(
            well_depth_m=data.get("well_depth_m", DEFAULT_WELL_DEPTH_M),
            gradient_k_per_km=data.get("gradient_k_per_km", GEOTHERMAL_GRADIENT_K_PER_KM),
            surface_temp_k=data.get("surface_temp_k", MARS_SURFACE_TEMP_K),
            flow_rate_kg_s=data.get("flow_rate_kg_s", DEFAULT_FLOW_RATE_KG_S),
            rock_temp_at_depth_k=data.get("rock_temp_at_depth_k", 0.0),
            fluid_outlet_temp_k=data.get("fluid_outlet_temp_k", 0.0),
            heat_exchange_efficiency=data.get("heat_exchange_efficiency", DEFAULT_HEAT_EXCHANGE_EFFICIENCY),
            orc_efficiency_fraction=data.get("orc_efficiency_fraction", 0.0),
            scaling_factor=data.get("scaling_factor", 1.0),
            thermal_power_kw=data.get("thermal_power_kw", 0.0),
            gross_electrical_power_kw=data.get("gross_electrical_power_kw", 0.0),
            pump_power_kw=data.get("pump_power_kw", 0.0),
            net_power_kw=data.get("net_power_kw", 0.0),
            cumulative_heat_extracted_kwh=data.get("cumulative_heat_extracted_kwh", 0.0),
            cumulative_electrical_kwh=data.get("cumulative_electrical_kwh", 0.0),
            sol=data.get("sol", 0),
            events=list(data.get("events", [])),
        )


# ── Tick result ─────────────────────────────────────────────────────

@dataclass
class TickResult:
    """Result snapshot for one sol of geothermal well operation."""
    sol: int = 0
    rock_temp_k: float = 0.0
    fluid_outlet_temp_k: float = 0.0
    thermal_power_kw: float = 0.0
    carnot_eff: float = 0.0
    orc_eff: float = 0.0
    gross_electrical_kw: float = 0.0
    pump_kw: float = 0.0
    net_kw: float = 0.0
    thermal_energy_kwh: float = 0.0
    net_energy_kwh: float = 0.0
    scaling_factor: float = 1.0
    events: list[str] = field(default_factory=list)


# ── Tick engine ─────────────────────────────────────────────────────

def tick(state: GeothermalWell) -> TickResult:
    """Advance the geothermal well simulation by one sol.

    Steps:
    1. Increment sol counter.
    2. Apply thermal drawdown to rock temperature.
    3. Calculate fluid outlet temperature (with scaling).
    4. Calculate thermal power extracted.
    5. Calculate ORC efficiency (fraction of Carnot).
    6. Calculate gross electrical power.
    7. Calculate pump parasitic power.
    8. Compute net power.
    9. Apply scaling degradation.
    10. Accumulate energy totals.

    Parameters
    ----------
    state : GeothermalWell
        Mutable well state — modified in place.

    Returns
    -------
    TickResult
        Snapshot of this sol's outputs.
    """
    state.sol += 1
    events: list[str] = []

    # 1 — Thermal drawdown: rock cools toward surface temp
    state.rock_temp_at_depth_k = thermal_drawdown(
        state.rock_temp_at_depth_k,
        state.surface_temp_k,
    )

    # 2 — Fluid outlet temperature
    state.fluid_outlet_temp_k = fluid_outlet_temperature(
        state.rock_temp_at_depth_k,
        state.surface_temp_k,
        state.heat_exchange_efficiency,
        state.scaling_factor,
    )

    # 3 — Thermal power
    t_hot = state.fluid_outlet_temp_k
    t_cold = state.surface_temp_k
    state.thermal_power_kw = thermal_power_kw(
        state.flow_rate_kg_s,
        FLUID_SPECIFIC_HEAT_J_KGK,
        t_hot,
        t_cold,
    )

    # 4 — ORC efficiency
    carnot_eff = carnot_efficiency(t_hot, t_cold)
    state.orc_efficiency_fraction = orc_efficiency(t_hot, t_cold)

    # 5 — Gross electrical power
    state.gross_electrical_power_kw = electrical_power_kw(
        state.thermal_power_kw, state.orc_efficiency_fraction
    )

    # 6 — Pump parasitic load
    state.pump_power_kw = pump_power_kw(state.flow_rate_kg_s)

    # 7 — Net power
    state.net_power_kw = net_power_kw(
        state.gross_electrical_power_kw, state.pump_power_kw
    )

    # 8 — Scaling degradation
    old_scaling = state.scaling_factor
    state.scaling_factor = apply_scaling_degradation(state.scaling_factor)
    if old_scaling >= 0.90 and state.scaling_factor < 0.90:
        events.append("SCALING WARNING — heat exchanger below 90% efficiency")
    if old_scaling >= 0.50 and state.scaling_factor < 0.50:
        events.append("SCALING CRITICAL — heat exchanger below 50% efficiency")

    # 9 — Cumulative energy
    sol_thermal_kwh = thermal_energy_kwh(state.thermal_power_kw)
    sol_net_kwh = thermal_energy_kwh(state.net_power_kw)
    state.cumulative_heat_extracted_kwh += sol_thermal_kwh
    state.cumulative_electrical_kwh += sol_net_kwh

    # 10 — Operational events
    if state.thermal_power_kw < MIN_THERMAL_POWER_KW:
        events.append("WELL EXHAUSTED — thermal output below threshold")
    if state.net_power_kw <= 0.0 and state.thermal_power_kw > 0.0:
        events.append("PARASITIC LOSS — pump exceeds generation")

    state.events = events

    return TickResult(
        sol=state.sol,
        rock_temp_k=state.rock_temp_at_depth_k,
        fluid_outlet_temp_k=state.fluid_outlet_temp_k,
        thermal_power_kw=state.thermal_power_kw,
        carnot_eff=carnot_eff,
        orc_eff=state.orc_efficiency_fraction,
        gross_electrical_kw=state.gross_electrical_power_kw,
        pump_kw=state.pump_power_kw,
        net_kw=state.net_power_kw,
        thermal_energy_kwh=sol_thermal_kwh,
        net_energy_kwh=sol_net_kwh,
        scaling_factor=state.scaling_factor,
        events=list(events),
    )


# ── Simulation runner ───────────────────────────────────────────────

def run_simulation(
    sols: int = 365,
    well_depth_m: float = DEFAULT_WELL_DEPTH_M,
    gradient_k_per_km: float = GEOTHERMAL_GRADIENT_K_PER_KM,
    flow_rate_kg_s: float = DEFAULT_FLOW_RATE_KG_S,
) -> list[TickResult]:
    """Run a geothermal well simulation for *sols* Mars sols.

    Parameters
    ----------
    sols : int
        Number of sols to simulate.
    well_depth_m : float
        Depth of the geothermal well (metres).
    gradient_k_per_km : float
        Local geothermal gradient (K/km).
    flow_rate_kg_s : float
        Working fluid mass flow rate (kg/s).

    Returns
    -------
    list[TickResult]
        Per-sol output snapshots.
    """
    state = GeothermalWell(
        well_depth_m=well_depth_m,
        gradient_k_per_km=gradient_k_per_km,
        flow_rate_kg_s=flow_rate_kg_s,
    )
    results: list[TickResult] = []
    for _ in range(sols):
        result = tick(state)
        results.append(result)
    return results


# ── CLI entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    sols = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    depth = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_WELL_DEPTH_M
    gradient = float(sys.argv[3]) if len(sys.argv) > 3 else GEOTHERMAL_GRADIENT_K_PER_KM

    results = run_simulation(sols=sols, well_depth_m=depth, gradient_k_per_km=gradient)

    print(f"Geothermal Well Simulation — {sols} sols")
    print(f"  Depth: {depth:.0f} m | Gradient: {gradient:.1f} K/km")
    print(f"  Initial rock temp: {results[0].rock_temp_k:.1f} K")
    print(f"  Final rock temp:   {results[-1].rock_temp_k:.1f} K")
    print(f"  Final net power:   {results[-1].net_kw:.2f} kW")
    print(f"  Total net energy:  {sum(r.net_energy_kwh for r in results):.1f} kWh")
    print()

    sample = results[:3] + results[-3:]
    for r in sample:
        print(json.dumps({
            "sol": r.sol,
            "rock_temp_k": round(r.rock_temp_k, 2),
            "net_kw": round(r.net_kw, 4),
            "scaling": round(r.scaling_factor, 6),
        }))
