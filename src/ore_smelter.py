"""ore_smelter.py -- Mars Regolith Metal Extraction via Molten Oxide Electrolysis.

The colony mines regolith (regolith_processor) and builds things
(fabricator), but cannot produce structural metal from local resources.
Every bolt, beam, and pipe must come from Earth at $54,000/kg.  Molten
Oxide Electrolysis (MOE) closes this gap: it melts regolith at ~1600 C,
passes current through the melt, and separates metal from oxygen.

Physics modelled
----------------
* Faraday's law of electrolysis -- m = (I * t * M * eta) / (n * F)
* Minimum cell voltage -- E_min = dG / (n * F) ~ 0.78 V for FeO
* Joule heating -- P = I^2 * R (offsets furnace heating)
* Stefan-Boltzmann radiation loss from furnace shell
* Electrode degradation -- linear wear per kg metal produced
* Oxygen byproduct -- FeO -> Fe + 1/2 O2 (~0.286 kg O2 per kg Fe)
* Current efficiency -- 70-90% of charge produces metal

Conservation laws: mass balance, Faraday limit, energy >= Gibbs,
temperature >= melting point, electrode health in [0,1], all >= 0.

One tick = one sol.  Power in kW, mass in kg, temperature in K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# -- Physical constants -------------------------------------------------------

FARADAY_CONSTANT = 96_485.0
STEFAN_BOLTZMANN = 5.670374419e-8

FE_MOLAR_MASS_KG = 0.05585
FEO_MOLAR_MASS_KG = 0.07184
O2_MOLAR_MASS_KG = 0.03200
FE_ELECTRONS = 2
FEO_GIBBS_KJ_MOL = 150.0

IRON_OXIDE_FRACTION = 0.18
FE_FRACTION_IN_FEO = FE_MOLAR_MASS_KG / FEO_MOLAR_MASS_KG

DEFAULT_CELL_VOLTAGE_V = 2.0
MIN_CELL_VOLTAGE_V = FEO_GIBBS_KJ_MOL * 1000.0 / (FE_ELECTRONS * FARADAY_CONSTANT)
DEFAULT_CURRENT_A = 500.0
DEFAULT_CURRENT_EFFICIENCY = 0.80
CELL_RESISTANCE_OHM = 0.003

MELTING_TEMP_K = 1873.0
MARS_AMBIENT_TEMP_K = 210.0
FURNACE_SURFACE_AREA_M2 = 2.0
FURNACE_EMISSIVITY = 0.85
INSULATION_FACTOR = 0.15
HEATER_POWER_KW = 5.0

ELECTRODE_WEAR_PER_KG_FE = 0.0001
ELECTRODE_REPLACEMENT_THRESHOLD = 0.10

DEFAULT_FEED_RATE_KG_PER_SOL = 50.0

HOURS_PER_SOL = 24.66
SECONDS_PER_SOL = HOURS_PER_SOL * 3600.0

O2_PER_KG_FE = (0.5 * O2_MOLAR_MASS_KG) / FE_MOLAR_MASS_KG


# -- Pure physics functions ---------------------------------------------------

def faraday_mass_kg(current_a: float, time_s: float,
                    molar_mass_kg: float = FE_MOLAR_MASS_KG,
                    electrons: int = FE_ELECTRONS,
                    efficiency: float = DEFAULT_CURRENT_EFFICIENCY) -> float:
    """Mass of metal deposited by electrolysis (Faraday law)."""
    current_a = max(0.0, current_a)
    time_s = max(0.0, time_s)
    molar_mass_kg = max(0.0, molar_mass_kg)
    electrons = max(1, electrons)
    efficiency = max(0.0, min(1.0, efficiency))
    return (current_a * time_s * molar_mass_kg * efficiency) / (electrons * FARADAY_CONSTANT)


def minimum_voltage(gibbs_kj_mol: float = FEO_GIBBS_KJ_MOL,
                    electrons: int = FE_ELECTRONS) -> float:
    """Thermodynamic minimum cell voltage."""
    gibbs_j_mol = max(0.0, gibbs_kj_mol) * 1000.0
    electrons = max(1, electrons)
    return gibbs_j_mol / (electrons * FARADAY_CONSTANT)


def joule_heating_kw(current_a: float,
                     resistance_ohm: float = CELL_RESISTANCE_OHM) -> float:
    """Joule heat from cell resistance (kW)."""
    current_a = max(0.0, current_a)
    resistance_ohm = max(0.0, resistance_ohm)
    return (current_a ** 2) * resistance_ohm / 1000.0


def radiation_loss_kw(furnace_temp_k: float,
                      ambient_temp_k: float = MARS_AMBIENT_TEMP_K,
                      surface_area_m2: float = FURNACE_SURFACE_AREA_M2,
                      emissivity: float = FURNACE_EMISSIVITY,
                      insulation: float = INSULATION_FACTOR) -> float:
    """Radiative heat loss from furnace (kW)."""
    furnace_temp_k = max(0.0, furnace_temp_k)
    ambient_temp_k = max(0.0, ambient_temp_k)
    surface_area_m2 = max(0.0, surface_area_m2)
    emissivity = max(0.0, min(1.0, emissivity))
    insulation = max(0.0, min(1.0, insulation))
    delta_t4 = max(0.0, furnace_temp_k ** 4 - ambient_temp_k ** 4)
    return emissivity * STEFAN_BOLTZMANN * surface_area_m2 * delta_t4 * insulation / 1000.0


def thermal_balance_kw(joule_kw: float, heater_kw: float,
                       radiation_kw: float) -> float:
    """Net thermal power into the furnace (kW)."""
    return joule_kw + heater_kw - radiation_kw


def electrical_power_kw(voltage_v: float, current_a: float) -> float:
    """Cell electrical power (kW)."""
    return max(0.0, voltage_v) * max(0.0, current_a) / 1000.0


def iron_from_regolith_kg(regolith_kg: float,
                          iron_oxide_frac: float = IRON_OXIDE_FRACTION,
                          fe_frac_in_feo: float = FE_FRACTION_IN_FEO) -> float:
    """Maximum iron extractable from regolith (kg)."""
    regolith_kg = max(0.0, regolith_kg)
    iron_oxide_frac = max(0.0, min(1.0, iron_oxide_frac))
    fe_frac_in_feo = max(0.0, min(1.0, fe_frac_in_feo))
    return regolith_kg * iron_oxide_frac * fe_frac_in_feo


def oxygen_byproduct_kg(iron_kg: float) -> float:
    """O2 from iron reduction (kg)."""
    return max(0.0, iron_kg) * O2_PER_KG_FE


def apply_electrode_wear(health: float, iron_produced_kg: float,
                         wear_per_kg: float = ELECTRODE_WEAR_PER_KG_FE) -> float:
    """Degrade electrode health. Clamped to [0, 1]."""
    new_health = health - max(0.0, iron_produced_kg) * max(0.0, wear_per_kg)
    return max(0.0, min(1.0, new_health))


def energy_kwh(power_kw: float, hours: float = HOURS_PER_SOL) -> float:
    """Convert power * time to energy."""
    return max(0.0, power_kw) * max(0.0, hours)


# -- State --------------------------------------------------------------------

@dataclass
class OreSmelter:
    """Mutable state of a Mars MOE smelter."""
    cell_voltage_v: float = DEFAULT_CELL_VOLTAGE_V
    cell_current_a: float = DEFAULT_CURRENT_A
    current_efficiency: float = DEFAULT_CURRENT_EFFICIENCY
    cell_resistance_ohm: float = CELL_RESISTANCE_OHM
    furnace_temp_k: float = MELTING_TEMP_K
    heater_power_kw: float = HEATER_POWER_KW
    regolith_feed_kg_per_sol: float = DEFAULT_FEED_RATE_KG_PER_SOL
    electrode_health: float = 1.0
    iron_produced_kg: float = 0.0
    oxygen_produced_kg: float = 0.0
    slag_produced_kg: float = 0.0
    power_consumed_kw: float = 0.0
    cumulative_iron_kg: float = 0.0
    cumulative_oxygen_kg: float = 0.0
    cumulative_energy_kwh: float = 0.0
    cumulative_regolith_kg: float = 0.0
    sol: int = 0
    events: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cell_voltage_v = max(0.0, self.cell_voltage_v)
        self.cell_current_a = max(0.0, self.cell_current_a)
        self.current_efficiency = max(0.0, min(1.0, self.current_efficiency))
        self.cell_resistance_ohm = max(0.0, self.cell_resistance_ohm)
        self.furnace_temp_k = max(0.0, self.furnace_temp_k)
        self.heater_power_kw = max(0.0, self.heater_power_kw)
        self.regolith_feed_kg_per_sol = max(0.0, self.regolith_feed_kg_per_sol)
        self.electrode_health = max(0.0, min(1.0, self.electrode_health))

    def to_dict(self) -> dict:
        return {
            "cell_voltage_v": self.cell_voltage_v,
            "cell_current_a": self.cell_current_a,
            "current_efficiency": self.current_efficiency,
            "cell_resistance_ohm": self.cell_resistance_ohm,
            "furnace_temp_k": self.furnace_temp_k,
            "heater_power_kw": self.heater_power_kw,
            "regolith_feed_kg_per_sol": self.regolith_feed_kg_per_sol,
            "electrode_health": self.electrode_health,
            "iron_produced_kg": self.iron_produced_kg,
            "oxygen_produced_kg": self.oxygen_produced_kg,
            "slag_produced_kg": self.slag_produced_kg,
            "power_consumed_kw": self.power_consumed_kw,
            "cumulative_iron_kg": self.cumulative_iron_kg,
            "cumulative_oxygen_kg": self.cumulative_oxygen_kg,
            "cumulative_energy_kwh": self.cumulative_energy_kwh,
            "cumulative_regolith_kg": self.cumulative_regolith_kg,
            "sol": self.sol,
            "events": list(self.events),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OreSmelter":
        obj = cls(
            cell_voltage_v=data.get("cell_voltage_v", DEFAULT_CELL_VOLTAGE_V),
            cell_current_a=data.get("cell_current_a", DEFAULT_CURRENT_A),
            current_efficiency=data.get("current_efficiency", DEFAULT_CURRENT_EFFICIENCY),
            cell_resistance_ohm=data.get("cell_resistance_ohm", CELL_RESISTANCE_OHM),
            furnace_temp_k=data.get("furnace_temp_k", MELTING_TEMP_K),
            heater_power_kw=data.get("heater_power_kw", HEATER_POWER_KW),
            regolith_feed_kg_per_sol=data.get("regolith_feed_kg_per_sol", DEFAULT_FEED_RATE_KG_PER_SOL),
            electrode_health=data.get("electrode_health", 1.0),
        )
        for k in ("iron_produced_kg", "oxygen_produced_kg", "slag_produced_kg",
                   "power_consumed_kw", "cumulative_iron_kg", "cumulative_oxygen_kg",
                   "cumulative_energy_kwh", "cumulative_regolith_kg"):
            setattr(obj, k, data.get(k, 0.0))
        obj.sol = data.get("sol", 0)
        obj.events = list(data.get("events", []))
        return obj


@dataclass
class TickResult:
    """Snapshot of one sol of smelter operation."""
    sol: int = 0
    furnace_temp_k: float = 0.0
    iron_kg: float = 0.0
    oxygen_kg: float = 0.0
    slag_kg: float = 0.0
    regolith_consumed_kg: float = 0.0
    power_consumed_kw: float = 0.0
    energy_consumed_kwh: float = 0.0
    joule_heating_kw: float = 0.0
    radiation_loss_kw: float = 0.0
    thermal_balance_kw: float = 0.0
    electrode_health: float = 1.0
    operational: bool = True
    events: list = field(default_factory=list)


def tick(state: OreSmelter) -> TickResult:
    """Advance the smelter by one sol."""
    state.sol += 1
    events = []
    operational = True

    if state.furnace_temp_k < MELTING_TEMP_K:
        events.append("FURNACE COLD -- below melting point, no electrolysis")
        operational = False
    if state.electrode_health <= ELECTRODE_REPLACEMENT_THRESHOLD:
        events.append("ELECTRODE SPENT -- health below threshold, replace anode")
        operational = False
    e_min = minimum_voltage()
    if state.cell_voltage_v < e_min:
        events.append("UNDERVOLTAGE -- cell below thermodynamic minimum")
        operational = False

    if not operational:
        state.iron_produced_kg = 0.0
        state.oxygen_produced_kg = 0.0
        state.slag_produced_kg = 0.0
        state.power_consumed_kw = 0.0
        state.events = events
        return TickResult(sol=state.sol, furnace_temp_k=state.furnace_temp_k,
                          electrode_health=state.electrode_health,
                          operational=False, events=list(events))

    faraday_iron = faraday_mass_kg(state.cell_current_a, SECONDS_PER_SOL,
                                   efficiency=state.current_efficiency)
    max_from_feed = iron_from_regolith_kg(state.regolith_feed_kg_per_sol)
    actual_iron = min(faraday_iron, max_from_feed)
    actual_oxygen = oxygen_byproduct_kg(actual_iron)

    if max_from_feed > 0:
        regolith_used = state.regolith_feed_kg_per_sol * (actual_iron / max_from_feed)
    else:
        regolith_used = 0.0
    slag = max(0.0, regolith_used - actual_iron - actual_oxygen)

    cell_power = electrical_power_kw(state.cell_voltage_v, state.cell_current_a)
    total_power = cell_power + state.heater_power_kw

    joule_kw = joule_heating_kw(state.cell_current_a, state.cell_resistance_ohm)
    rad_kw = radiation_loss_kw(state.furnace_temp_k)
    thermal_net = thermal_balance_kw(joule_kw, state.heater_power_kw, rad_kw)
    if thermal_net < 0:
        events.append("THERMAL DEFICIT -- furnace cooling")

    old_health = state.electrode_health
    state.electrode_health = apply_electrode_wear(state.electrode_health, actual_iron)
    if old_health >= 0.50 and state.electrode_health < 0.50:
        events.append("ELECTRODE WARNING -- health below 50%")
    if old_health > ELECTRODE_REPLACEMENT_THRESHOLD and state.electrode_health <= ELECTRODE_REPLACEMENT_THRESHOLD:
        events.append("ELECTRODE CRITICAL -- replacement needed next sol")

    state.iron_produced_kg = actual_iron
    state.oxygen_produced_kg = actual_oxygen
    state.slag_produced_kg = slag
    state.power_consumed_kw = total_power
    sol_energy = energy_kwh(total_power)
    state.cumulative_iron_kg += actual_iron
    state.cumulative_oxygen_kg += actual_oxygen
    state.cumulative_energy_kwh += sol_energy
    state.cumulative_regolith_kg += regolith_used
    state.events = events

    return TickResult(
        sol=state.sol, furnace_temp_k=state.furnace_temp_k,
        iron_kg=actual_iron, oxygen_kg=actual_oxygen, slag_kg=slag,
        regolith_consumed_kg=regolith_used, power_consumed_kw=total_power,
        energy_consumed_kwh=sol_energy, joule_heating_kw=joule_kw,
        radiation_loss_kw=rad_kw, thermal_balance_kw=thermal_net,
        electrode_health=state.electrode_health, operational=True,
        events=list(events))


def run_simulation(sols: int = 365, cell_current_a: float = DEFAULT_CURRENT_A,
                   cell_voltage_v: float = DEFAULT_CELL_VOLTAGE_V,
                   feed_rate_kg: float = DEFAULT_FEED_RATE_KG_PER_SOL) -> list:
    """Run smelter for sols Mars sols."""
    state = OreSmelter(cell_current_a=cell_current_a, cell_voltage_v=cell_voltage_v,
                       regolith_feed_kg_per_sol=feed_rate_kg)
    return [tick(state) for _ in range(sols)]


if __name__ == "__main__":
    import json, sys
    sols = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    results = run_simulation(sols=sols)
    total_fe = sum(r.iron_kg for r in results)
    total_o2 = sum(r.oxygen_kg for r in results)
    print(f"Mars Ore Smelter -- {sols} sols")
    print(f"  Total iron: {total_fe:.1f} kg | O2: {total_o2:.1f} kg")
    print(f"  Electrode:  {results[-1].electrode_health:.2%}")
