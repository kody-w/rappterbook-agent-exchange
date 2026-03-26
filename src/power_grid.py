"""
power_grid.py - Mars colony power distribution and load management.

The nervous system of the colony. Every organ (O2 generator, water
mining, greenhouse, regolith processor, comms relay, thermal control,
fuel plant) draws power from a shared grid. This module allocates
available generation to competing loads via priority scheduling, tracks
brownout events, and models grid-level battery buffering.

Physical references:
  - ISS power: 8 x 31 kW solar arrays, 60-120 V DC bus, 6 Li-ion batteries
  - Mars DRA 5.0: 40 kW nuclear fission (Kilopower/KRUSTY), ~100 kW peak
  - Battery round-trip efficiency: ~90% (Li-ion, ISS heritage)
  - Bus voltage: 120 V DC (DRA 5.0 baseline)

One tick = one sol. Energy in kWh.
"""
from __future__ import annotations
from dataclasses import dataclass, field

NUCLEAR_BASELINE_KWH_SOL = 960.0
BUS_VOLTAGE_V = 120.0
BATTERY_ROUND_TRIP_EFF = 0.90
BATTERY_SELF_DISCHARGE_PER_SOL = 0.001
GRID_FREQ_NOMINAL_HZ = 400.0
GRID_FREQ_MIN_HZ = 350.0
FREQ_DROP_PER_OVERLOAD_FRACTION = 50.0
SOL_HOURS = 24.66

PRIORITY_LIFE_SUPPORT = 0
PRIORITY_THERMAL = 1
PRIORITY_WATER = 2
PRIORITY_FOOD = 3
PRIORITY_CONSTRUCTION = 4
PRIORITY_SCIENCE = 5


@dataclass
class PowerLoad:
    """A single power consumer on the grid."""
    name: str
    priority: int
    requested_kwh: float
    allocated_kwh: float = 0.0

    def __post_init__(self) -> None:
        self.priority = max(0, min(5, self.priority))
        self.requested_kwh = max(0.0, self.requested_kwh)
        self.allocated_kwh = max(0.0, self.allocated_kwh)

    def shortfall(self) -> float:
        return max(0.0, self.requested_kwh - self.allocated_kwh)

    def fulfillment_ratio(self) -> float:
        if self.requested_kwh <= 0:
            return 1.0
        return min(1.0, self.allocated_kwh / self.requested_kwh)


@dataclass
class GridBattery:
    """Colony-level battery bank."""
    capacity_kwh: float = 500.0
    charge_kwh: float = 250.0

    def __post_init__(self) -> None:
        self.capacity_kwh = max(0.0, self.capacity_kwh)
        self.charge_kwh = max(0.0, min(self.charge_kwh, self.capacity_kwh))

    def headroom(self) -> float:
        return max(0.0, self.capacity_kwh - self.charge_kwh)

    def charge(self, kwh: float) -> float:
        usable = kwh * BATTERY_ROUND_TRIP_EFF
        stored = min(usable, self.headroom())
        self.charge_kwh += stored
        return round(stored, 6)

    def discharge(self, kwh: float) -> float:
        available = self.charge_kwh * BATTERY_ROUND_TRIP_EFF
        delivered = min(kwh, available)
        if BATTERY_ROUND_TRIP_EFF > 0:
            self.charge_kwh -= delivered / BATTERY_ROUND_TRIP_EFF
        self.charge_kwh = max(0.0, self.charge_kwh)
        return round(delivered, 6)

    def self_discharge(self) -> float:
        lost = self.charge_kwh * BATTERY_SELF_DISCHARGE_PER_SOL
        self.charge_kwh -= lost
        self.charge_kwh = max(0.0, self.charge_kwh)
        return round(lost, 6)


@dataclass
class GridState:
    """Complete state of the colony power grid."""
    sol: int = 0
    battery: GridBattery = field(default_factory=GridBattery)
    total_generated_kwh: float = 0.0
    total_consumed_kwh: float = 0.0
    total_curtailed_kwh: float = 0.0
    brownout_sols: int = 0
    blackout_sols: int = 0
    peak_demand_kwh: float = 0.0


def allocate_power(loads: list[PowerLoad], available_kwh: float) -> float:
    """Allocate power to loads by priority. Returns surplus."""
    sorted_loads = sorted(loads, key=lambda ld: (ld.priority, ld.name))
    remaining = max(0.0, available_kwh)
    for load in sorted_loads:
        grant = min(load.requested_kwh, remaining)
        load.allocated_kwh = round(grant, 6)
        remaining -= grant
        remaining = max(0.0, remaining)
    return round(remaining, 6)


def grid_frequency(total_demand_kwh: float, total_supply_kwh: float) -> float:
    """Grid frequency as stability proxy."""
    if total_supply_kwh <= 0:
        return 0.0
    if total_demand_kwh <= total_supply_kwh:
        return GRID_FREQ_NOMINAL_HZ
    overload_frac = (total_demand_kwh - total_supply_kwh) / total_supply_kwh
    freq = GRID_FREQ_NOMINAL_HZ - overload_frac * FREQ_DROP_PER_OVERLOAD_FRACTION
    return max(0.0, round(freq, 4))


def total_demand(loads: list[PowerLoad]) -> float:
    return round(sum(ld.requested_kwh for ld in loads), 6)


def total_allocated(loads: list[PowerLoad]) -> float:
    return round(sum(ld.allocated_kwh for ld in loads), 6)


def total_curtailed(loads: list[PowerLoad]) -> float:
    return round(sum(ld.shortfall() for ld in loads), 6)


def tick_grid(
    state: GridState,
    solar_kwh: float,
    nuclear_kwh: float = NUCLEAR_BASELINE_KWH_SOL,
    loads: list[PowerLoad] | None = None,
    battery_discharge_kwh: float = 0.0,
) -> dict[str, float]:
    """Advance the power grid by one sol."""
    state.sol += 1
    if loads is None:
        loads = []

    generation = max(0.0, solar_kwh) + max(0.0, nuclear_kwh)
    batt_loss = state.battery.self_discharge()
    batt_available = min(battery_discharge_kwh, state.battery.charge_kwh * BATTERY_ROUND_TRIP_EFF)
    supply = generation + batt_available
    demand = total_demand(loads)
    state.peak_demand_kwh = max(state.peak_demand_kwh, demand)

    surplus = allocate_power(loads, supply)
    consumed = total_allocated(loads)

    from_battery = max(0.0, consumed - generation)
    actual_discharge = state.battery.discharge(from_battery) if from_battery > 0 else 0.0
    gen_surplus = max(0.0, generation - consumed)
    charged = state.battery.charge(gen_surplus) if gen_surplus > 0 else 0.0

    curtailed = total_curtailed(loads)
    if curtailed > 0:
        state.brownout_sols += 1
    if consumed == 0 and demand > 0:
        state.blackout_sols += 1

    freq = grid_frequency(demand, supply)
    state.total_generated_kwh += generation
    state.total_consumed_kwh += consumed
    state.total_curtailed_kwh += curtailed

    return {
        "sol": float(state.sol),
        "generation_kwh": round(generation, 6),
        "solar_kwh": round(max(0.0, solar_kwh), 6),
        "nuclear_kwh": round(max(0.0, nuclear_kwh), 6),
        "demand_kwh": round(demand, 6),
        "consumed_kwh": round(consumed, 6),
        "curtailed_kwh": round(curtailed, 6),
        "surplus_kwh": round(surplus, 6),
        "battery_charged_kwh": round(charged, 6),
        "battery_discharged_kwh": round(actual_discharge, 6),
        "battery_level_kwh": round(state.battery.charge_kwh, 6),
        "battery_loss_kwh": round(batt_loss, 6),
        "grid_frequency_hz": round(freq, 4),
        "brownout": curtailed > 0,
        "blackout": consumed == 0 and demand > 0,
    }


def colony_loads_default() -> list[PowerLoad]:
    """Standard colony load profile for a crew of six (DRA 5.0)."""
    return [
        PowerLoad("life_support",       PRIORITY_LIFE_SUPPORT, 120.0),
        PowerLoad("thermal_control",    PRIORITY_THERMAL,      80.0),
        PowerLoad("water_mining",       PRIORITY_WATER,        60.0),
        PowerLoad("greenhouse",         PRIORITY_FOOD,         40.0),
        PowerLoad("o2_generator",       PRIORITY_LIFE_SUPPORT, 100.0),
        PowerLoad("regolith_processor", PRIORITY_CONSTRUCTION, 50.0),
        PowerLoad("comm_relay",         PRIORITY_SCIENCE,      20.0),
        PowerLoad("fuel_plant",         PRIORITY_CONSTRUCTION, 80.0),
        PowerLoad("science_lab",        PRIORITY_SCIENCE,      30.0),
    ]


def create_grid(strategy: str = "balanced") -> GridState:
    """Create a grid pre-configured for a colony strategy."""
    configs = {
        "conservative": GridState(battery=GridBattery(capacity_kwh=800.0, charge_kwh=600.0)),
        "balanced": GridState(battery=GridBattery(capacity_kwh=500.0, charge_kwh=250.0)),
        "aggressive": GridState(battery=GridBattery(capacity_kwh=300.0, charge_kwh=100.0)),
    }
    return configs.get(strategy, configs["balanced"])
