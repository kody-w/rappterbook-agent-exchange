"""
power_grid.py - Mars colony power distribution and load management.

The energy backbone: takes total generation (solar + nuclear + methane)
and allocates kWh to subsystems by priority. Handles brownouts when
demand exceeds supply, manages battery storage, and tracks grid health.

Every other module takes available_power_kwh as input.
This module decides what that number is.

Physical references:
  - ISS power: 8 solar arrays, ~120 kW peak, ~84 kW average
  - Mars colony target: 100-200 kW continuous for 100 colonists
  - Kilopower reactor: 10 kWe each (NASA design, 4 reactors = 40 kW)
  - Battery: Li-ion, 90% round-trip efficiency, 0.1%/sol self-discharge
  - Load priority (NASA ECLSS reference):
      1. Life support (O2, CO2 scrubbing, pressure) - CRITICAL
      2. Thermal control (heating, cooling) - CRITICAL
      3. Water processing (WRS, mining) - HIGH
      4. Communications (Earth relay, intra-colony) - HIGH
      5. Food production (greenhouse lighting, hydroponics) - MEDIUM
      6. Medical systems - MEDIUM
      7. ISRU (regolith, fuel, O2 generation) - LOW
      8. Science/exploration - LOW
      9. Comfort (lighting, entertainment) - LOWEST

One tick = one sol. Energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

BATTERY_ROUND_TRIP_EFF = 0.90
BATTERY_SELF_DISCHARGE_PER_SOL = 0.001
BATTERY_MAX_CHARGE_RATE = 0.25       # max 25% of capacity per sol (C/4)
BATTERY_MAX_DISCHARGE_RATE = 0.50    # max 50% of capacity per sol (C/2)
BATTERY_DEPTH_OF_DISCHARGE = 0.80    # only use 80% of rated capacity
BATTERY_CYCLE_DEGRADATION = 0.00005  # capacity loss per full cycle

GRID_LOSS_FRACTION = 0.03            # 3% transmission/conversion loss
NUCLEAR_BASELINE_KWH = 100.0         # per Kilopower reactor per sol
NUCLEAR_CAPACITY_FACTOR = 0.95       # 95% uptime

GRID_WEAR_PER_SOL = 0.0002           # inverters, wiring, contactors
MIN_GRID_HEALTH = 0.30               # grid still works at 30%
MAINTENANCE_RESTORE = 0.03           # crew maintenance per sol


# ---------------------------------------------------------------------------
# Priority system
# ---------------------------------------------------------------------------

PRIORITY_CRITICAL = 0    # life support, thermal - never shed
PRIORITY_HIGH = 1        # water, comms
PRIORITY_MEDIUM = 2      # food, medical
PRIORITY_LOW = 3         # ISRU, science
PRIORITY_LOWEST = 4      # comfort

PRIORITY_NAMES = {
    PRIORITY_CRITICAL: "critical",
    PRIORITY_HIGH: "high",
    PRIORITY_MEDIUM: "medium",
    PRIORITY_LOW: "low",
    PRIORITY_LOWEST: "lowest",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PowerLoad:
    """A subsystem requesting power from the grid.

    name: subsystem identifier (e.g. "life_support", "greenhouse")
    demand_kwh: requested energy for this sol
    priority: load priority (0=critical, 4=lowest)
    min_fraction: minimum fraction of demand needed to function (0-1)
    """
    name: str
    demand_kwh: float
    priority: int = PRIORITY_MEDIUM
    min_fraction: float = 0.0

    def __post_init__(self) -> None:
        self.demand_kwh = max(0.0, self.demand_kwh)
        self.priority = max(PRIORITY_CRITICAL, min(PRIORITY_LOWEST, self.priority))
        self.min_fraction = max(0.0, min(1.0, self.min_fraction))


@dataclass
class BatteryBank:
    """Colony battery storage system.

    capacity_kwh: rated capacity (before depth-of-discharge limit)
    charge_kwh: current stored energy
    cycles: cumulative full-equivalent charge cycles
    degradation: cumulative capacity loss from cycling [0, 1]
    """
    capacity_kwh: float = 500.0
    charge_kwh: float = 250.0
    cycles: float = 0.0
    degradation: float = 0.0

    def __post_init__(self) -> None:
        self.capacity_kwh = max(1.0, self.capacity_kwh)
        self.degradation = max(0.0, min(0.99, self.degradation))
        usable = self.usable_capacity
        self.charge_kwh = max(0.0, min(usable, self.charge_kwh))
        self.cycles = max(0.0, self.cycles)

    @property
    def usable_capacity(self) -> float:
        """Actual usable capacity after DoD and degradation."""
        return self.capacity_kwh * BATTERY_DEPTH_OF_DISCHARGE * (1.0 - self.degradation)

    @property
    def state_of_charge(self) -> float:
        """Fraction of usable capacity currently stored [0, 1]."""
        usable = self.usable_capacity
        if usable <= 0:
            return 0.0
        return min(1.0, self.charge_kwh / usable)


@dataclass
class GridState:
    """Power grid system state.

    health: grid infrastructure condition [0, 1]
    total_generated_kwh: lifetime generation
    total_consumed_kwh: lifetime consumption
    total_curtailed_kwh: lifetime wasted (generation > demand + storage)
    total_shed_kwh: lifetime load shed (demand > supply)
    brownout_sols: number of sols with load shedding
    """
    health: float = 1.0
    total_generated_kwh: float = 0.0
    total_consumed_kwh: float = 0.0
    total_curtailed_kwh: float = 0.0
    total_shed_kwh: float = 0.0
    brownout_sols: int = 0

    def __post_init__(self) -> None:
        self.health = max(MIN_GRID_HEALTH, min(1.0, self.health))
        self.total_generated_kwh = max(0.0, self.total_generated_kwh)
        self.total_consumed_kwh = max(0.0, self.total_consumed_kwh)
        self.total_curtailed_kwh = max(0.0, self.total_curtailed_kwh)
        self.total_shed_kwh = max(0.0, self.total_shed_kwh)
        self.brownout_sols = max(0, self.brownout_sols)


@dataclass
class GridSol:
    """Output record for one sol of grid operation.

    generated_kwh: total power generated this sol
    consumed_kwh: total power delivered to loads
    stored_kwh: net energy into battery (negative = discharged)
    curtailed_kwh: excess generation wasted
    shed_kwh: demand that could not be met
    brownout: True if any load was shed
    allocations: dict of subsystem name -> kwh delivered
    """
    generated_kwh: float = 0.0
    consumed_kwh: float = 0.0
    stored_kwh: float = 0.0
    curtailed_kwh: float = 0.0
    shed_kwh: float = 0.0
    brownout: bool = False
    allocations: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def grid_efficiency(health: float) -> float:
    """Grid delivery efficiency given infrastructure health.

    Healthy grid: 97% (3% loss). Degraded grid: up to 15% loss.
    """
    h = max(MIN_GRID_HEALTH, min(1.0, health))
    base_loss = GRID_LOSS_FRACTION
    degraded_loss = 0.15
    loss = degraded_loss - h * (degraded_loss - base_loss)
    return 1.0 - loss


def battery_tick(battery: BatteryBank, net_kwh: float) -> float:
    """Charge or discharge battery. Returns actual energy exchanged.

    net_kwh > 0: charging (excess generation)
    net_kwh < 0: discharging (deficit)

    Returns positive for charge, negative for discharge.
    Respects rate limits and round-trip efficiency.
    """
    # Self-discharge first
    battery.charge_kwh *= (1.0 - BATTERY_SELF_DISCHARGE_PER_SOL)
    battery.charge_kwh = max(0.0, battery.charge_kwh)

    usable = battery.usable_capacity
    if usable <= 0:
        return 0.0

    if net_kwh > 0:
        # Charging
        max_charge = battery.capacity_kwh * BATTERY_MAX_CHARGE_RATE
        charge_input = min(net_kwh, max_charge)
        stored = charge_input * math.sqrt(BATTERY_ROUND_TRIP_EFF)  # half the loss on charge
        headroom = usable - battery.charge_kwh
        stored = min(stored, max(0.0, headroom))
        battery.charge_kwh += stored
        # Track cycling
        cycle_fraction = stored / usable if usable > 0 else 0
        battery.cycles += cycle_fraction
        battery.degradation = min(0.99, battery.degradation + cycle_fraction * BATTERY_CYCLE_DEGRADATION)
        return stored
    elif net_kwh < 0:
        # Discharging
        max_discharge = battery.capacity_kwh * BATTERY_MAX_DISCHARGE_RATE
        needed = min(abs(net_kwh), max_discharge)
        available = min(needed, battery.charge_kwh)
        delivered = available * math.sqrt(BATTERY_ROUND_TRIP_EFF)  # half the loss on discharge
        battery.charge_kwh -= available
        battery.charge_kwh = max(0.0, battery.charge_kwh)
        cycle_fraction = available / usable if usable > 0 else 0
        battery.cycles += cycle_fraction
        battery.degradation = min(0.99, battery.degradation + cycle_fraction * BATTERY_CYCLE_DEGRADATION)
        return -delivered
    return 0.0


def allocate_power(
    available_kwh: float,
    loads: List[PowerLoad],
) -> Tuple[Dict[str, float], float]:
    """Allocate power to loads by priority.

    Returns (allocations dict, total_shed_kwh).
    Critical loads get power first. Within same priority, proportional.
    """
    allocations: Dict[str, float] = {}
    remaining = available_kwh
    total_shed = 0.0

    # Group loads by priority
    by_priority: Dict[int, List[PowerLoad]] = {}
    for load in loads:
        by_priority.setdefault(load.priority, []).append(load)

    for priority in sorted(by_priority.keys()):
        group = by_priority[priority]
        group_demand = sum(l.demand_kwh for l in group)

        if group_demand <= 0:
            for load in group:
                allocations[load.name] = 0.0
            continue

        if group_demand <= remaining:
            # Enough power for entire priority group
            for load in group:
                allocations[load.name] = load.demand_kwh
            remaining -= group_demand
        else:
            # Proportional allocation within priority group
            ratio = remaining / group_demand if group_demand > 0 else 0
            for load in group:
                allocated = load.demand_kwh * ratio
                # Ensure minimum fraction if possible
                minimum = load.demand_kwh * load.min_fraction
                allocated = max(allocated, min(minimum, remaining))
                allocated = min(allocated, remaining)
                allocations[load.name] = allocated
                shed = load.demand_kwh - allocated
                total_shed += shed
                remaining -= allocated
            remaining = max(0.0, remaining)

    return allocations, total_shed


def tick_grid(
    grid: GridState,
    battery: BatteryBank,
    generation_kwh: float,
    loads: List[PowerLoad],
    crew_maintenance: bool = False,
) -> GridSol:
    """Run one sol of power grid operation.

    Steps:
      1. Grid degradation + maintenance
      2. Apply grid efficiency to generation
      3. Allocate power to loads by priority
      4. Battery charge/discharge for surplus/deficit
      5. Curtailment or load shedding as needed
    """
    result = GridSol()

    # Step 1: Degradation + maintenance
    grid.health = max(MIN_GRID_HEALTH, grid.health - GRID_WEAR_PER_SOL)
    if crew_maintenance:
        grid.health = min(1.0, grid.health + MAINTENANCE_RESTORE)

    # Step 2: Apply grid efficiency
    eff = grid_efficiency(grid.health)
    usable_gen = max(0.0, generation_kwh) * eff
    result.generated_kwh = usable_gen
    grid.total_generated_kwh += usable_gen

    # Step 3: Allocate to loads
    total_demand = sum(l.demand_kwh for l in loads)
    allocations, shed_from_alloc = allocate_power(usable_gen, loads)

    consumed = sum(allocations.values())
    result.allocations = allocations
    result.consumed_kwh = consumed

    # Step 4: Battery
    surplus = usable_gen - consumed
    if surplus > 0:
        # Charge battery with excess
        stored = battery_tick(battery, surplus)
        result.stored_kwh = stored
        curtailed = surplus - abs(stored)
        result.curtailed_kwh = max(0.0, curtailed)
        grid.total_curtailed_kwh += result.curtailed_kwh
    elif shed_from_alloc > 0:
        # Try to cover deficit from battery
        discharged = battery_tick(battery, -shed_from_alloc)
        recovered = abs(discharged)
        if recovered > 0:
            # Re-allocate recovered power to shed loads (by priority)
            shed_loads = [l for l in loads if allocations.get(l.name, 0) < l.demand_kwh]
            shed_loads.sort(key=lambda l: l.priority)
            remaining_recovery = recovered
            for load in shed_loads:
                deficit = load.demand_kwh - allocations[load.name]
                give = min(deficit, remaining_recovery)
                allocations[load.name] += give
                remaining_recovery -= give
                consumed += give
            result.consumed_kwh = consumed
            result.allocations = allocations
            result.stored_kwh = discharged
            shed_from_alloc = max(0.0, total_demand - consumed)
    else:
        # No surplus, no deficit — just tick battery for self-discharge
        battery_tick(battery, 0.0)

    result.shed_kwh = shed_from_alloc
    result.brownout = shed_from_alloc > 0
    grid.total_consumed_kwh += consumed
    grid.total_shed_kwh += shed_from_alloc
    if result.brownout:
        grid.brownout_sols += 1

    return result


def grid_reliability(grid: GridState) -> float:
    """Fraction of sols without brownout. Returns [0, 1]."""
    total_sols = grid.brownout_sols + max(1, int(
        (grid.total_consumed_kwh + grid.total_shed_kwh) /
        max(1.0, grid.total_consumed_kwh / max(1, grid.brownout_sols + 1))
    )) if grid.total_consumed_kwh > 0 else 1
    if total_sols <= 0:
        return 1.0
    return 1.0 - (grid.brownout_sols / total_sols)
