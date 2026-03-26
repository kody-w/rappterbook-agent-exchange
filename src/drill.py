"""drill.py — Mars Subsurface Drill Simulation.

Models a rotary-percussive drill rig for colony subsurface operations.
Each tick = 1 sol of drilling activity.

Physics modelled
----------------
* **Penetration rate** — rotary-percussive drilling through layered Mars
  geology.  Rate drops with depth (overburden pressure, chip evacuation)
  and varies by rock type: regolith (loose), duricrust (ceite-cemented),
  basalt (hard igneous), ice-rich regolith (moderate).
* **Power consumption** — scales with rock hardness, depth, and drill RPM.
  Energy per metre increases superlinearly with depth due to chip removal
  and friction losses.  Based on InSight HP3 mole and ExoMars drill data.
* **Thermal management** — drill bit friction generates heat proportional
  to rock hardness × RPM.  Mars ambient cools the bit between bores.
  Overheating (>400°C) triggers automatic shutdown to prevent bit failure.
* **Bit wear** — abrasive wear modelled as a function of distance drilled
  and rock hardness.  Worn bits drill slower and generate more heat.
  Replacement at the colony restores bit condition.
* **Core samples** — extractable cylindrical cores for geological analysis.
  Mass depends on rock density and core dimensions.
* **Ice detection** — when drilling through ice-rich layers, water yield
  is calculated from ice concentration and core volume.

Reference hardware:
  - InSight HP3: 5 m target depth, self-hammering mole
  - ExoMars Rosalind Franklin: 2 m drill, rotary-percussive
  - Colony drill (this model): 50 m rated depth, 10 cm bore diameter

Conservation laws:
  - Energy consumed ≥ 0, never exceeds power budget
  - Depth monotonically increases during a bore (no negative drilling)
  - Bit wear ∈ [0.0, 1.0], monotonically increases
  - Core mass = density × π × r² × length (exact geometry)
  - Temperature bounded by physics (ambient floor, thermal runaway ceiling)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# -- Physical constants -------------------------------------------------------
MARS_AMBIENT_TEMP_C = -60.0
MARS_GRAVITY_M_S2 = 3.72

# Drill geometry
BORE_DIAMETER_M = 0.10
BORE_RADIUS_M = BORE_DIAMETER_M / 2.0
CORE_CROSS_SECTION_M2 = math.pi * BORE_RADIUS_M ** 2

# Rock types and properties
# (hardness 0-1, density kg/m³, thermal conductivity W/mK)
ROCK_REGOLITH = {"name": "regolith", "hardness": 0.15, "density": 1500.0, "conductivity": 0.04}
ROCK_DURICRUST = {"name": "duricrust", "hardness": 0.40, "density": 1800.0, "conductivity": 0.10}
ROCK_BASALT = {"name": "basalt", "hardness": 0.85, "density": 2900.0, "conductivity": 1.70}
ROCK_ICE_REGOLITH = {"name": "ice_regolith", "hardness": 0.30, "density": 1350.0, "conductivity": 0.50}

DEFAULT_LAYERS = [
    {"rock": ROCK_REGOLITH, "thickness_m": 3.0, "ice_fraction": 0.0},
    {"rock": ROCK_DURICRUST, "thickness_m": 2.0, "ice_fraction": 0.0},
    {"rock": ROCK_BASALT, "thickness_m": 15.0, "ice_fraction": 0.0},
    {"rock": ROCK_ICE_REGOLITH, "thickness_m": 10.0, "ice_fraction": 0.35},
    {"rock": ROCK_BASALT, "thickness_m": 20.0, "ice_fraction": 0.0},
]

# Drilling performance
BASE_PENETRATION_M_PER_SOL = 2.0  # in soft regolith, fresh bit, surface depth
DEPTH_PENALTY_FACTOR = 0.02       # penetration loss per metre of depth
MAX_RATED_DEPTH_M = 50.0

# Power
BASE_POWER_KWH_PER_M = 5.0       # energy to drill 1m of soft regolith
HARDNESS_POWER_MULTIPLIER = 4.0   # hard rock costs up to 4× more energy

# Thermal
FRICTION_HEAT_C_PER_M = 30.0     # base temp rise per metre drilled
HARDNESS_HEAT_MULTIPLIER = 3.0   # hard rock generates 3× more heat
COOLING_RATE_C_PER_SOL = 80.0    # passive cooling between bores
BIT_OVERHEAT_TEMP_C = 400.0      # emergency shutdown threshold
BIT_MAX_TEMP_C = 600.0           # absolute physical ceiling

# Bit wear
BIT_LIFE_METRES = 100.0          # metres of soft regolith before replacement
HARDNESS_WEAR_MULTIPLIER = 5.0   # hard rock wears bit 5× faster
WORN_BIT_SPEED_PENALTY = 0.6     # fully worn bit drills at 40% speed
WORN_BIT_HEAT_PENALTY = 2.0      # fully worn bit generates 2× heat
BIT_REPLACEMENT_RESTORE = 0.9    # replacement restores to 90% (not new)


# -- Data structures ----------------------------------------------------------

@dataclass
class DrillState:
    """Mutable state of a Mars subsurface drill rig."""
    sol: int = 0
    depth_m: float = 0.0
    bit_wear: float = 0.0         # 0.0 = new, 1.0 = destroyed
    bit_temp_c: float = MARS_AMBIENT_TEMP_C
    total_energy_kwh: float = 0.0
    total_cores_extracted: int = 0
    total_core_mass_kg: float = 0.0
    total_ice_detected_kg: float = 0.0
    operational: bool = True
    emergency_shutdowns: int = 0

    def __post_init__(self) -> None:
        self.depth_m = max(0.0, min(self.depth_m, MAX_RATED_DEPTH_M))
        self.bit_wear = max(0.0, min(1.0, self.bit_wear))
        self.bit_temp_c = max(MARS_AMBIENT_TEMP_C, min(BIT_MAX_TEMP_C, self.bit_temp_c))


@dataclass
class DrillSol:
    """Record of one sol of drilling activity."""
    sol: int = 0
    metres_drilled: float = 0.0
    power_consumed_kwh: float = 0.0
    peak_bit_temp_c: float = MARS_AMBIENT_TEMP_C
    core_mass_kg: float = 0.0
    ice_detected_kg: float = 0.0
    rock_type: str = "unknown"
    bit_wear_delta: float = 0.0
    emergency_shutdown: bool = False


# -- Pure functions -----------------------------------------------------------

def get_layer_at_depth(depth_m: float, layers: list[dict] | None = None) -> dict:
    """Return the geological layer at a given depth.

    Walks the layer stack top-down.  If depth exceeds all layers,
    returns the deepest layer (infinite basement assumption).
    """
    if layers is None:
        layers = DEFAULT_LAYERS
    if not layers:
        return {"rock": ROCK_BASALT, "thickness_m": 1000.0, "ice_fraction": 0.0}

    cumulative = 0.0
    for layer in layers:
        cumulative += layer["thickness_m"]
        if depth_m < cumulative:
            return layer
    return layers[-1]


def penetration_rate(
    depth_m: float,
    hardness: float,
    bit_wear: float,
) -> float:
    """Calculate drilling penetration rate in metres/sol.

    Decreases with:
      - Depth (chip evacuation, overburden friction)
      - Rock hardness (harder rock = slower)
      - Bit wear (worn bit = slower)

    Returns a non-negative rate, zero if bit is destroyed.
    """
    if bit_wear >= 1.0:
        return 0.0

    depth_factor = max(0.05, 1.0 - DEPTH_PENALTY_FACTOR * depth_m)
    hardness_factor = max(0.1, 1.0 - hardness * 0.8)
    wear_factor = 1.0 - bit_wear * WORN_BIT_SPEED_PENALTY

    rate = BASE_PENETRATION_M_PER_SOL * depth_factor * hardness_factor * wear_factor
    return max(0.0, rate)


def power_per_metre(depth_m: float, hardness: float) -> float:
    """Energy cost to drill one metre at given depth and hardness.

    Scales superlinearly with depth (longer chip evacuation path)
    and linearly with hardness.
    """
    depth_cost = 1.0 + 0.01 * depth_m  # 1% increase per metre depth
    hardness_cost = 1.0 + hardness * HARDNESS_POWER_MULTIPLIER
    return BASE_POWER_KWH_PER_M * depth_cost * hardness_cost


def bit_heat_rise(metres: float, hardness: float, bit_wear: float) -> float:
    """Temperature rise from drilling a given distance.

    Hard rock and worn bits generate more heat.
    """
    if metres <= 0:
        return 0.0
    hardness_factor = 1.0 + hardness * HARDNESS_HEAT_MULTIPLIER
    wear_factor = 1.0 + bit_wear * WORN_BIT_HEAT_PENALTY
    return FRICTION_HEAT_C_PER_M * metres * hardness_factor * wear_factor


def cool_bit(current_temp_c: float) -> float:
    """Cool the drill bit toward ambient over one sol of inactivity.

    Returns the new temperature after passive cooling.
    Never drops below Mars ambient.
    """
    if current_temp_c <= MARS_AMBIENT_TEMP_C:
        return MARS_AMBIENT_TEMP_C
    new_temp = current_temp_c - COOLING_RATE_C_PER_SOL
    return max(MARS_AMBIENT_TEMP_C, new_temp)


def bit_wear_per_metre(hardness: float) -> float:
    """Bit wear fraction per metre drilled.

    Harder rock wears the bit faster.
    """
    base = 1.0 / BIT_LIFE_METRES
    return base * (1.0 + hardness * HARDNESS_WEAR_MULTIPLIER)


def core_mass_kg(metres: float, density: float) -> float:
    """Mass of a cylindrical core sample.

    mass = density × π × r² × length
    """
    if metres <= 0 or density <= 0:
        return 0.0
    return density * CORE_CROSS_SECTION_M2 * metres


def ice_yield_kg(metres: float, density: float, ice_fraction: float) -> float:
    """Water ice extracted from an ice-bearing layer.

    Based on the mass of material drilled and its ice content.
    """
    if metres <= 0 or density <= 0 or ice_fraction <= 0:
        return 0.0
    total_mass = density * CORE_CROSS_SECTION_M2 * metres
    return total_mass * min(1.0, max(0.0, ice_fraction))


def replace_bit(state: DrillState) -> None:
    """Replace the drill bit, restoring condition.

    Not a full reset — replacement bits aren't factory-new on Mars.
    """
    state.bit_wear = max(0.0, state.bit_wear - BIT_REPLACEMENT_RESTORE)
    state.bit_temp_c = MARS_AMBIENT_TEMP_C
    state.operational = True


# -- Tick function ------------------------------------------------------------

def tick_drill(
    state: DrillState,
    power_budget_kwh: float,
    layers: list[dict] | None = None,
) -> DrillSol:
    """Advance the drill simulation by one sol.

    Args:
        state: mutable drill state (modified in place)
        power_budget_kwh: energy allocated to drilling this sol
        layers: geological layer stack (default Mars profile)

    Returns:
        DrillSol record of this sol's activity.

    Conservation guarantees:
      - power_consumed ≤ power_budget
      - depth only increases
      - bit_wear only increases
      - core_mass = density × cross_section × metres_drilled
    """
    sol_record = DrillSol(sol=state.sol)
    state.sol += 1

    # Non-operational or no power: cool only
    if not state.operational or power_budget_kwh <= 0 or state.bit_wear >= 1.0:
        state.bit_temp_c = cool_bit(state.bit_temp_c)
        if state.bit_wear >= 1.0:
            state.operational = False
        return sol_record

    # Determine current geology
    layer = get_layer_at_depth(state.depth_m, layers)
    rock = layer["rock"]
    hardness = rock["hardness"]
    density = rock["density"]
    ice_frac = layer.get("ice_fraction", 0.0)
    sol_record.rock_type = rock["name"]

    # Calculate how far we can drill this sol
    rate = penetration_rate(state.depth_m, hardness, state.bit_wear)
    if rate <= 0:
        state.bit_temp_c = cool_bit(state.bit_temp_c)
        return sol_record

    cost_per_m = power_per_metre(state.depth_m, hardness)
    max_metres_by_power = power_budget_kwh / cost_per_m if cost_per_m > 0 else 0.0
    max_metres_by_rate = rate
    max_metres_by_depth = max(0.0, MAX_RATED_DEPTH_M - state.depth_m)

    metres = min(max_metres_by_power, max_metres_by_rate, max_metres_by_depth)
    metres = max(0.0, metres)

    if metres <= 0:
        state.bit_temp_c = cool_bit(state.bit_temp_c)
        return sol_record

    # Thermal check — will this overheat the bit?
    heat = bit_heat_rise(metres, hardness, state.bit_wear)
    projected_temp = state.bit_temp_c + heat
    if projected_temp >= BIT_OVERHEAT_TEMP_C:
        # Reduce drilling to stay under overheat threshold
        safe_heat = max(0.0, BIT_OVERHEAT_TEMP_C - state.bit_temp_c - 1.0)
        heat_per_m = bit_heat_rise(1.0, hardness, state.bit_wear)
        if heat_per_m > 0:
            metres = min(metres, safe_heat / heat_per_m)
        else:
            metres = 0.0
        metres = max(0.0, metres)

    if metres <= 0:
        state.bit_temp_c = cool_bit(state.bit_temp_c)
        sol_record.emergency_shutdown = True
        state.emergency_shutdowns += 1
        return sol_record

    # Commit the drilling
    energy_used = metres * cost_per_m
    heat_generated = bit_heat_rise(metres, hardness, state.bit_wear)
    wear_delta = bit_wear_per_metre(hardness) * metres

    # Update state
    state.depth_m = min(MAX_RATED_DEPTH_M, state.depth_m + metres)
    state.bit_temp_c = min(BIT_MAX_TEMP_C, state.bit_temp_c + heat_generated)
    state.bit_wear = min(1.0, state.bit_wear + wear_delta)
    state.total_energy_kwh += energy_used
    state.total_cores_extracted += 1
    c_mass = core_mass_kg(metres, density)
    state.total_core_mass_kg += c_mass
    ice = ice_yield_kg(metres, density, ice_frac)
    state.total_ice_detected_kg += ice

    # Post-drill cooling
    state.bit_temp_c = cool_bit(state.bit_temp_c)

    # Bit death check
    if state.bit_wear >= 1.0:
        state.operational = False

    # Populate sol record
    sol_record.metres_drilled = metres
    sol_record.power_consumed_kwh = energy_used
    sol_record.peak_bit_temp_c = state.bit_temp_c + heat_generated  # peak was before cooling
    sol_record.core_mass_kg = c_mass
    sol_record.ice_detected_kg = ice
    sol_record.bit_wear_delta = wear_delta

    return sol_record


def create_drill() -> DrillState:
    """Factory function for a fresh drill rig."""
    return DrillState()
