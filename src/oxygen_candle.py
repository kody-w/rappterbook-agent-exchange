"""oxygen_candle.py — Mars Emergency Chemical Oxygen Generator.

When the power dies — dust storm buries the solar panels, reactor
SCRAMs, battery bank drains to zero — the colony has minutes before
CO₂ narcosis sets in.  The main O₂ generator (SOEC electrolysis)
needs 300 W continuous.  The life-support electrolyser needs 5 kWh/kg.
No power, no oxygen.

Oxygen candles solve this.  A solid chemical charge (sodium chlorate +
iron powder) is ignited with a percussive primer.  The exothermic
decomposition produces pure O₂ with zero electrical input.  Real
technology: every submarine on Earth carries them.  The ISS stores
them as backup.  The Russian Vika (ОКД) generators have logged 20+
years of service on Mir and ISS.

Chemistry
---------
  2 NaClO₃ → 2 NaCl + 3 O₂     (ΔH = −45 kJ/mol NaClO₃)

  Iron powder (5–10 wt%) acts as fuel to sustain the reaction once
  ignited.  No external heat or electricity required after ignition.

Physics modelled
----------------
* **Stoichiometric O₂ yield** — 0.45 kg O₂ per kg NaClO₃.
  With iron binder and packaging: ~0.33 kg O₂ per kg candle.

* **Burn rate** — 6.5 g O₂/min for a standard 1 kg candle.
  Total burn time: ~50 min per candle.

* **Heat output** — 45 kJ/mol NaClO₃ → ~1.9 MJ per kg candle.
  Surface temperature reaches 500–600 °C.  The candle housing
  must be insulated to prevent hab fires.

* **Shelf life** — NaClO₃ is hygroscopic.  On Mars (near-zero
  humidity) shelf life is excellent: 0.05% degradation per sol,
  decades of usable storage.

* **Ignition** — Percussive primer (like a shotgun shell).
  Ignition energy ~50 J.  Spring-loaded striker, no electricity.

* **CO₂ and moisture** — The reaction produces trace moisture
  (~5% by mass).  NaCl residue is inert solid waste.

* **Crew O₂ demand** — 0.84 kg/person/sol (NASA ECLSS).
  One 1 kg candle provides ~9.4 hours of O₂ for one person.

* **Candle inventory** — Colony tracks total candles, ignited
  candles, spent candles, and O₂ delivered.

Conservation laws
-----------------
- Mass: NaClO₃ consumed + Fe consumed = NaCl produced + O₂ produced + H₂O trace
  (stoichiometric, exact)
- O₂ produced ≤ stoichiometric limit (cannot exceed chemical yield)
- Heat produced = ΔH × moles reacted (first law of thermodynamics)
- Candle inventory: total = ready + burning + spent (exact, integer)
- Shelf degradation monotonically increases, yield monotonically decreases
- Burn progress in [0, 1], never reverses
- O₂ rate ≥ 0 always

Reference:
  - NaClO₃ molar mass: 106.44 g/mol
  - NaCl molar mass: 58.44 g/mol
  - O₂ molar mass: 32.00 g/mol
  - Fe molar mass: 55.85 g/mol
  - Russian Vika (ОКД) system: ~0.6 L O₂/min per cartridge
  - ISS SFOG (Solid Fuel Oxygen Generator): 600 L O₂ per canister
  - US Navy oxygen candle: ~115 L O₂ (~165 g) per candle
  - NASA ECLSS O₂ consumption: 0.84 kg/person/sol

One tick = one minute.  Mass in kg, temperature in K, energy in kJ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any


# -- Chemical constants -------------------------------------------------------

NACLO3_MOLAR_MASS_G = 106.44       # g/mol sodium chlorate
NACL_MOLAR_MASS_G = 58.44          # g/mol sodium chloride
O2_MOLAR_MASS_G = 32.00            # g/mol molecular oxygen
FE_MOLAR_MASS_G = 55.845           # g/mol iron

# 2 NaClO₃ → 2 NaCl + 3 O₂
# Per mol NaClO₃: 1.5 mol O₂ = 48 g O₂ per 106.44 g NaClO₃
O2_YIELD_PER_KG_NACLO3 = (1.5 * O2_MOLAR_MASS_G) / NACLO3_MOLAR_MASS_G  # ~0.4511
NACL_YIELD_PER_KG_NACLO3 = NACL_MOLAR_MASS_G / NACLO3_MOLAR_MASS_G      # ~0.5489

# Enthalpy of decomposition (exothermic)
DELTA_H_KJ_PER_MOL = 45.0          # kJ per mol NaClO₃ decomposed

# -- Candle composition -------------------------------------------------------

NACLO3_MASS_FRACTION = 0.85         # 85% sodium chlorate by mass
FE_POWDER_MASS_FRACTION = 0.08      # 8% iron powder fuel/binder
PACKAGING_MASS_FRACTION = 0.07      # 7% steel casing + insulation
# Sum = 1.00

DEFAULT_CANDLE_MASS_KG = 1.0        # standard candle mass

# Effective O₂ yield per kg of complete candle
O2_PER_KG_CANDLE = O2_YIELD_PER_KG_NACLO3 * NACLO3_MASS_FRACTION  # ~0.383

# -- Burn parameters ----------------------------------------------------------

# Standard burn rate: 6.5 g O₂/min for a 1 kg candle
BURN_RATE_KG_O2_PER_MIN = 0.0065
BURN_DURATION_MIN = O2_PER_KG_CANDLE * DEFAULT_CANDLE_MASS_KG / BURN_RATE_KG_O2_PER_MIN

# Heat generation
HEAT_PER_KG_CANDLE_KJ = (
    DELTA_H_KJ_PER_MOL
    * NACLO3_MASS_FRACTION
    * 1000.0  # kg to g
    / NACLO3_MOLAR_MASS_G
)  # ~359 kJ/kg candle

# Surface temperature of active candle
CANDLE_SURFACE_TEMP_K = 823.0       # ~550 °C during burn
CANDLE_AMBIENT_TEMP_K = 293.0       # hab interior ~20 °C
INSULATION_EFFECTIVENESS = 0.85     # housing blocks 85% of heat

# -- Storage / shelf life -----------------------------------------------------

SHELF_DEGRADATION_PER_SOL = 0.0005  # 0.05% yield loss per sol (Mars dry)
SHELF_DEGRADATION_PER_MIN = SHELF_DEGRADATION_PER_SOL / (24.66 * 60.0)
MIN_YIELD_FRACTION = 0.50           # below 50% yield, candle is unreliable

# -- Crew parameters ----------------------------------------------------------

O2_KG_PER_PERSON_PER_MIN = 0.84 / (24.66 * 60.0)  # 0.84 kg/person/sol

# -- Ignition -----------------------------------------------------------------

IGNITION_ENERGY_KJ = 0.05           # 50 J percussive primer
IGNITION_DELAY_MIN = 0.5            # 30 seconds to reach full burn rate

# -- Safety -------------------------------------------------------------------

MAX_SIMULTANEOUS_CANDLES = 8        # thermal limit in enclosed hab
FIRE_RISK_TEMP_K = 373.0            # hab air temp where fire risk spikes
VENTILATION_REQUIRED_M3_MIN = 0.5   # minimum airflow past burning candle


# -- Pure physics functions ---------------------------------------------------

def stoichiometric_o2_kg(naclo3_kg: float) -> float:
    """Maximum O₂ yield from given mass of sodium chlorate (kg).

    2 NaClO₃ → 2 NaCl + 3 O₂
    """
    naclo3_kg = max(0.0, naclo3_kg)
    return naclo3_kg * O2_YIELD_PER_KG_NACLO3


def stoichiometric_nacl_kg(naclo3_kg: float) -> float:
    """NaCl waste produced from given mass of NaClO₃ (kg)."""
    naclo3_kg = max(0.0, naclo3_kg)
    return naclo3_kg * NACL_YIELD_PER_KG_NACLO3


def reaction_heat_kj(naclo3_kg: float) -> float:
    """Total heat released by decomposing given mass of NaClO₃ (kJ)."""
    naclo3_kg = max(0.0, naclo3_kg)
    moles = naclo3_kg * 1000.0 / NACLO3_MOLAR_MASS_G
    return moles * DELTA_H_KJ_PER_MOL


def candle_o2_yield_kg(candle_mass_kg: float,
                       shelf_degradation: float = 0.0) -> float:
    """Total O₂ a candle can produce, accounting for degradation.

    Args:
        candle_mass_kg: total candle mass (kg)
        shelf_degradation: cumulative degradation fraction [0, 1]
    """
    candle_mass_kg = max(0.0, candle_mass_kg)
    shelf_degradation = max(0.0, min(1.0, shelf_degradation))
    base_yield = candle_mass_kg * O2_PER_KG_CANDLE
    return base_yield * (1.0 - shelf_degradation)


def burn_rate_kg_o2_min(candle_mass_kg: float,
                        burn_progress: float) -> float:
    """Instantaneous O₂ production rate (kg/min).

    Rate ramps up during ignition, sustains during main burn,
    and tapers in the last 10% as reactant is exhausted.

    Args:
        candle_mass_kg: total candle mass (kg)
        burn_progress: fraction of candle consumed [0, 1]
    """
    candle_mass_kg = max(0.0, candle_mass_kg)
    burn_progress = max(0.0, min(1.0, burn_progress))
    base_rate = BURN_RATE_KG_O2_PER_MIN * candle_mass_kg / DEFAULT_CANDLE_MASS_KG
    if burn_progress < 0.05:
        # Ignition ramp: linear from 0 to full over first 5%
        return base_rate * (burn_progress / 0.05)
    elif burn_progress > 0.90:
        # Taper: linear from full to 0 over last 10%
        return base_rate * ((1.0 - burn_progress) / 0.10)
    else:
        return base_rate


def heat_output_kw(candle_mass_kg: float,
                   burn_progress: float) -> float:
    """Instantaneous heat output (kW) from burning candle.

    Proportional to O₂ rate (both track reaction rate).
    """
    o2_rate = burn_rate_kg_o2_min(candle_mass_kg, burn_progress)
    if o2_rate <= 0.0:
        return 0.0
    # Heat per kg O₂ = total heat / total O₂
    total_o2 = candle_o2_yield_kg(candle_mass_kg)
    if total_o2 <= 0.0:
        return 0.0
    total_heat_kj = HEAT_PER_KG_CANDLE_KJ * candle_mass_kg
    heat_per_kg_o2 = total_heat_kj / total_o2
    heat_kj_per_min = o2_rate * heat_per_kg_o2
    return heat_kj_per_min / 60.0  # kJ/min → kW


def hab_temp_rise_k(heat_kw: float, hab_volume_m3: float,
                    ventilation_m3_min: float) -> float:
    """Temperature rise in hab from candle heat (K per minute).

    Simplified: air density ~1.0 kg/m³ at 70 kPa, Cp ~1.005 kJ/(kg·K).
    Ventilation carries heat away.

    Args:
        heat_kw: total heat input (kW)
        hab_volume_m3: hab interior volume (m³)
        ventilation_m3_min: air exchange rate (m³/min)
    """
    if hab_volume_m3 <= 0.0 or heat_kw <= 0.0:
        return 0.0
    air_density = 1.0  # kg/m³ at ~70 kPa, ~293 K
    cp_air = 1.005     # kJ/(kg·K)
    # Heat input per minute (kJ)
    heat_per_min = heat_kw * 60.0
    # Effective air mass exchanged per minute
    mass_exchanged = ventilation_m3_min * air_density
    # Heat absorbed by hab air mass
    hab_air_mass = hab_volume_m3 * air_density
    effective_mass = hab_air_mass + mass_exchanged
    return heat_per_min / (effective_mass * cp_air)


def personnel_hours(candle_mass_kg: float, crew_size: int,
                    shelf_degradation: float = 0.0) -> float:
    """Hours of breathable O₂ for given crew from one candle.

    Args:
        candle_mass_kg: candle mass (kg)
        crew_size: number of crew breathing
        shelf_degradation: cumulative degradation [0, 1]
    """
    if crew_size <= 0:
        return 0.0
    total_o2 = candle_o2_yield_kg(candle_mass_kg, shelf_degradation)
    o2_per_person_per_hour = O2_KG_PER_PERSON_PER_MIN * 60.0
    return total_o2 / (crew_size * o2_per_person_per_hour)


def candles_needed(crew_size: int, hours: float,
                   candle_mass_kg: float = DEFAULT_CANDLE_MASS_KG,
                   shelf_degradation: float = 0.0) -> int:
    """Minimum candles needed to sustain crew for given hours.

    Returns integer (ceiling), since you can't light half a candle
    and save the rest.
    """
    if crew_size <= 0 or hours <= 0.0:
        return 0
    per_candle_hours = personnel_hours(candle_mass_kg, crew_size,
                                       shelf_degradation)
    if per_candle_hours <= 0.0:
        return 0
    return math.ceil(hours / per_candle_hours)


def shelf_degradation_after(sols: float,
                            initial_degradation: float = 0.0) -> float:
    """Cumulative shelf degradation after storage for given sols.

    Linear model — Mars is extremely dry so NaClO₃ is stable.
    Capped at 1.0.
    """
    sols = max(0.0, sols)
    initial_degradation = max(0.0, min(1.0, initial_degradation))
    return min(1.0, initial_degradation + sols * SHELF_DEGRADATION_PER_SOL)


def is_candle_viable(shelf_degradation: float) -> bool:
    """Whether a candle is still reliable enough to ignite."""
    return shelf_degradation < (1.0 - MIN_YIELD_FRACTION)


# -- Candle state (one physical candle) ---------------------------------------

@dataclass
class Candle:
    """A single oxygen candle.

    Attributes:
        candle_id: unique identifier
        mass_kg: total candle mass (kg)
        shelf_degradation: cumulative storage degradation [0, 1]
        burn_progress: fraction consumed [0, 1] (0 = unlit, 1 = spent)
        is_burning: currently ignited
        o2_delivered_kg: total O₂ produced so far
        heat_delivered_kj: total heat released so far
        nacl_produced_kg: total NaCl waste produced
        age_sols: time since manufacture
    """
    candle_id: str = "candle-0"
    mass_kg: float = DEFAULT_CANDLE_MASS_KG
    shelf_degradation: float = 0.0
    burn_progress: float = 0.0
    is_burning: bool = False
    o2_delivered_kg: float = 0.0
    heat_delivered_kj: float = 0.0
    nacl_produced_kg: float = 0.0
    age_sols: float = 0.0

    def __post_init__(self) -> None:
        """Clamp fields to physical bounds."""
        self.mass_kg = max(0.0, self.mass_kg)
        self.shelf_degradation = max(0.0, min(1.0, self.shelf_degradation))
        self.burn_progress = max(0.0, min(1.0, self.burn_progress))
        self.o2_delivered_kg = max(0.0, self.o2_delivered_kg)
        self.heat_delivered_kj = max(0.0, self.heat_delivered_kj)
        self.nacl_produced_kg = max(0.0, self.nacl_produced_kg)
        self.age_sols = max(0.0, self.age_sols)

    @property
    def is_spent(self) -> bool:
        """Whether the candle has fully burned."""
        return self.burn_progress >= 1.0

    @property
    def is_ready(self) -> bool:
        """Whether the candle can be ignited."""
        return (not self.is_burning
                and not self.is_spent
                and is_candle_viable(self.shelf_degradation))

    @property
    def remaining_o2_kg(self) -> float:
        """O₂ still available from this candle."""
        total = candle_o2_yield_kg(self.mass_kg, self.shelf_degradation)
        return max(0.0, total - self.o2_delivered_kg)


def create_candle(candle_id: str = "candle-0",
                  mass_kg: float = DEFAULT_CANDLE_MASS_KG,
                  age_sols: float = 0.0) -> Candle:
    """Factory: create a new oxygen candle with shelf aging applied."""
    degradation = shelf_degradation_after(age_sols)
    return Candle(
        candle_id=candle_id,
        mass_kg=mass_kg,
        shelf_degradation=degradation,
        age_sols=age_sols,
    )


def ignite_candle(candle: Candle) -> str | None:
    """Ignite a candle. Returns error string or None on success."""
    if candle.is_burning:
        return "candle already burning"
    if candle.is_spent:
        return "candle is spent"
    if not is_candle_viable(candle.shelf_degradation):
        return "candle too degraded to ignite reliably"
    candle.is_burning = True
    return None


def tick_candle(candle: Candle, dt_min: float = 1.0) -> Dict[str, float]:
    """Advance one candle by dt minutes.

    Uses midpoint integration: estimate progress advance, evaluate rate
    at the midpoint of the interval, then compute actual O₂.  This
    handles the ignition ramp (where rate changes rapidly) correctly.

    Returns dict with:
        o2_kg: O₂ produced this tick
        heat_kj: heat released this tick
        nacl_kg: NaCl waste produced this tick
        burn_rate: current O₂ rate (kg/min)
    """
    result: Dict[str, float] = {
        "o2_kg": 0.0,
        "heat_kj": 0.0,
        "nacl_kg": 0.0,
        "burn_rate": 0.0,
    }
    dt_min = max(0.0, dt_min)
    if not candle.is_burning or candle.is_spent or dt_min == 0.0:
        return result

    total_yield = candle_o2_yield_kg(candle.mass_kg, candle.shelf_degradation)
    remaining = candle.remaining_o2_kg

    if total_yield <= 0.0 or remaining <= 1e-12:
        candle.is_burning = False
        candle.burn_progress = 1.0
        return result

    # If remaining is < 1% of total, deliver everything and finish.
    # Prevents asymptotic taper from creating a Zeno's paradox.
    if remaining < total_yield * 0.01:
        o2_this_tick = remaining
    else:
        # Midpoint integration: estimate progress at dt/2 for rate evaluation
        nominal_rate = BURN_RATE_KG_O2_PER_MIN * candle.mass_kg / DEFAULT_CANDLE_MASS_KG
        if nominal_rate > 0.0 and total_yield > 0.0:
            est_progress_delta = (nominal_rate * dt_min) / total_yield
            midpoint_progress = min(1.0, candle.burn_progress + est_progress_delta / 2.0)
        else:
            midpoint_progress = candle.burn_progress

        rate = burn_rate_kg_o2_min(candle.mass_kg, midpoint_progress)
        o2_this_tick = min(rate * dt_min, remaining)

    if o2_this_tick <= 0.0:
        # Truly no production possible — extinguish
        candle.is_burning = False
        candle.burn_progress = 1.0
        return result

    # Advance burn progress proportional to O₂ delivered
    progress_delta = o2_this_tick / total_yield
    candle.burn_progress = min(1.0, candle.burn_progress + progress_delta)

    # Heat output (proportional to O₂ produced)
    total_heat = HEAT_PER_KG_CANDLE_KJ * candle.mass_kg
    heat_kj = (o2_this_tick / total_yield) * total_heat

    # NaCl waste (stoichiometric: 2 NaCl per 3 O₂ by moles)
    nacl_kg = o2_this_tick * (2.0 * NACL_MOLAR_MASS_G) / (1.5 * O2_MOLAR_MASS_G)

    # Update candle state
    candle.o2_delivered_kg += o2_this_tick
    candle.heat_delivered_kj += heat_kj
    candle.nacl_produced_kg += nacl_kg

    # Check if spent
    if candle.burn_progress >= 1.0:
        candle.is_burning = False
        candle.burn_progress = 1.0

    result["o2_kg"] = o2_this_tick
    result["heat_kj"] = heat_kj
    result["nacl_kg"] = nacl_kg
    result["burn_rate"] = burn_rate_kg_o2_min(candle.mass_kg, candle.burn_progress)
    return result


# -- Candle inventory (colony level) ------------------------------------------

@dataclass
class CandleInventory:
    """Colony's oxygen candle stockpile.

    Tracks ready, burning, and spent candles. Provides emergency
    O₂ capacity calculations.

    Attributes:
        candles: all candles in inventory
        hab_volume_m3: habitat interior volume (for heat calculations)
        ventilation_m3_min: airflow rate (m³/min)
    """
    candles: List[Candle] = field(default_factory=list)
    hab_volume_m3: float = 500.0
    ventilation_m3_min: float = 2.0
    total_o2_delivered_kg: float = 0.0
    total_heat_delivered_kj: float = 0.0

    @property
    def ready_count(self) -> int:
        """Candles available for ignition."""
        return sum(1 for c in self.candles if c.is_ready)

    @property
    def burning_count(self) -> int:
        """Currently burning candles."""
        return sum(1 for c in self.candles if c.is_burning)

    @property
    def spent_count(self) -> int:
        """Fully consumed candles."""
        return sum(1 for c in self.candles if c.is_spent)

    @property
    def total_count(self) -> int:
        """Total candles in inventory."""
        return len(self.candles)

    @property
    def emergency_capacity_hours(self) -> float:
        """Hours of emergency O₂ for 1 person from all ready candles."""
        return sum(
            personnel_hours(c.mass_kg, 1, c.shelf_degradation)
            for c in self.candles if c.is_ready
        )

    def inventory_check(self) -> bool:
        """Verify conservation: total = ready + burning + spent + degraded."""
        ready = self.ready_count
        burning = self.burning_count
        spent = self.spent_count
        degraded = sum(
            1 for c in self.candles
            if not c.is_burning and not c.is_spent
            and not is_candle_viable(c.shelf_degradation)
        )
        return (ready + burning + spent + degraded) == self.total_count


def create_inventory(num_candles: int = 20,
                     candle_mass_kg: float = DEFAULT_CANDLE_MASS_KG,
                     age_sols: float = 0.0,
                     hab_volume_m3: float = 500.0) -> CandleInventory:
    """Factory: create a colony candle stockpile."""
    candles = [
        create_candle(
            candle_id=f"candle-{i:03d}",
            mass_kg=candle_mass_kg,
            age_sols=age_sols,
        )
        for i in range(num_candles)
    ]
    return CandleInventory(candles=candles, hab_volume_m3=hab_volume_m3)


def activate_emergency(inventory: CandleInventory,
                       num_candles: int = 1) -> List[str]:
    """Ignite N candles for emergency O₂.

    Returns list of error messages (empty = all succeeded).
    """
    errors: List[str] = []
    ignited = 0
    for candle in inventory.candles:
        if ignited >= num_candles:
            break
        if candle.is_ready:
            err = ignite_candle(candle)
            if err:
                errors.append(f"{candle.candle_id}: {err}")
            else:
                ignited += 1
    if ignited < num_candles:
        errors.append(
            f"only {ignited}/{num_candles} candles ignited "
            f"({inventory.ready_count} were ready)"
        )
    return errors


def tick_inventory(inventory: CandleInventory,
                   dt_min: float = 1.0) -> Dict[str, float]:
    """Advance all burning candles by dt minutes.

    Returns aggregate results:
        o2_kg: total O₂ produced this tick
        heat_kj: total heat released
        nacl_kg: total NaCl waste
        active_candles: number still burning
        hab_temp_rise_k: estimated hab temperature rise
    """
    total_o2 = 0.0
    total_heat = 0.0
    total_nacl = 0.0
    active = 0

    for candle in inventory.candles:
        if candle.is_burning:
            result = tick_candle(candle, dt_min)
            total_o2 += result["o2_kg"]
            total_heat += result["heat_kj"]
            total_nacl += result["nacl_kg"]
            if candle.is_burning:
                active += 1

    # Update inventory totals
    inventory.total_o2_delivered_kg += total_o2
    inventory.total_heat_delivered_kj += total_heat

    # Estimate hab temperature impact
    total_heat_kw = total_heat / (dt_min * 60.0) if dt_min > 0 else 0.0
    temp_rise = hab_temp_rise_k(
        total_heat_kw * (1.0 - INSULATION_EFFECTIVENESS),
        inventory.hab_volume_m3,
        inventory.ventilation_m3_min,
    )

    return {
        "o2_kg": total_o2,
        "heat_kj": total_heat,
        "nacl_kg": total_nacl,
        "active_candles": active,
        "hab_temp_rise_k": temp_rise,
    }


def age_inventory(inventory: CandleInventory, sols: float) -> int:
    """Age all candles in storage. Returns count that became non-viable."""
    expired = 0
    for candle in inventory.candles:
        if candle.is_burning or candle.is_spent:
            continue
        was_viable = is_candle_viable(candle.shelf_degradation)
        candle.age_sols += sols
        candle.shelf_degradation = shelf_degradation_after(
            candle.age_sols
        )
        if was_viable and not is_candle_viable(candle.shelf_degradation):
            expired += 1
    return expired


def emergency_duration_hours(inventory: CandleInventory,
                             crew_size: int) -> float:
    """Total emergency O₂ duration for given crew size (hours).

    Accounts for shelf degradation of each candle individually.
    """
    if crew_size <= 0:
        return 0.0
    total_hours = 0.0
    for candle in inventory.candles:
        if candle.is_ready or candle.is_burning:
            total_hours += personnel_hours(
                candle.mass_kg, crew_size, candle.shelf_degradation
            )
    return total_hours


def to_dict(inventory: CandleInventory) -> Dict[str, Any]:
    """Serialize inventory to JSON-safe dict."""
    return {
        "total_candles": inventory.total_count,
        "ready": inventory.ready_count,
        "burning": inventory.burning_count,
        "spent": inventory.spent_count,
        "hab_volume_m3": inventory.hab_volume_m3,
        "total_o2_delivered_kg": round(inventory.total_o2_delivered_kg, 6),
        "total_heat_delivered_kj": round(inventory.total_heat_delivered_kj, 3),
        "candles": [
            {
                "id": c.candle_id,
                "mass_kg": c.mass_kg,
                "shelf_degradation": round(c.shelf_degradation, 6),
                "burn_progress": round(c.burn_progress, 6),
                "is_burning": c.is_burning,
                "o2_delivered_kg": round(c.o2_delivered_kg, 6),
                "age_sols": round(c.age_sols, 2),
            }
            for c in inventory.candles
        ],
    }
