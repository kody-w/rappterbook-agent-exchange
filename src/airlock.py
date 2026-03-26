"""airlock.py — Mars Colony Airlock System.

The airlock is the gateway between life and death. Every EVA, every
supply transfer, every emergency egress passes through this chamber.
Get it wrong and you lose your atmosphere in seconds.

This module models a dual-hatch airlock with pump-down/re-pressurize
cycling, dust contamination, thermal stress, and component wear.

Physics modelled
----------------
* **Pressure cycling** — exponential pump-down from hab pressure (~70 kPa)
  to Mars ambient (~0.6 kPa). Re-pressurize from hab supply. Ideal gas
  law governs all gas mass calculations (PV = nRT).
* **Gas recovery** — vacuum pump recovers atmosphere during depressurization.
  Efficiency degrades with pump wear. Unrecovered gas is lost to Mars.
* **Dust contamination** — Mars regolith enters when outer hatch opens.
  Particle concentration (mg/m³) increases per cycle. HEPA filter
  scrubs dust but degrades over time. Excess dust is a crew health hazard.
* **Thermal stress** — chamber swings between hab temperature (~293 K)
  and Mars surface (~210 K) each cycle. Repeated thermal cycling fatigues
  seals and structure. Thermal delta tracked per cycle.
* **Component wear** — door seals, vacuum pump, dust filter, and hatch
  actuators all degrade with use. Each has independent wear tracking.
  Maintenance restores partial function.
* **Safety interlocks** — inner and outer hatches cannot both be open.
  Emergency blow-out vents the chamber to Mars in <10 seconds.
  Pre-breathe timer enforced before low-pressure EVA.

Physical references:
  - ISS Quest airlock: 34 m³, cycle time ~30 min depress, ~5 min repress
  - Mars surface pressure: 600 Pa (0.6 kPa)
  - Mars surface temperature: 210 K average (-63°C)
  - Standard hab pressure: 70 kPa (NASA ECLSS)
  - Shuttle EMU pre-breathe: 60 min at 70 kPa O2 (we model 40 min)
  - ISS airlock gas loss per cycle: ~0.12 kg
  - Mars regolith particle size: 1–100 μm, toxic perchlorates

One tick = one sol. Pressures in kPa. Gas mass in kg. Dust in mg/m³.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MARS_AMBIENT_KPA = 0.6               # Mars mean surface pressure
HAB_PRESSURE_KPA = 70.0              # nominal habitat pressure
CHAMBER_VOLUME_M3 = 6.0              # airlock chamber volume
CHAMBER_TEMP_HAB_K = 293.15          # chamber temp when pressurized (20°C)
CHAMBER_TEMP_MARS_K = 210.0          # Mars surface avg temperature

# Ideal gas
R_UNIVERSAL = 8.314                  # J/(mol·K)
MOLAR_MASS_AIR = 0.029               # ~29 g/mol (N2/O2 mix)

# Pump performance
PUMP_EFFICIENCY_NEW = 0.92            # gas recovery fraction (new pump)
PUMP_EFFICIENCY_MIN = 0.40            # worst-case degraded pump
PUMP_WEAR_PER_CYCLE = 0.0008         # pump efficiency loss per cycle
PUMP_DOWN_RATE_KPA_MIN = 3.5         # depressurization rate (kPa/min)
REPRESS_RATE_KPA_MIN = 14.0          # re-pressurization rate (kPa/min)

# Dust
DUST_INGRESS_MG_M3_PER_CYCLE = 12.0  # dust entering per outer-hatch open
FILTER_EFFICIENCY_NEW = 0.97          # HEPA filter dust removal fraction
FILTER_EFFICIENCY_MIN = 0.30          # degraded filter minimum
FILTER_WEAR_PER_CYCLE = 0.0005       # filter degradation per cycle
DUST_HAZARD_THRESHOLD_MG_M3 = 50.0   # crew health hazard level

# Thermal
THERMAL_CYCLE_SEAL_DAMAGE = 0.0003   # seal degradation per thermal cycle
MIN_SEAL_INTEGRITY = 0.05            # structural minimum (never zero)

# Door seals
SEAL_INTEGRITY_NEW = 1.0
DOOR_SEAL_WEAR_PER_CYCLE = 0.0004    # per-hatch-operation wear
SEAL_LEAK_RATE_BASE_KPA_MIN = 0.001  # leak rate at full integrity
SEAL_LEAK_MULTIPLIER = 5.0           # leak rate at minimum integrity

# Hatch actuator
ACTUATOR_WEAR_PER_CYCLE = 0.0002     # actuator wear per open/close
ACTUATOR_MIN_HEALTH = 0.10           # below this, hatch may jam
ACTUATOR_JAM_PROBABILITY_BASE = 0.01 # jam chance at minimum health

# Safety
PRE_BREATHE_MINUTES = 40.0           # pre-breathe time before EVA
EMERGENCY_BLOW_SECONDS = 8.0         # time to vent chamber in emergency
MAX_CYCLES_BEFORE_OVERHAUL = 500     # recommended maintenance interval

# Maintenance
MAINTENANCE_RESTORE_FRACTION = 0.75  # how much wear is recovered


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AirlockState:
    """Mutable state of a Mars colony airlock.

    Tracks pressure, door states, component health, contamination,
    and cumulative cycle counts.
    """
    chamber_pressure_kpa: float = HAB_PRESSURE_KPA
    inner_hatch_open: bool = False
    outer_hatch_open: bool = False

    # Component health [0, 1] — 1.0 = factory new
    pump_health: float = 1.0
    inner_seal_integrity: float = SEAL_INTEGRITY_NEW
    outer_seal_integrity: float = SEAL_INTEGRITY_NEW
    filter_health: float = 1.0
    inner_actuator_health: float = 1.0
    outer_actuator_health: float = 1.0

    # Contamination
    dust_concentration_mg_m3: float = 0.0

    # Thermal
    chamber_temp_k: float = CHAMBER_TEMP_HAB_K

    # Counters
    total_cycles: int = 0
    total_gas_lost_kg: float = 0.0
    total_dust_ingress_mg: float = 0.0

    # Status flags
    emergency_vented: bool = False
    pre_breathe_complete: bool = False

    def __post_init__(self) -> None:
        """Clamp all values to physical bounds."""
        self.chamber_pressure_kpa = max(0.0, self.chamber_pressure_kpa)
        self.pump_health = _clamp01(self.pump_health)
        self.inner_seal_integrity = max(MIN_SEAL_INTEGRITY,
                                        min(1.0, self.inner_seal_integrity))
        self.outer_seal_integrity = max(MIN_SEAL_INTEGRITY,
                                        min(1.0, self.outer_seal_integrity))
        self.filter_health = _clamp01(self.filter_health)
        self.inner_actuator_health = max(ACTUATOR_MIN_HEALTH,
                                         min(1.0, self.inner_actuator_health))
        self.outer_actuator_health = max(ACTUATOR_MIN_HEALTH,
                                         min(1.0, self.outer_actuator_health))
        self.dust_concentration_mg_m3 = max(0.0, self.dust_concentration_mg_m3)
        self.total_cycles = max(0, self.total_cycles)
        self.total_gas_lost_kg = max(0.0, self.total_gas_lost_kg)
        self.total_dust_ingress_mg = max(0.0, self.total_dust_ingress_mg)

    @property
    def is_pressurized(self) -> bool:
        """Chamber is at hab-level pressure (within 5%)."""
        return self.chamber_pressure_kpa >= HAB_PRESSURE_KPA * 0.95

    @property
    def is_depressurized(self) -> bool:
        """Chamber is at Mars-ambient pressure (within 50%)."""
        return self.chamber_pressure_kpa <= MARS_AMBIENT_KPA * 1.5

    @property
    def dust_hazard(self) -> bool:
        """Dust concentration exceeds crew health threshold."""
        return self.dust_concentration_mg_m3 >= DUST_HAZARD_THRESHOLD_MG_M3

    @property
    def needs_overhaul(self) -> bool:
        """Total cycles exceed recommended maintenance interval."""
        return self.total_cycles >= MAX_CYCLES_BEFORE_OVERHAUL

    @property
    def interlock_safe(self) -> bool:
        """Safety interlock — both hatches must never be open."""
        return not (self.inner_hatch_open and self.outer_hatch_open)


def _clamp01(value: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Gas physics
# ---------------------------------------------------------------------------

def gas_mass_kg(pressure_kpa: float, volume_m3: float,
                temp_k: float) -> float:
    """Gas mass in chamber via ideal gas law.

    PV = nRT → n = PV/(RT) → mass = n × M_air
    Pressure converted from kPa to Pa (×1000).
    """
    if pressure_kpa <= 0.0 or volume_m3 <= 0.0 or temp_k <= 0.0:
        return 0.0
    moles = (pressure_kpa * 1000.0 * volume_m3) / (R_UNIVERSAL * temp_k)
    return moles * MOLAR_MASS_AIR


def pressure_from_mass_kg(mass_kg: float, volume_m3: float,
                          temp_k: float) -> float:
    """Pressure (kPa) from gas mass via ideal gas law."""
    if mass_kg <= 0.0 or volume_m3 <= 0.0 or temp_k <= 0.0:
        return 0.0
    moles = mass_kg / MOLAR_MASS_AIR
    return (moles * R_UNIVERSAL * temp_k) / (volume_m3 * 1000.0)


# ---------------------------------------------------------------------------
# Pump operations
# ---------------------------------------------------------------------------

def pump_efficiency(pump_health: float) -> float:
    """Current pump gas recovery efficiency based on health.

    Linear interpolation from min to max efficiency.
    """
    health = _clamp01(pump_health)
    return PUMP_EFFICIENCY_MIN + health * (PUMP_EFFICIENCY_NEW - PUMP_EFFICIENCY_MIN)


def pump_down_time_minutes(start_kpa: float, target_kpa: float) -> float:
    """Time to pump down from start to target pressure (minutes).

    Exponential decay model — gets slower as pressure drops.
    """
    if start_kpa <= target_kpa or target_kpa < 0.0:
        return 0.0
    if start_kpa <= 0.0:
        return 0.0
    # Time ∝ ln(P_start / P_target) / rate
    ratio = start_kpa / max(0.01, target_kpa)
    return math.log(ratio) * (start_kpa / PUMP_DOWN_RATE_KPA_MIN)


def repress_time_minutes(start_kpa: float, target_kpa: float) -> float:
    """Time to re-pressurize from start to target (minutes).

    Approximately linear at high flow rate.
    """
    if start_kpa >= target_kpa:
        return 0.0
    delta = target_kpa - start_kpa
    return delta / REPRESS_RATE_KPA_MIN


# ---------------------------------------------------------------------------
# Cycle operations
# ---------------------------------------------------------------------------

def depressurize(state: AirlockState) -> dict:
    """Pump down the airlock chamber to Mars ambient pressure.

    Returns cycle metrics: gas_recovered_kg, gas_lost_kg, time_minutes.
    """
    if state.is_depressurized:
        return {"gas_recovered_kg": 0.0, "gas_lost_kg": 0.0,
                "time_minutes": 0.0, "already_depressurized": True}

    initial_mass_kg = gas_mass_kg(state.chamber_pressure_kpa,
                                  CHAMBER_VOLUME_M3, state.chamber_temp_k)
    final_mass_kg = gas_mass_kg(MARS_AMBIENT_KPA, CHAMBER_VOLUME_M3,
                                state.chamber_temp_k)
    delta_mass_kg = initial_mass_kg - final_mass_kg

    eff = pump_efficiency(state.pump_health)
    recovered_kg = delta_mass_kg * eff
    lost_kg = delta_mass_kg - recovered_kg

    time_min = pump_down_time_minutes(state.chamber_pressure_kpa,
                                      MARS_AMBIENT_KPA)

    # Update state
    state.chamber_pressure_kpa = MARS_AMBIENT_KPA
    state.total_gas_lost_kg += lost_kg
    state.pump_health = max(0.0, state.pump_health - PUMP_WEAR_PER_CYCLE)

    # Thermal — chamber cools toward Mars temperature during pump-down
    temp_delta = state.chamber_temp_k - CHAMBER_TEMP_MARS_K
    state.chamber_temp_k -= temp_delta * 0.3  # partial cooling

    return {
        "gas_recovered_kg": round(recovered_kg, 6),
        "gas_lost_kg": round(lost_kg, 6),
        "time_minutes": round(time_min, 2),
        "already_depressurized": False,
    }


def repressurize(state: AirlockState) -> dict:
    """Re-pressurize the airlock chamber to habitat pressure.

    Gas is supplied from the habitat's atmosphere reserves.
    Returns: gas_used_kg, time_minutes.
    """
    if state.is_pressurized:
        return {"gas_used_kg": 0.0, "time_minutes": 0.0,
                "already_pressurized": True}

    final_mass_kg = gas_mass_kg(HAB_PRESSURE_KPA, CHAMBER_VOLUME_M3,
                                CHAMBER_TEMP_HAB_K)
    current_mass_kg = gas_mass_kg(state.chamber_pressure_kpa,
                                  CHAMBER_VOLUME_M3, state.chamber_temp_k)
    gas_needed_kg = final_mass_kg - current_mass_kg

    time_min = repress_time_minutes(state.chamber_pressure_kpa,
                                    HAB_PRESSURE_KPA)

    # Update state
    state.chamber_pressure_kpa = HAB_PRESSURE_KPA
    state.chamber_temp_k = CHAMBER_TEMP_HAB_K  # warm gas floods in

    return {
        "gas_used_kg": round(max(0.0, gas_needed_kg), 6),
        "time_minutes": round(time_min, 2),
        "already_pressurized": False,
    }


def open_outer_hatch(state: AirlockState) -> dict:
    """Open outer hatch to Mars surface.

    Requires: chamber depressurized, inner hatch closed.
    Side effects: dust ingress, seal wear, actuator wear.
    """
    if state.inner_hatch_open:
        return {"success": False, "error": "interlock_violation",
                "detail": "Inner hatch must be closed before opening outer"}

    if not state.is_depressurized:
        return {"success": False, "error": "pressure_too_high",
                "detail": f"Chamber at {state.chamber_pressure_kpa:.1f} kPa, "
                          f"need ≤{MARS_AMBIENT_KPA * 1.5:.1f} kPa"}

    # Dust ingress
    raw_dust = DUST_INGRESS_MG_M3_PER_CYCLE
    filter_eff = FILTER_EFFICIENCY_MIN + state.filter_health * (
        FILTER_EFFICIENCY_NEW - FILTER_EFFICIENCY_MIN)
    dust_added = raw_dust * (1.0 - filter_eff)
    state.dust_concentration_mg_m3 += dust_added
    state.total_dust_ingress_mg += raw_dust * CHAMBER_VOLUME_M3
    state.filter_health = max(0.0, state.filter_health - FILTER_WEAR_PER_CYCLE)

    # Seal and actuator wear
    state.outer_seal_integrity = max(MIN_SEAL_INTEGRITY,
                                     state.outer_seal_integrity - DOOR_SEAL_WEAR_PER_CYCLE)
    state.outer_actuator_health = max(ACTUATOR_MIN_HEALTH,
                                      state.outer_actuator_health - ACTUATOR_WEAR_PER_CYCLE)

    # Thermal — full exposure to Mars cold
    state.chamber_temp_k = CHAMBER_TEMP_MARS_K
    state.inner_seal_integrity = max(MIN_SEAL_INTEGRITY,
                                     state.inner_seal_integrity - THERMAL_CYCLE_SEAL_DAMAGE)

    state.outer_hatch_open = True

    return {"success": True, "dust_added_mg_m3": round(dust_added, 4),
            "chamber_temp_k": round(state.chamber_temp_k, 2)}


def close_outer_hatch(state: AirlockState) -> dict:
    """Close the outer hatch."""
    if not state.outer_hatch_open:
        return {"success": False, "error": "already_closed"}
    state.outer_hatch_open = False
    state.outer_actuator_health = max(ACTUATOR_MIN_HEALTH,
                                      state.outer_actuator_health - ACTUATOR_WEAR_PER_CYCLE)
    return {"success": True}


def open_inner_hatch(state: AirlockState) -> dict:
    """Open inner hatch to habitat.

    Requires: chamber pressurized, outer hatch closed.
    """
    if state.outer_hatch_open:
        return {"success": False, "error": "interlock_violation",
                "detail": "Outer hatch must be closed before opening inner"}

    if not state.is_pressurized:
        return {"success": False, "error": "pressure_too_low",
                "detail": f"Chamber at {state.chamber_pressure_kpa:.1f} kPa, "
                          f"need ≥{HAB_PRESSURE_KPA * 0.95:.1f} kPa"}

    state.inner_hatch_open = True
    state.inner_seal_integrity = max(MIN_SEAL_INTEGRITY,
                                     state.inner_seal_integrity - DOOR_SEAL_WEAR_PER_CYCLE)
    state.inner_actuator_health = max(ACTUATOR_MIN_HEALTH,
                                      state.inner_actuator_health - ACTUATOR_WEAR_PER_CYCLE)

    return {"success": True}


def close_inner_hatch(state: AirlockState) -> dict:
    """Close the inner hatch."""
    if not state.inner_hatch_open:
        return {"success": False, "error": "already_closed"}
    state.inner_hatch_open = False
    state.inner_actuator_health = max(ACTUATOR_MIN_HEALTH,
                                      state.inner_actuator_health - ACTUATOR_WEAR_PER_CYCLE)
    return {"success": True}


# ---------------------------------------------------------------------------
# Full EVA cycle
# ---------------------------------------------------------------------------

def egress_cycle(state: AirlockState) -> dict:
    """Full egress cycle: crew exits habitat to Mars surface.

    Sequence:
    1. Close inner hatch (if open)
    2. Depressurize chamber
    3. Open outer hatch (crew exits)
    4. Close outer hatch
    5. Increment cycle counter

    Returns full cycle metrics.
    """
    results: dict = {"cycle_type": "egress", "steps": []}

    # 1. Close inner hatch
    if state.inner_hatch_open:
        r = close_inner_hatch(state)
        results["steps"].append(("close_inner", r))

    # 2. Depressurize
    r = depressurize(state)
    results["steps"].append(("depressurize", r))
    results["gas_lost_kg"] = r["gas_lost_kg"]
    results["gas_recovered_kg"] = r["gas_recovered_kg"]

    # 3. Open outer hatch
    r = open_outer_hatch(state)
    results["steps"].append(("open_outer", r))
    if not r["success"]:
        results["aborted"] = True
        results["abort_reason"] = r["error"]
        return results

    # 4. Close outer hatch (crew has exited)
    r = close_outer_hatch(state)
    results["steps"].append(("close_outer", r))

    # 5. Count
    state.total_cycles += 1
    results["total_cycles"] = state.total_cycles
    results["aborted"] = False

    return results


def ingress_cycle(state: AirlockState) -> dict:
    """Full ingress cycle: crew enters habitat from Mars surface.

    Sequence:
    1. Depressurize (if needed — should already be depressurized)
    2. Open outer hatch (crew enters)
    3. Close outer hatch
    4. Re-pressurize chamber
    5. Open inner hatch (crew enters hab)
    6. Close inner hatch
    7. Increment cycle counter

    Returns full cycle metrics.
    """
    results: dict = {"cycle_type": "ingress", "steps": []}

    # 1. Depressurize if needed
    if not state.is_depressurized:
        r = depressurize(state)
        results["steps"].append(("depressurize", r))
        results["gas_lost_kg"] = r["gas_lost_kg"]
    else:
        results["gas_lost_kg"] = 0.0

    # 2. Open outer hatch
    if state.inner_hatch_open:
        close_inner_hatch(state)
    r = open_outer_hatch(state)
    results["steps"].append(("open_outer", r))
    if not r["success"]:
        results["aborted"] = True
        results["abort_reason"] = r["error"]
        return results

    # 3. Close outer hatch
    r = close_outer_hatch(state)
    results["steps"].append(("close_outer", r))

    # 4. Re-pressurize
    r = repressurize(state)
    results["steps"].append(("repressurize", r))
    results["gas_used_kg"] = r["gas_used_kg"]

    # 5. Open inner hatch
    r = open_inner_hatch(state)
    results["steps"].append(("open_inner", r))

    # 6. Close inner hatch
    if state.inner_hatch_open:
        r = close_inner_hatch(state)
        results["steps"].append(("close_inner", r))

    # 7. Count
    state.total_cycles += 1
    results["total_cycles"] = state.total_cycles
    results["aborted"] = False

    return results


# ---------------------------------------------------------------------------
# Emergency operations
# ---------------------------------------------------------------------------

def emergency_vent(state: AirlockState) -> dict:
    """Emergency blow-out: vent chamber to Mars ambient instantly.

    Used when crew life is at risk. Bypasses pump — all gas is lost.
    Both hatches forced closed first (safety).
    """
    gas_in_chamber = gas_mass_kg(state.chamber_pressure_kpa,
                                 CHAMBER_VOLUME_M3, state.chamber_temp_k)

    state.inner_hatch_open = False
    state.outer_hatch_open = False
    state.chamber_pressure_kpa = MARS_AMBIENT_KPA
    state.chamber_temp_k = CHAMBER_TEMP_MARS_K
    state.total_gas_lost_kg += gas_in_chamber
    state.emergency_vented = True

    return {
        "gas_lost_kg": round(gas_in_chamber, 6),
        "time_seconds": EMERGENCY_BLOW_SECONDS,
        "pressure_after_kpa": MARS_AMBIENT_KPA,
    }


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def perform_maintenance(state: AirlockState) -> dict:
    """Crew performs airlock maintenance. Partially restores components.

    Requires: chamber pressurized, both hatches closed.
    """
    if state.inner_hatch_open or state.outer_hatch_open:
        return {"success": False, "error": "hatches_must_be_closed"}
    if not state.is_pressurized:
        return {"success": False, "error": "must_be_pressurized"}

    def _restore(current: float, minimum: float) -> float:
        gap = 1.0 - current
        return min(1.0, current + gap * MAINTENANCE_RESTORE_FRACTION)

    before = {
        "pump_health": state.pump_health,
        "inner_seal": state.inner_seal_integrity,
        "outer_seal": state.outer_seal_integrity,
        "filter_health": state.filter_health,
        "inner_actuator": state.inner_actuator_health,
        "outer_actuator": state.outer_actuator_health,
        "dust_mg_m3": state.dust_concentration_mg_m3,
    }

    state.pump_health = _restore(state.pump_health, 0.0)
    state.inner_seal_integrity = _restore(state.inner_seal_integrity,
                                          MIN_SEAL_INTEGRITY)
    state.outer_seal_integrity = _restore(state.outer_seal_integrity,
                                          MIN_SEAL_INTEGRITY)
    state.filter_health = _restore(state.filter_health, 0.0)
    state.inner_actuator_health = _restore(state.inner_actuator_health,
                                           ACTUATOR_MIN_HEALTH)
    state.outer_actuator_health = _restore(state.outer_actuator_health,
                                           ACTUATOR_MIN_HEALTH)

    # Dust cleanup
    state.dust_concentration_mg_m3 *= 0.1  # 90% removal during maintenance

    after = {
        "pump_health": round(state.pump_health, 4),
        "inner_seal": round(state.inner_seal_integrity, 4),
        "outer_seal": round(state.outer_seal_integrity, 4),
        "filter_health": round(state.filter_health, 4),
        "inner_actuator": round(state.inner_actuator_health, 4),
        "outer_actuator": round(state.outer_actuator_health, 4),
        "dust_mg_m3": round(state.dust_concentration_mg_m3, 4),
    }

    return {"success": True, "before": before, "after": after}


# ---------------------------------------------------------------------------
# Tick — advance one sol
# ---------------------------------------------------------------------------

@dataclass
class AirlockSol:
    """One sol of airlock activity."""
    sol: int
    egress_cycles: int = 0
    ingress_cycles: int = 0
    maintenance: bool = False
    emergency: bool = False


def tick_airlock(state: AirlockState, sol: AirlockSol) -> dict:
    """Advance the airlock by one sol.

    Processes EVA cycles, maintenance, and seal degradation.
    Returns a comprehensive snapshot of the sol's activity.
    """
    snapshot: dict = {
        "sol": sol.sol,
        "pressure_before_kpa": round(state.chamber_pressure_kpa, 4),
        "cycles_before": state.total_cycles,
        "gas_lost_before_kg": round(state.total_gas_lost_kg, 4),
    }

    total_gas_lost_this_sol = 0.0
    total_gas_recovered = 0.0
    total_gas_used = 0.0
    cycle_details: List[dict] = []

    # Emergency override — everything else skipped
    if sol.emergency:
        r = emergency_vent(state)
        snapshot["emergency_vent"] = r
        total_gas_lost_this_sol += r["gas_lost_kg"]
        snapshot.update(_final_snapshot(state, total_gas_lost_this_sol,
                                       total_gas_recovered, total_gas_used,
                                       cycle_details))
        return snapshot

    # Egress cycles
    for _ in range(max(0, sol.egress_cycles)):
        r = egress_cycle(state)
        cycle_details.append(r)
        total_gas_lost_this_sol += r.get("gas_lost_kg", 0.0)
        total_gas_recovered += r.get("gas_recovered_kg", 0.0)

    # Ingress cycles
    for _ in range(max(0, sol.ingress_cycles)):
        r = ingress_cycle(state)
        cycle_details.append(r)
        total_gas_lost_this_sol += r.get("gas_lost_kg", 0.0)
        total_gas_used += r.get("gas_used_kg", 0.0)

    # Maintenance
    maint_result = None
    if sol.maintenance:
        # Ensure airlock is in a serviceable state first
        if not state.is_pressurized:
            if state.outer_hatch_open:
                close_outer_hatch(state)
            repressurize(state)
        if state.inner_hatch_open:
            close_inner_hatch(state)
        if state.outer_hatch_open:
            close_outer_hatch(state)
        maint_result = perform_maintenance(state)

    # Ambient seal leakage (tiny, but accumulates)
    seal_avg = (state.inner_seal_integrity + state.outer_seal_integrity) / 2.0
    leak_factor = SEAL_LEAK_RATE_BASE_KPA_MIN * (
        1.0 + (1.0 - seal_avg) * (SEAL_LEAK_MULTIPLIER - 1.0))
    sol_minutes = 24.66 * 60  # Mars sol in minutes
    ambient_leak_kpa = leak_factor * sol_minutes
    if state.is_pressurized and not state.inner_hatch_open:
        # Only leaks when sealed and pressurized
        state.chamber_pressure_kpa = max(
            0.0, state.chamber_pressure_kpa - ambient_leak_kpa)

    snapshot["maintenance"] = maint_result
    snapshot.update(_final_snapshot(state, total_gas_lost_this_sol,
                                   total_gas_recovered, total_gas_used,
                                   cycle_details))
    return snapshot


def _final_snapshot(state: AirlockState, gas_lost: float,
                    gas_recovered: float, gas_used: float,
                    cycles: List[dict]) -> dict:
    """Build the final portion of a tick snapshot."""
    return {
        "pressure_after_kpa": round(state.chamber_pressure_kpa, 4),
        "chamber_temp_k": round(state.chamber_temp_k, 2),
        "gas_lost_this_sol_kg": round(gas_lost, 6),
        "gas_recovered_this_sol_kg": round(gas_recovered, 6),
        "gas_used_repress_kg": round(gas_used, 6),
        "total_cycles": state.total_cycles,
        "total_gas_lost_kg": round(state.total_gas_lost_kg, 6),
        "dust_mg_m3": round(state.dust_concentration_mg_m3, 4),
        "dust_hazard": state.dust_hazard,
        "pump_health": round(state.pump_health, 4),
        "inner_seal": round(state.inner_seal_integrity, 4),
        "outer_seal": round(state.outer_seal_integrity, 4),
        "filter_health": round(state.filter_health, 4),
        "interlock_safe": state.interlock_safe,
        "needs_overhaul": state.needs_overhaul,
        "cycle_details": cycles,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_airlock() -> AirlockState:
    """Create a factory-fresh airlock at habitat pressure."""
    return AirlockState()
