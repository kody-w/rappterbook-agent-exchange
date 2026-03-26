"""nuclear_reactor.py — Mars Colony Fission Reactor.

Models a Kilopower-class fission reactor for reliable colony power.
Solar arrays fail during dust storms (weeks of darkness).  The reactor
keeps the lights on.  One tick = one sol.

Physics modelled
----------------
* **Neutron chain reaction** — reactor thermal power from U-235 fission.
  Control rod position modulates reactivity.  Negative temperature
  coefficient provides passive safety (hotter = less reactive).
* **Stirling conversion** — heat-to-electricity via free-piston Stirling
  engines.  Efficiency ~25% (Carnot-limited by hot/cold side temps).
* **Fuel depletion** — U-235 burnup over years.  10 kWe Kilopower has
  ~15 year fuel life.  Burnup reduces max thermal power linearly.
* **Thermal management** — radiator panels reject waste heat to Mars
  environment.  Radiator degradation from dust and micrometeorites.
* **Reactor states** — shutdown, startup (criticality approach), nominal,
  SCRAM (emergency shutdown).  Startup takes ~5 sols for thermal soak.
* **Radiation shielding** — shadow shield protects habitat.  Shield
  effectiveness degrades if habitat moves relative to reactor.

Reference hardware:
  - NASA Kilopower/KRUSTY: 1-10 kWe, U-235 core, Stirling engines
  - Tested at Nevada National Security Site (2018)
  - Colony reactor (this model): 40 kWe nominal, 15-year fuel life
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Core physics
U235_HALF_LIFE_YEARS = 703.8e6      # U-235 half-life (years)
FUEL_LOAD_KG = 50.0                 # enriched uranium fuel mass
FUEL_LIFE_SOLS = 15 * 365           # ~15 Earth years in sols (~5475)
THERMAL_POWER_MAX_KW = 160.0        # max thermal output at beginning of life

# Stirling conversion
STIRLING_EFFICIENCY = 0.25           # thermal-to-electric efficiency
ELECTRIC_POWER_MAX_KW = THERMAL_POWER_MAX_KW * STIRLING_EFFICIENCY  # 40 kWe
HOT_SIDE_TEMP_K = 1073.0            # core hot side (~800°C)
COLD_SIDE_TEMP_K = 373.0            # radiator cold side (~100°C)
CARNOT_EFFICIENCY = 1.0 - COLD_SIDE_TEMP_K / HOT_SIDE_TEMP_K  # ~0.65

# Control
CONTROL_ROD_MIN = 0.0               # fully inserted (shutdown)
CONTROL_ROD_MAX = 1.0               # fully withdrawn (max power)
NOMINAL_ROD_POSITION = 0.7          # normal operating position
TEMP_COEFFICIENT = -0.002           # negative temp coefficient (per K above nominal)
NOMINAL_CORE_TEMP_K = 1073.0

# Startup/shutdown
STARTUP_SOLS = 5                    # sols to reach nominal from cold
SCRAM_COOLDOWN_SOLS = 3             # sols to cool after emergency shutdown

# Thermal management
RADIATOR_AREA_M2 = 20.0
RADIATOR_DEGRADATION_PER_SOL = 0.0001  # dust and micrometeorite wear
RADIATOR_MIN_EFFICIENCY = 0.5       # minimum before maintenance required
RADIATOR_REPAIR_PER_SOL = 0.02     # maintenance restoration rate

# Radiation
SHIELD_EFFECTIVENESS = 0.999        # fraction of radiation blocked
SAFE_DISTANCE_M = 100.0             # minimum hab-to-reactor distance

# Failure modes
STIRLING_FAILURE_PROB_PER_SOL = 0.0005  # per engine per sol
NUM_STIRLING_ENGINES = 8            # redundant engines
MIN_ENGINES_FOR_OPERATION = 4       # minimum for continued operation


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

REACTOR_STATES = ("shutdown", "startup", "nominal", "scram", "degraded")

@dataclass
class ReactorState:
    """Persistent state of the fission reactor."""
    sol: int = 0
    state: str = "shutdown"
    control_rod_position: float = CONTROL_ROD_MIN
    core_temp_k: float = 300.0          # ambient Mars temp at start
    thermal_power_kw: float = 0.0
    electric_power_kw: float = 0.0
    fuel_remaining_fraction: float = 1.0
    total_energy_kwh: float = 0.0
    radiator_efficiency: float = 1.0
    stirling_engines_active: int = NUM_STIRLING_ENGINES
    startup_sols_remaining: int = 0
    scram_cooldown_remaining: int = 0
    scram_count: int = 0
    total_operating_sols: int = 0
    operational: bool = True
    errors: list = field(default_factory=list)


@dataclass
class ReactorSol:
    """Record of one sol of reactor operations."""
    sol: int = 0
    state: str = "shutdown"
    thermal_power_kw: float = 0.0
    electric_power_kw: float = 0.0
    core_temp_k: float = 300.0
    fuel_remaining: float = 1.0
    radiator_efficiency: float = 1.0
    engines_active: int = NUM_STIRLING_ENGINES
    engine_failed: bool = False
    rod_position: float = 0.0


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def thermal_power(rod_position, fuel_fraction, core_temp_k):
    """Thermal power output in kW.

    Power = max_thermal * rod_position * fuel_fraction * temp_feedback
    Negative temperature coefficient: hotter core = less reactivity.
    """
    if rod_position <= 0 or fuel_fraction <= 0:
        return 0.0

    temp_delta = core_temp_k - NOMINAL_CORE_TEMP_K
    temp_feedback = 1.0 + TEMP_COEFFICIENT * max(temp_delta, 0.0)
    temp_feedback = max(temp_feedback, 0.0)

    power = THERMAL_POWER_MAX_KW * rod_position * fuel_fraction * temp_feedback
    return max(power, 0.0)


def electric_output(thermal_kw, engines_active, radiator_eff):
    """Electric power from Stirling conversion.

    Efficiency scales with active engine count and radiator health.
    Fewer engines = less conversion capacity.
    Poor radiator = higher cold-side temp = lower efficiency.
    """
    if thermal_kw <= 0 or engines_active <= 0:
        return 0.0

    engine_fraction = engines_active / NUM_STIRLING_ENGINES
    eff = STIRLING_EFFICIENCY * engine_fraction * radiator_eff
    return thermal_kw * max(eff, 0.0)


def fuel_burnup_per_sol(thermal_kw):
    """Fraction of fuel consumed per sol at given thermal power.

    Linear model: full power for FUEL_LIFE_SOLS depletes fuel completely.
    """
    if thermal_kw <= 0:
        return 0.0
    power_fraction = thermal_kw / THERMAL_POWER_MAX_KW
    return power_fraction / FUEL_LIFE_SOLS


def core_temperature(current_temp_k, thermal_kw, radiator_eff):
    """Core temperature evolution.

    Heating from fission, cooling from radiators.
    Equilibrium: thermal input = radiative output.
    Simple model: temp moves toward equilibrium over 1 sol.
    """
    if thermal_kw <= 0:
        # Cooling toward Mars ambient (~210 K average)
        target = 210.0
        rate = 0.1  # slow cooldown
    else:
        # Equilibrium temp based on power and radiator
        heat_rejection = radiator_eff * RADIATOR_AREA_M2 * 0.1  # simplified
        target = NOMINAL_CORE_TEMP_K + (thermal_kw - heat_rejection * 50) * 0.5
        target = max(target, 300.0)
        rate = 0.3  # faster approach to operating temp

    new_temp = current_temp_k + rate * (target - current_temp_k)
    return max(new_temp, 210.0)


def should_scram(core_temp_k, engines_active):
    """Check if emergency shutdown conditions are met.

    SCRAM triggers:
    - Core temp > 1200 K (overheating)
    - Fewer than MIN_ENGINES active (can't reject heat)
    """
    if core_temp_k > 1200.0:
        return True, "core overtemp (%.0f K)" % core_temp_k
    if engines_active < MIN_ENGINES_FOR_OPERATION:
        return True, "insufficient Stirling engines (%d/%d)" % (
            engines_active, MIN_ENGINES_FOR_OPERATION
        )
    return False, ""


# ---------------------------------------------------------------------------
# Tick function
# ---------------------------------------------------------------------------

def tick_reactor(state, command="hold", maintain_radiator=False, rng=None):
    """Advance the reactor by one sol.

    Parameters
    ----------
    state : ReactorState
        Current state (mutated in place).
    command : str
        "startup", "shutdown", "hold" (maintain current state).
    maintain_radiator : bool
        If True, crew is maintaining the radiator.
    rng : random.Random or None
        For reproducible tests.

    Returns
    -------
    (ReactorState, ReactorSol)
    """
    if rng is None:
        rng = random.Random()

    sol = ReactorSol(sol=state.sol)
    state.sol += 1
    sol.sol = state.sol

    if not state.operational:
        sol.state = state.state
        return state, sol

    # --- Radiator degradation/maintenance ---
    state.radiator_efficiency = max(
        state.radiator_efficiency - RADIATOR_DEGRADATION_PER_SOL,
        RADIATOR_MIN_EFFICIENCY,
    )
    if maintain_radiator:
        state.radiator_efficiency = min(
            state.radiator_efficiency + RADIATOR_REPAIR_PER_SOL, 1.0
        )
    sol.radiator_efficiency = state.radiator_efficiency

    # --- Stirling engine failures ---
    sol.engine_failed = False
    if state.stirling_engines_active > 0:
        for _ in range(state.stirling_engines_active):
            if rng.random() < STIRLING_FAILURE_PROB_PER_SOL:
                state.stirling_engines_active -= 1
                sol.engine_failed = True
                state.errors.append(
                    "Sol %d: Stirling engine failed (%d/%d remaining)"
                    % (state.sol, state.stirling_engines_active, NUM_STIRLING_ENGINES)
                )
    sol.engines_active = state.stirling_engines_active

    # --- State machine ---
    if state.state == "shutdown":
        if command == "startup" and state.fuel_remaining_fraction > 0:
            state.state = "startup"
            state.startup_sols_remaining = STARTUP_SOLS
            state.control_rod_position = 0.1  # begin withdrawal

    elif state.state == "startup":
        state.startup_sols_remaining -= 1
        state.control_rod_position = min(
            state.control_rod_position + NOMINAL_ROD_POSITION / STARTUP_SOLS,
            NOMINAL_ROD_POSITION,
        )
        if state.startup_sols_remaining <= 0:
            state.state = "nominal"
            state.control_rod_position = NOMINAL_ROD_POSITION

    elif state.state == "nominal":
        if command == "shutdown":
            state.state = "shutdown"
            state.control_rod_position = CONTROL_ROD_MIN
        else:
            # Check for SCRAM conditions
            needs_scram, reason = should_scram(
                state.core_temp_k, state.stirling_engines_active
            )
            if needs_scram:
                state.state = "scram"
                state.control_rod_position = CONTROL_ROD_MIN
                state.scram_cooldown_remaining = SCRAM_COOLDOWN_SOLS
                state.scram_count += 1
                state.errors.append("Sol %d: SCRAM — %s" % (state.sol, reason))
            else:
                state.total_operating_sols += 1

    elif state.state == "scram":
        state.scram_cooldown_remaining -= 1
        state.control_rod_position = CONTROL_ROD_MIN
        if state.scram_cooldown_remaining <= 0:
            state.state = "shutdown"

    elif state.state == "degraded":
        if command == "shutdown":
            state.state = "shutdown"
            state.control_rod_position = CONTROL_ROD_MIN

    # --- Physics ---
    th_power = thermal_power(
        state.control_rod_position,
        state.fuel_remaining_fraction,
        state.core_temp_k,
    )
    state.thermal_power_kw = th_power

    el_power = electric_output(
        th_power, state.stirling_engines_active, state.radiator_efficiency,
    )
    state.electric_power_kw = el_power

    # Fuel burnup
    burnup = fuel_burnup_per_sol(th_power)
    state.fuel_remaining_fraction = max(state.fuel_remaining_fraction - burnup, 0.0)
    if state.fuel_remaining_fraction <= 0:
        state.state = "shutdown"
        state.control_rod_position = CONTROL_ROD_MIN
        state.errors.append("Sol %d: fuel depleted" % state.sol)

    # Core temperature
    state.core_temp_k = core_temperature(
        state.core_temp_k, th_power, state.radiator_efficiency,
    )

    # Energy accumulation (kWh over 24.66 Mars hours)
    sol_hours = 24.66
    energy = el_power * sol_hours
    state.total_energy_kwh += energy

    # Fill sol record
    sol.state = state.state
    sol.thermal_power_kw = th_power
    sol.electric_power_kw = el_power
    sol.core_temp_k = state.core_temp_k
    sol.fuel_remaining = state.fuel_remaining_fraction
    sol.rod_position = state.control_rod_position

    return state, sol


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_reactor():
    """Create a new reactor in shutdown state."""
    return ReactorState()


# ---------------------------------------------------------------------------
# Multi-sol runner
# ---------------------------------------------------------------------------

def run_reactor(n_sols, auto_start=True, maintain_every_n=20, seed=None):
    """Run the reactor for n sols.

    Starts up automatically if auto_start is True.
    Crew maintains radiator every maintain_every_n sols.
    """
    rng = random.Random(seed)
    state = create_reactor()
    history = []

    for i in range(n_sols):
        if i == 0 and auto_start:
            command = "startup"
        else:
            command = "hold"

        maintain = (i > 0 and i % maintain_every_n == 0)

        state, sol_record = tick_reactor(
            state, command=command, maintain_radiator=maintain, rng=rng,
        )
        history.append(sol_record)

    return state, history
