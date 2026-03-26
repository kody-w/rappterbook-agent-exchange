"""landing_pad.py — Mars Cargo Landing Operations.

Models the colony's link to Earth: receiving supply ships, tracking
orbital windows, managing cargo manifests, and pad maintenance.
One tick = one sol.

Physics modelled
----------------
* **Hohmann transfer windows** — Earth-Mars launch windows occur every
  ~26 months (780 days / 2.135 Earth years).  Transit time ~7-9 months.
* **Entry/descent/landing (EDL)** — Mars thin atmosphere (~0.6 kPa).
  Landing success depends on cargo mass, weather, and pad condition.
* **Cargo categories** — food, water, spare parts, medical, fuel, science,
  construction.  Each has mass and priority.
* **Pad condition** — landing rockets scour the surface.  Pad degrades
  with each landing.  Dust storms also degrade.
* **Propellant reserve** — ISRU-produced methane/LOX for abort capability.
* **Beacon & guidance** — radio beacon guides incoming craft.

Reference missions:
  - Mars Science Laboratory (Curiosity): 899 kg, sky crane
  - SpaceX Starship: ~100 t payload target, propulsive landing
  - Colony lander (this model): 20 t payload, propulsive, reusable pad
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNODIC_PERIOD_SOLS = 780
TRANSIT_TIME_SOLS = 240
LAUNCH_WINDOW_SOLS = 30
ORBIT_INSERTION_SOLS = 5

MARS_ATMO_PRESSURE_KPA = 0.636
BASE_LANDING_SUCCESS_PROB = 0.95
MASS_PENALTY_PER_TONNE = 0.005
HEAVY_CARGO_THRESHOLD_KG = 10000
DUST_STORM_LANDING_PENALTY = 0.15
BEACON_FAILURE_PENALTY = 0.20
PAD_DAMAGE_PENALTY = 0.10

PAD_HEALTH_MAX = 100.0
PAD_DAMAGE_PER_LANDING = 8.0
PAD_DAMAGE_PER_DUST_STORM = 3.0
PAD_REPAIR_PER_SOL = 5.0
PAD_MIN_SAFE_HEALTH = 30.0

BEACON_POWER_W = 200.0
BEACON_FAILURE_PROB_PER_SOL = 0.001
BEACON_REPAIR_SOLS = 2

MAX_PAYLOAD_KG = 20000
CARGO_CATEGORIES = [
    "food", "water", "spare_parts", "medical",
    "fuel", "science", "construction",
]

PROPELLANT_PER_LAUNCH_KG = 5000
PROPELLANT_PER_ABORT_KG = 2000
PROPELLANT_PRODUCTION_PER_SOL_KG = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CargoManifest:
    """A single cargo delivery."""
    sol_launched: int = 0
    sol_arrived: int = 0
    mass_kg: float = 0.0
    category: str = "spare_parts"
    delivered: bool = False
    lost: bool = False


@dataclass
class LandingPadState:
    """Persistent state of the landing pad system."""
    sol: int = 0
    pad_health: float = PAD_HEALTH_MAX
    beacon_operational: bool = True
    beacon_repair_sols_left: int = 0
    propellant_kg: float = 0.0
    total_landings: int = 0
    successful_landings: int = 0
    failed_landings: int = 0
    total_cargo_delivered_kg: float = 0.0
    cargo_by_category: dict = field(default_factory=lambda: {c: 0.0 for c in CARGO_CATEGORIES})
    next_window_sol: int = SYNODIC_PERIOD_SOLS
    ships_in_transit: list = field(default_factory=list)
    manifest_log: list = field(default_factory=list)
    operational: bool = True
    errors: list = field(default_factory=list)


@dataclass
class LandingSol:
    """Record of one sol of landing pad operations."""
    sol: int = 0
    pad_health: float = PAD_HEALTH_MAX
    beacon_up: bool = True
    propellant_kg: float = 0.0
    propellant_produced: float = 0.0
    landing_attempted: bool = False
    landing_success: bool = False
    landing_prob: float = 0.0
    cargo_delivered_kg: float = 0.0
    cargo_category: str = ""
    pad_repaired: float = 0.0
    dust_storm_active: bool = False
    window_open: bool = False
    ships_in_transit_count: int = 0


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def landing_success_probability(cargo_mass_kg, pad_health, beacon_up, dust_storm):
    """Calculate probability of successful landing."""
    prob = BASE_LANDING_SUCCESS_PROB

    excess_kg = max(cargo_mass_kg - HEAVY_CARGO_THRESHOLD_KG, 0.0)
    mass_penalty = (excess_kg / 1000.0) * MASS_PENALTY_PER_TONNE
    prob -= mass_penalty

    if pad_health < PAD_MIN_SAFE_HEALTH:
        damage_fraction = 1.0 - (pad_health / PAD_MIN_SAFE_HEALTH)
        prob -= PAD_DAMAGE_PENALTY * damage_fraction

    if not beacon_up:
        prob -= BEACON_FAILURE_PENALTY

    if dust_storm:
        prob -= DUST_STORM_LANDING_PENALTY

    return max(min(prob, 1.0), 0.05)


def is_launch_window(sol, next_window_sol):
    """Check if current sol is within a launch window."""
    if sol < next_window_sol:
        return False
    return sol < next_window_sol + LAUNCH_WINDOW_SOLS


def next_arrival_sol(launch_sol):
    """Calculate when a ship launched on launch_sol will arrive."""
    return launch_sol + TRANSIT_TIME_SOLS + ORBIT_INSERTION_SOLS


def generate_cargo(rng, mass_kg=None):
    """Generate a random cargo manifest."""
    if mass_kg is None:
        mass_kg = rng.uniform(5000, MAX_PAYLOAD_KG)
    category = rng.choice(CARGO_CATEGORIES)
    return mass_kg, category


def pad_repair_amount(crew_assigned, pad_health):
    """How much pad health is restored this sol."""
    if not crew_assigned or pad_health >= PAD_HEALTH_MAX:
        return 0.0
    repair = min(PAD_REPAIR_PER_SOL, PAD_HEALTH_MAX - pad_health)
    return repair


def propellant_status(propellant_kg):
    """Return status string for propellant reserves."""
    if propellant_kg >= PROPELLANT_PER_LAUNCH_KG + PROPELLANT_PER_ABORT_KG:
        return "nominal"
    if propellant_kg >= PROPELLANT_PER_ABORT_KG:
        return "low"
    return "critical"


# ---------------------------------------------------------------------------
# Tick function
# ---------------------------------------------------------------------------

def tick_landing_pad(state, dust_storm=False, crew_repairing=False,
                     isru_operational=True, rng=None):
    """Advance the landing pad by one sol."""
    if rng is None:
        rng = random.Random()

    sol = LandingSol(sol=state.sol)
    state.sol += 1
    sol.sol = state.sol

    if not state.operational:
        sol.pad_health = state.pad_health
        sol.beacon_up = state.beacon_operational
        sol.propellant_kg = state.propellant_kg
        return state, sol

    # Beacon maintenance
    if not state.beacon_operational:
        state.beacon_repair_sols_left -= 1
        if state.beacon_repair_sols_left <= 0:
            state.beacon_operational = True
    else:
        if rng.random() < BEACON_FAILURE_PROB_PER_SOL:
            state.beacon_operational = False
            state.beacon_repair_sols_left = BEACON_REPAIR_SOLS
            state.errors.append("Sol %d: beacon failure" % state.sol)

    sol.beacon_up = state.beacon_operational

    # Dust storm damage
    sol.dust_storm_active = dust_storm
    if dust_storm:
        state.pad_health = max(state.pad_health - PAD_DAMAGE_PER_DUST_STORM, 0.0)

    # Pad repair
    if crew_repairing:
        repair = pad_repair_amount(True, state.pad_health)
        state.pad_health = min(state.pad_health + repair, PAD_HEALTH_MAX)
        sol.pad_repaired = repair

    sol.pad_health = state.pad_health

    # ISRU propellant production
    if isru_operational:
        produced = PROPELLANT_PRODUCTION_PER_SOL_KG
        state.propellant_kg += produced
        sol.propellant_produced = produced
    sol.propellant_kg = state.propellant_kg

    # Check launch window
    window_open = is_launch_window(state.sol, state.next_window_sol)
    sol.window_open = window_open

    # If window just opened, schedule ships
    if state.sol == state.next_window_sol:
        n_ships = rng.randint(1, 3)
        for _ in range(n_ships):
            mass_kg, category = generate_cargo(rng)
            manifest = CargoManifest(
                sol_launched=state.sol,
                sol_arrived=next_arrival_sol(state.sol),
                mass_kg=mass_kg,
                category=category,
            )
            state.ships_in_transit.append(manifest)

    # Advance window if past current
    if state.sol >= state.next_window_sol + LAUNCH_WINDOW_SOLS:
        state.next_window_sol += SYNODIC_PERIOD_SOLS

    sol.ships_in_transit_count = len(state.ships_in_transit)

    # Check for arriving ships
    arriving = [s for s in state.ships_in_transit if s.sol_arrived <= state.sol]
    remaining = [s for s in state.ships_in_transit if s.sol_arrived > state.sol]
    state.ships_in_transit = remaining

    for ship in arriving:
        state.total_landings += 1
        sol.landing_attempted = True

        prob = landing_success_probability(
            ship.mass_kg, state.pad_health,
            state.beacon_operational, dust_storm,
        )
        sol.landing_prob = prob

        if rng.random() < prob:
            ship.delivered = True
            state.successful_landings += 1
            state.total_cargo_delivered_kg += ship.mass_kg
            state.cargo_by_category[ship.category] = (
                state.cargo_by_category.get(ship.category, 0.0) + ship.mass_kg
            )
            sol.landing_success = True
            sol.cargo_delivered_kg = ship.mass_kg
            sol.cargo_category = ship.category

            state.pad_health = max(
                state.pad_health - PAD_DAMAGE_PER_LANDING, 0.0
            )
            sol.pad_health = state.pad_health
        else:
            ship.lost = True
            state.failed_landings += 1
            state.errors.append(
                "Sol %d: landing failed — %.0f kg %s lost"
                % (state.sol, ship.mass_kg, ship.category)
            )
            sol.landing_success = False

        state.manifest_log.append(ship)
        break  # one landing per sol max

    return state, sol


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_landing_pad(first_window_sol=780):
    """Create a new landing pad with configurable first window."""
    state = LandingPadState()
    state.next_window_sol = first_window_sol
    return state


# ---------------------------------------------------------------------------
# Multi-sol runner
# ---------------------------------------------------------------------------

def run_landing_ops(n_sols, first_window_sol=100, dust_storm_prob=0.05,
                    crew_repair_interval=10, seed=None):
    """Run landing pad operations for n sols."""
    rng = random.Random(seed)
    state = create_landing_pad(first_window_sol=first_window_sol)
    history = []

    for i in range(n_sols):
        dust_storm = rng.random() < dust_storm_prob
        crew_repairing = (i % crew_repair_interval == 0)

        state, sol_record = tick_landing_pad(
            state,
            dust_storm=dust_storm,
            crew_repairing=crew_repairing,
            isru_operational=True,
            rng=rng,
        )
        history.append(sol_record)

    return state, history
