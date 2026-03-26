"""regolith_brick.py -- Mars Regolith Microwave Sintering for Structural Bricks.

The colony can mine regolith (regolith_processor), extract iron
(ore_smelter), and fabricate parts (fabricator) -- but it cannot BUILD.
Every habitat wall, landing pad, and radiation berm requires structural
material.  Shipping concrete from Earth costs $54,000/kg.  The answer
is under our boots: Martian regolith sintered into bricks.

Microwave sintering at 2.45 GHz heats regolith to 1000-1200 C,
bonding grains without full melting.  The result is a ceramic-like
brick with 20-50 MPa compressive strength -- enough for load-bearing
walls under Mars gravity (3.72 m/s^2).

Physics modelled
----------------
* Microwave absorption -- volumetric heating, Beer-Lambert law:
  P_abs = P_in * (1 - exp(-alpha * d)).  Martian regolith (high Fe2O3)
  absorbs 2.45 GHz well: alpha ~ 15-25 /m.

* Temperature rise -- dT = P_abs * dt / (mass * c_p).  Specific heat
  of basaltic regolith ~ 800 J/(kg*K).

* Sintering densification -- Arrhenius kinetics: rate ~ A * exp(-Ea/RT).
  Porosity decreases from ~0.40 (loose) toward ~0.05 (fully dense).

* Compressive strength -- Ryshkewitch equation:
  sigma = sigma_0 * exp(-b * porosity).  sigma_0 ~ 80 MPa (fully
  dense basalt), b ~ 4.0.

* Thermal radiation loss -- Stefan-Boltzmann from kiln surface.

* Cooling -- Newton's law in thin Mars atmosphere (~600 Pa CO2).
  Convective coefficient ~ 0.5 W/(m^2*K) (nearly vacuum).

* Mold wear -- linear degradation per firing cycle.

* Thermal shock cracking -- if cooling rate > threshold, brick cracks
  and strength drops by 60%.

Conservation laws: mass_in == mass_out, energy >= 0, 0 <= porosity <= 1,
strength >= 0, temperature >= ambient, mold_health in [0, 1].

One tick = one sol.  Multiple bricks per sol if cycle time permits.
Power in kW, mass in kg, temperature in K, dimensions in metres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# -- Physical constants -------------------------------------------------------

MARS_AMBIENT_TEMP_K = 210.0
MARS_GRAVITY_M_S2 = 3.72
MARS_ATMO_PRESSURE_PA = 610.0

STEFAN_BOLTZMANN = 5.670374419e-8
GAS_CONSTANT = 8.314

REGOLITH_DENSITY_LOOSE_KG_M3 = 1500.0
REGOLITH_DENSITY_DENSE_KG_M3 = 2800.0
REGOLITH_SPECIFIC_HEAT_J_KG_K = 800.0
REGOLITH_ABSORPTION_COEFF_M = 20.0  # Beer-Lambert alpha at 2.45 GHz

MICROWAVE_FREQ_GHZ = 2.45
DEFAULT_MICROWAVE_POWER_KW = 5.0
DEFAULT_MAGNETRON_EFFICIENCY = 0.70

SINTERING_TARGET_TEMP_K = 1373.0  # ~1100 C
SINTERING_ACTIVATION_ENERGY_J_MOL = 250_000.0  # basaltic sintering
SINTERING_PRE_EXPONENTIAL = 1.0e6
INITIAL_POROSITY = 0.40
MIN_POROSITY = 0.02

RYSHKEWITCH_SIGMA_0_MPA = 80.0  # fully dense basalt
RYSHKEWITCH_B = 4.0

BRICK_LENGTH_M = 0.20
BRICK_WIDTH_M = 0.10
BRICK_HEIGHT_M = 0.10
BRICK_VOLUME_M3 = BRICK_LENGTH_M * BRICK_WIDTH_M * BRICK_HEIGHT_M

KILN_SURFACE_AREA_M2 = 0.50
KILN_EMISSIVITY = 0.85
KILN_INSULATION_FACTOR = 0.20  # fraction of radiation that escapes

CONVECTIVE_COEFF_W_M2_K = 0.5  # near-vacuum Mars atmosphere
THERMAL_SHOCK_RATE_K_PER_MIN = 50.0  # max safe cooling rate
THERMAL_SHOCK_STRENGTH_FACTOR = 0.40  # brick retains 40% strength if cracked

MOLD_WEAR_PER_CYCLE = 0.001
MOLD_REPLACEMENT_THRESHOLD = 0.10

SOAK_TIME_HOURS = 2.0
HEATING_RATE_K_PER_MIN = 5.0  # controlled ramp
COOLING_RATE_K_PER_MIN = 3.0  # natural cooling in kiln

HOURS_PER_SOL = 24.66
SECONDS_PER_SOL = HOURS_PER_SOL * 3600.0
MINUTES_PER_SOL = HOURS_PER_SOL * 60.0

MIN_STRENGTH_MPA_STRUCTURAL = 10.0  # minimum for load-bearing


# -- Pure physics functions ---------------------------------------------------

def microwave_absorption_fraction(thickness_m: float,
                                  alpha: float = REGOLITH_ABSORPTION_COEFF_M) -> float:
    """Fraction of microwave power absorbed by regolith of given thickness."""
    thickness_m = max(0.0, thickness_m)
    alpha = max(0.0, alpha)
    return 1.0 - math.exp(-alpha * thickness_m)


def absorbed_power_kw(input_power_kw: float, thickness_m: float,
                      magnetron_efficiency: float = DEFAULT_MAGNETRON_EFFICIENCY) -> float:
    """Net microwave power absorbed by the regolith brick (kW)."""
    input_power_kw = max(0.0, input_power_kw)
    magnetron_efficiency = max(0.0, min(1.0, magnetron_efficiency))
    frac = microwave_absorption_fraction(thickness_m)
    return input_power_kw * magnetron_efficiency * frac


def temperature_rise_k(power_kw: float, time_s: float,
                       mass_kg: float,
                       specific_heat: float = REGOLITH_SPECIFIC_HEAT_J_KG_K) -> float:
    """Temperature rise from absorbed power over time (K)."""
    if mass_kg <= 0.0 or specific_heat <= 0.0:
        return 0.0
    power_kw = max(0.0, power_kw)
    time_s = max(0.0, time_s)
    return (power_kw * 1000.0 * time_s) / (mass_kg * specific_heat)


def radiation_loss_kw(surface_temp_k: float,
                      ambient_temp_k: float = MARS_AMBIENT_TEMP_K) -> float:
    """Thermal radiation loss from kiln surface (kW)."""
    surface_temp_k = max(0.0, surface_temp_k)
    ambient_temp_k = max(0.0, ambient_temp_k)
    q = (STEFAN_BOLTZMANN * KILN_EMISSIVITY * KILN_SURFACE_AREA_M2
         * KILN_INSULATION_FACTOR
         * (surface_temp_k**4 - ambient_temp_k**4))
    return max(0.0, q / 1000.0)


def convective_loss_kw(surface_temp_k: float,
                       ambient_temp_k: float = MARS_AMBIENT_TEMP_K) -> float:
    """Convective heat loss in thin Mars atmosphere (kW)."""
    surface_temp_k = max(0.0, surface_temp_k)
    ambient_temp_k = max(0.0, ambient_temp_k)
    q = (CONVECTIVE_COEFF_W_M2_K * KILN_SURFACE_AREA_M2
         * max(0.0, surface_temp_k - ambient_temp_k))
    return q / 1000.0


def sintering_rate(temperature_k: float) -> float:
    """Arrhenius sintering densification rate (dimensionless per second).

    Returns the instantaneous fractional porosity reduction rate.
    """
    if temperature_k <= 0.0:
        return 0.0
    exponent = -SINTERING_ACTIVATION_ENERGY_J_MOL / (GAS_CONSTANT * temperature_k)
    exponent = max(-500.0, exponent)  # prevent underflow
    return SINTERING_PRE_EXPONENTIAL * math.exp(exponent)


def densify_porosity(current_porosity: float, temperature_k: float,
                     time_s: float) -> float:
    """Reduce porosity via sintering kinetics over given time.

    Uses forward-Euler on: dp/dt = -rate * p * (p - p_min)
    """
    current_porosity = max(MIN_POROSITY, min(1.0, current_porosity))
    if time_s <= 0.0 or temperature_k <= 0.0:
        return current_porosity
    rate = sintering_rate(temperature_k)
    dp = rate * current_porosity * (current_porosity - MIN_POROSITY) * time_s
    new_porosity = current_porosity - dp
    return max(MIN_POROSITY, min(current_porosity, new_porosity))


def compressive_strength_mpa(porosity: float) -> float:
    """Ryshkewitch equation: strength vs porosity (MPa)."""
    porosity = max(0.0, min(1.0, porosity))
    return RYSHKEWITCH_SIGMA_0_MPA * math.exp(-RYSHKEWITCH_B * porosity)


def brick_mass_kg(porosity: float) -> float:
    """Mass of one brick at given porosity (kg)."""
    porosity = max(0.0, min(1.0, porosity))
    density = REGOLITH_DENSITY_DENSE_KG_M3 * (1.0 - porosity)
    return density * BRICK_VOLUME_M3


def regolith_needed_kg(porosity: float = INITIAL_POROSITY) -> float:
    """Regolith mass needed to fill one brick mold at initial porosity."""
    return brick_mass_kg(porosity)


def heating_time_minutes(target_temp_k: float,
                         ambient_temp_k: float = MARS_AMBIENT_TEMP_K) -> float:
    """Time to ramp kiln from ambient to target at controlled rate."""
    delta = max(0.0, target_temp_k - ambient_temp_k)
    if HEATING_RATE_K_PER_MIN <= 0.0:
        return 0.0
    return delta / HEATING_RATE_K_PER_MIN


def cooling_time_minutes(kiln_temp_k: float,
                         safe_handling_temp_k: float = 350.0,
                         ambient_temp_k: float = MARS_AMBIENT_TEMP_K) -> float:
    """Time to cool brick from kiln temp to safe handling temperature."""
    delta = max(0.0, kiln_temp_k - max(safe_handling_temp_k, ambient_temp_k))
    if COOLING_RATE_K_PER_MIN <= 0.0:
        return 0.0
    return delta / COOLING_RATE_K_PER_MIN


def cycle_time_minutes(target_temp_k: float = SINTERING_TARGET_TEMP_K) -> float:
    """Total time for one brick: heat + soak + cool (minutes)."""
    heat = heating_time_minutes(target_temp_k)
    soak = SOAK_TIME_HOURS * 60.0
    cool = cooling_time_minutes(target_temp_k)
    return heat + soak + cool


def bricks_per_sol(target_temp_k: float = SINTERING_TARGET_TEMP_K) -> int:
    """Maximum bricks producible in one sol."""
    ct = cycle_time_minutes(target_temp_k)
    if ct <= 0.0:
        return 0
    return max(1, int(MINUTES_PER_SOL / ct))


def is_thermally_shocked(cooling_rate_k_per_min: float) -> bool:
    """Return True if cooling is too fast, risking thermal shock cracks."""
    return cooling_rate_k_per_min > THERMAL_SHOCK_RATE_K_PER_MIN


def energy_per_brick_kwh(power_kw: float, target_temp_k: float = SINTERING_TARGET_TEMP_K) -> float:
    """Total energy for one brick (kWh): heating + soak phases."""
    heat_min = heating_time_minutes(target_temp_k)
    soak_min = SOAK_TIME_HOURS * 60.0
    total_hours = (heat_min + soak_min) / 60.0
    return max(0.0, power_kw * total_hours)


def apply_mold_wear(health: float, n_cycles: int = 1) -> float:
    """Degrade mold health by n firing cycles."""
    return max(0.0, health - MOLD_WEAR_PER_CYCLE * n_cycles)


# -- State dataclass ----------------------------------------------------------

@dataclass
class BrickKiln:
    """State of the microwave brick sintering kiln."""

    sol: int = 0
    kiln_temp_k: float = MARS_AMBIENT_TEMP_K
    microwave_power_kw: float = DEFAULT_MICROWAVE_POWER_KW
    magnetron_efficiency: float = DEFAULT_MAGNETRON_EFFICIENCY
    sintering_target_k: float = SINTERING_TARGET_TEMP_K
    mold_health: float = 1.0
    regolith_feed_kg_per_sol: float = 100.0

    # cumulative production
    total_bricks: int = 0
    total_structural_bricks: int = 0
    total_cracked_bricks: int = 0
    total_regolith_consumed_kg: float = 0.0
    total_energy_kwh: float = 0.0
    best_strength_mpa: float = 0.0
    worst_strength_mpa: float = 999.0

    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sol": self.sol,
            "kiln_temp_k": self.kiln_temp_k,
            "microwave_power_kw": self.microwave_power_kw,
            "magnetron_efficiency": self.magnetron_efficiency,
            "sintering_target_k": self.sintering_target_k,
            "mold_health": self.mold_health,
            "regolith_feed_kg_per_sol": self.regolith_feed_kg_per_sol,
            "total_bricks": self.total_bricks,
            "total_structural_bricks": self.total_structural_bricks,
            "total_cracked_bricks": self.total_cracked_bricks,
            "total_regolith_consumed_kg": self.total_regolith_consumed_kg,
            "total_energy_kwh": self.total_energy_kwh,
            "best_strength_mpa": self.best_strength_mpa,
            "worst_strength_mpa": self.worst_strength_mpa,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BrickKiln:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TickResult:
    """Result of one sol of brick production."""

    sol: int = 0
    bricks_produced: int = 0
    structural_bricks: int = 0
    cracked_bricks: int = 0
    avg_porosity: float = INITIAL_POROSITY
    avg_strength_mpa: float = 0.0
    regolith_consumed_kg: float = 0.0
    energy_consumed_kwh: float = 0.0
    power_draw_kw: float = 0.0
    radiation_loss_kw: float = 0.0
    mold_health: float = 1.0
    operational: bool = True
    events: list[str] = field(default_factory=list)


# -- Simulation ---------------------------------------------------------------

def tick(state: BrickKiln) -> TickResult:
    """Advance the brick kiln by one sol.

    Produces as many bricks as cycle time allows.  Each brick goes through:
    heating ramp -> soak at sintering temp -> controlled cooling.
    """
    state.sol += 1
    events: list[str] = []

    # -- Check operational constraints ----------------------------------------
    operational = True

    if state.mold_health <= MOLD_REPLACEMENT_THRESHOLD:
        events.append("MOLD WORN -- replacement needed")
        operational = False

    if state.microwave_power_kw <= 0.0:
        events.append("NO POWER -- microwave offline")
        operational = False

    if state.regolith_feed_kg_per_sol <= 0.0:
        events.append("NO FEEDSTOCK -- regolith supply empty")
        operational = False

    if not operational:
        state.events = events
        return TickResult(sol=state.sol, operational=False,
                          mold_health=state.mold_health, events=list(events))

    # -- Compute how many bricks fit in one sol -------------------------------
    max_bricks = bricks_per_sol(state.sintering_target_k)
    mass_per_brick = regolith_needed_kg()
    max_from_feed = int(state.regolith_feed_kg_per_sol / mass_per_brick) if mass_per_brick > 0 else 0
    n_bricks = min(max_bricks, max_from_feed)

    if n_bricks <= 0:
        events.append("INSUFFICIENT FEED -- not enough regolith for one brick")
        state.events = events
        return TickResult(sol=state.sol, operational=True,
                          mold_health=state.mold_health, events=list(events))

    # -- Simulate each brick --------------------------------------------------
    total_porosity = 0.0
    total_strength = 0.0
    structural_count = 0
    cracked_count = 0

    abs_power = absorbed_power_kw(state.microwave_power_kw, BRICK_HEIGHT_M,
                                  state.magnetron_efficiency)

    soak_seconds = SOAK_TIME_HOURS * 3600.0

    for _ in range(n_bricks):
        # Heating phase: ramp to target
        state.kiln_temp_k = state.sintering_target_k

        # Sintering during soak
        porosity = densify_porosity(INITIAL_POROSITY, state.sintering_target_k,
                                    soak_seconds)

        # Cooling phase
        actual_cooling_rate = COOLING_RATE_K_PER_MIN
        shocked = is_thermally_shocked(actual_cooling_rate)

        strength = compressive_strength_mpa(porosity)
        if shocked:
            strength *= THERMAL_SHOCK_STRENGTH_FACTOR
            cracked_count += 1

        if strength >= MIN_STRENGTH_MPA_STRUCTURAL:
            structural_count += 1

        total_porosity += porosity
        total_strength += strength

        state.kiln_temp_k = MARS_AMBIENT_TEMP_K  # cooled down

    # -- Energy accounting ----------------------------------------------------
    energy_per = energy_per_brick_kwh(state.microwave_power_kw,
                                      state.sintering_target_k)
    total_energy = energy_per * n_bricks
    rad_loss = radiation_loss_kw(state.sintering_target_k)

    regolith_used = n_bricks * mass_per_brick

    # -- Mold wear ------------------------------------------------------------
    old_health = state.mold_health
    state.mold_health = apply_mold_wear(state.mold_health, n_bricks)
    if old_health >= 0.50 and state.mold_health < 0.50:
        events.append("MOLD WARNING -- health below 50%")
    if (old_health > MOLD_REPLACEMENT_THRESHOLD
            and state.mold_health <= MOLD_REPLACEMENT_THRESHOLD):
        events.append("MOLD CRITICAL -- replacement needed next sol")

    # -- Update cumulative state ----------------------------------------------
    avg_porosity = total_porosity / n_bricks
    avg_strength = total_strength / n_bricks

    state.total_bricks += n_bricks
    state.total_structural_bricks += structural_count
    state.total_cracked_bricks += cracked_count
    state.total_regolith_consumed_kg += regolith_used
    state.total_energy_kwh += total_energy
    state.best_strength_mpa = max(state.best_strength_mpa, avg_strength)
    if avg_strength > 0:
        state.worst_strength_mpa = min(state.worst_strength_mpa, avg_strength)

    if structural_count == n_bricks:
        events.append(f"FULL YIELD -- all {n_bricks} bricks structural grade")
    elif structural_count == 0:
        events.append("ZERO YIELD -- no structural-grade bricks this sol")

    state.events = events

    return TickResult(
        sol=state.sol,
        bricks_produced=n_bricks,
        structural_bricks=structural_count,
        cracked_bricks=cracked_count,
        avg_porosity=avg_porosity,
        avg_strength_mpa=avg_strength,
        regolith_consumed_kg=regolith_used,
        energy_consumed_kwh=total_energy,
        power_draw_kw=state.microwave_power_kw,
        radiation_loss_kw=rad_loss,
        mold_health=state.mold_health,
        operational=True,
        events=list(events),
    )


def run_simulation(sols: int = 365,
                   power_kw: float = DEFAULT_MICROWAVE_POWER_KW,
                   feed_rate_kg: float = 100.0) -> list[TickResult]:
    """Run the brick kiln for multiple sols."""
    state = BrickKiln(microwave_power_kw=power_kw,
                      regolith_feed_kg_per_sol=feed_rate_kg)
    return [tick(state) for _ in range(sols)]


if __name__ == "__main__":
    import json
    import sys

    sols = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    results = run_simulation(sols=sols)
    total_bricks = sum(r.bricks_produced for r in results)
    structural = sum(r.structural_bricks for r in results)
    cracked = sum(r.cracked_bricks for r in results)
    energy = sum(r.energy_consumed_kwh for r in results)
    regolith = sum(r.regolith_consumed_kg for r in results)
    print(f"Mars Brick Kiln -- {sols} sols")
    print(f"  Total bricks: {total_bricks} | Structural: {structural} | Cracked: {cracked}")
    print(f"  Regolith consumed: {regolith:.1f} kg")
    print(f"  Energy consumed: {energy:.1f} kWh")
    print(f"  Mold health: {results[-1].mold_health:.2%}")
    print(f"  Avg strength: {results[-1].avg_strength_mpa:.1f} MPa")
