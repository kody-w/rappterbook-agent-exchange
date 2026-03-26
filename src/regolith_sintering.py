"""regolith_sintering.py — Mars Regolith Sintering Kiln.

The colony can mine regolith and smelt ore — but it can't build.
Every habitat module was shipped from Earth at $2M/kg.  To grow
beyond pre-fabricated cans, Mars must make its own bricks.

Sintering heats regolith to 1000-1100°C without melting it.  At that
temperature, iron-oxide-rich basalt particles fuse at grain boundaries
into a solid ceramic brick.  No binder.  No water.  Just heat and
pressure.  The result: compressive-strength bricks from Martian dirt.

Physics modelled
----------------
* **Kiln thermal model** — Energy input heats the charge from ambient
  (~210 K) to sintering temperature (~1300 K).  Losses through kiln
  walls by conduction, radiation from the opening, and the thin Mars
  atmosphere provides negligible convection.

* **Sintering kinetics** — Densification follows an Arrhenius rate:
  rate = A * exp(-Ea / (R * T)).  Higher temperature = faster sintering
  but higher energy cost.  Optimal soak time balances strength vs energy.

* **Brick strength** — Compressive strength depends on final density
  (fraction of theoretical).  Empirical fit from lunar regolith
  simulant studies: strength_MPa = k * (density_frac)^n.

* **Cooling schedule** — Too-fast cooling cracks bricks (thermal
  shock).  Cooling rate limited to 2 K/min.  Mars ambient is the
  heat sink.  Controlled cooling adds time but saves bricks.

* **Feedstock quality** — Iron oxide content (FeO + Fe2O3) in Mars
  regolith varies 15-20%.  Higher iron = lower sintering temp = less
  energy.  Feedstock is pre-screened by regolith_processor.

* **Energy budget** — Kiln powered by electricity (nuclear or solar).
  Energy per brick = mass * Cp * ΔT + losses.  ~2.5 kWh per 5 kg brick.

* **Dust contamination** — Fine dust (<10 μm) in feedstock reduces
  packing density.  Pre-sieving improves final strength by ~20%.

Conservation laws:
  - Energy in >= energy stored in brick + losses (first law)
  - Brick mass <= feedstock mass (mass conservation)
  - Temperature >= ambient (second law)
  - Strength >= 0, density fraction in [0, 1]
  - Cooling rate <= max safe rate (or brick cracks)

One tick = one sol.  Temperatures in Kelvin, energy in kWh, mass in kg.

Reference hardware:
  - ESA URBAN lunar brick studies (2019)
  - NASA Swamp Works regolith sintering (2020)
  - Barmac VSI crusher + muffle kiln concept (MIT, 2022)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MARS_AMBIENT_K = 210.0              # average Mars surface temperature
MARS_SOL_SECONDS = 88_775.0         # one sol in seconds
MARS_SOL_HOURS = 24.66              # hours per sol
BOLTZMANN = 5.670_374_419e-8        # Stefan-Boltzmann (W/m²/K⁴)
GAS_CONSTANT = 8.314                # J/(mol·K)

# Regolith properties (Mars basaltic regolith)
REGOLITH_CP_J_KG_K = 800.0         # specific heat capacity
REGOLITH_DENSITY_KG_M3 = 1_500.0   # bulk density (loose)
SINTERED_DENSITY_KG_M3 = 2_800.0   # theoretical full density (basalt)
REGOLITH_IRON_OXIDE_FRAC = 0.18    # typical FeO + Fe2O3 fraction

# Sintering parameters
SINTER_TEMP_BASE_K = 1_373.0       # base sintering temp (1100°C)
IRON_OXIDE_TEMP_REDUCTION_K = 500.0 # max reduction from high iron content
ACTIVATION_ENERGY_J_MOL = 250_000.0 # Ea for sintering (basalt grain diffusion)
ARRHENIUS_PREFACTOR = 1.0e10        # pre-exponential factor (1/s)
SOAK_TIME_HOURS = 4.0              # default soak time at sintering temp

# Brick specs
BRICK_MASS_KG = 5.0                # standard brick mass
BRICK_VOLUME_M3 = BRICK_MASS_KG / SINTERED_DENSITY_KG_M3
STRENGTH_COEFF_MPA = 120.0         # empirical coefficient
STRENGTH_EXPONENT = 3.5            # empirical exponent

# Kiln specs
KILN_CAPACITY_KG = 50.0            # max charge per batch (10 bricks)
KILN_WALL_THICKNESS_M = 0.15       # insulating refractory thickness
KILN_WALL_CONDUCTIVITY_W_MK = 0.3  # regolith-block insulation
KILN_INNER_AREA_M2 = 1.2           # inner surface area
KILN_OPENING_AREA_M2 = 0.05        # radiation loss through opening
KILN_EMISSIVITY = 0.85             # inner wall emissivity
KILN_HEATER_MAX_KW = 8.0           # electric heating element max power

# Cooling
MAX_COOLING_RATE_K_PER_MIN = 2.0   # above this → thermal shock cracks
CRACK_PROBABILITY_BASE = 0.02      # base probability per brick per batch

# Dust contamination
FINE_DUST_STRENGTH_PENALTY = 0.20  # 20% strength reduction if unsieved


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class KilnState:
    """Mutable state for one sintering kiln, advanced each sol."""

    # Kiln thermal state
    kiln_temp_k: float = MARS_AMBIENT_K
    charge_mass_kg: float = 0.0
    charge_temp_k: float = MARS_AMBIENT_K
    iron_oxide_frac: float = REGOLITH_IRON_OXIDE_FRAC
    feedstock_sieved: bool = True

    # Operating mode
    phase: str = "idle"         # idle | heating | soaking | cooling | done
    phase_hours: float = 0.0    # hours spent in current phase
    target_temp_k: float = 0.0  # computed sintering target

    # Energy tracking
    energy_input_kwh: float = 0.0   # energy consumed this batch
    energy_total_kwh: float = 0.0   # lifetime energy consumed

    # Production tracking
    bricks_this_batch: int = 0
    bricks_cracked: int = 0
    bricks_produced_total: int = 0
    bricks_cracked_total: int = 0
    batches_completed: int = 0

    # Current batch output
    density_fraction: float = 0.0
    compressive_strength_mpa: float = 0.0

    # Sol counter
    sol: int = 0

    # Per-sol log
    log: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def sintering_temperature(iron_oxide_frac: float) -> float:
    """Compute optimal sintering temperature based on iron oxide content.

    Higher iron oxide lowers the sintering temperature (acts as flux).
    Returns temperature in Kelvin.
    """
    frac_clamped = max(0.0, min(1.0, iron_oxide_frac))
    # Linear reduction: 18% iron → ~90 K reduction from base
    reduction = IRON_OXIDE_TEMP_REDUCTION_K * frac_clamped
    temp = SINTER_TEMP_BASE_K - reduction
    return max(MARS_AMBIENT_K + 100.0, temp)  # never below 310 K


def sintering_rate(temp_k: float) -> float:
    """Arrhenius sintering densification rate (fraction per second).

    Returns rate in [0, 1] per second.  At sintering temp, this gives
    meaningful densification over the soak period.
    """
    if temp_k <= MARS_AMBIENT_K:
        return 0.0
    exponent = -ACTIVATION_ENERGY_J_MOL / (GAS_CONSTANT * temp_k)
    # Clamp exponent to avoid underflow
    exponent = max(exponent, -500.0)
    return ARRHENIUS_PREFACTOR * math.exp(exponent)


def densification(rate_per_s: float, soak_seconds: float,
                  current_fraction: float) -> float:
    """Compute new density fraction after soaking.

    Densification slows as porosity decreases (asymptotic to 1.0).
    Uses first-order kinetics: df/dt = rate * (1 - f).
    Analytical solution: f(t) = 1 - (1-f0) * exp(-rate * t).
    """
    if rate_per_s <= 0.0 or soak_seconds <= 0.0:
        return current_fraction
    f0 = max(0.0, min(1.0, current_fraction))
    exponent = -rate_per_s * soak_seconds
    exponent = max(exponent, -500.0)
    new_frac = 1.0 - (1.0 - f0) * math.exp(exponent)
    return max(f0, min(1.0, new_frac))


def brick_strength(density_frac: float, sieved: bool = True) -> float:
    """Compressive strength in MPa from density fraction.

    Empirical power law fit from sintered regolith simulant data.
    Unsieved feedstock with fine dust reduces strength by 20%.
    """
    frac = max(0.0, min(1.0, density_frac))
    if frac < 0.01:
        return 0.0
    strength = STRENGTH_COEFF_MPA * (frac ** STRENGTH_EXPONENT)
    if not sieved:
        strength *= (1.0 - FINE_DUST_STRENGTH_PENALTY)
    return max(0.0, strength)


def energy_to_heat(mass_kg: float, temp_from_k: float,
                   temp_to_k: float) -> float:
    """Energy in kWh to heat a mass from one temperature to another.

    Uses constant specific heat (reasonable for regolith over this range).
    """
    if temp_to_k <= temp_from_k or mass_kg <= 0.0:
        return 0.0
    delta_t = temp_to_k - temp_from_k
    energy_j = mass_kg * REGOLITH_CP_J_KG_K * delta_t
    return energy_j / 3_600_000.0  # J → kWh


def kiln_wall_loss_kw(kiln_temp_k: float) -> float:
    """Conductive heat loss through kiln walls in kW.

    Q = k * A * ΔT / thickness.
    """
    if kiln_temp_k <= MARS_AMBIENT_K:
        return 0.0
    delta_t = kiln_temp_k - MARS_AMBIENT_K
    loss_w = (KILN_WALL_CONDUCTIVITY_W_MK * KILN_INNER_AREA_M2
              * delta_t / KILN_WALL_THICKNESS_M)
    return loss_w / 1000.0


def kiln_radiation_loss_kw(kiln_temp_k: float) -> float:
    """Radiative heat loss through kiln opening in kW.

    Stefan-Boltzmann through the opening area.
    """
    if kiln_temp_k <= MARS_AMBIENT_K:
        return 0.0
    loss_w = (KILN_EMISSIVITY * BOLTZMANN * KILN_OPENING_AREA_M2
              * (kiln_temp_k**4 - MARS_AMBIENT_K**4))
    return loss_w / 1000.0


def total_loss_kw(kiln_temp_k: float) -> float:
    """Total kiln heat loss in kW."""
    return kiln_wall_loss_kw(kiln_temp_k) + kiln_radiation_loss_kw(kiln_temp_k)


def heating_hours(mass_kg: float, temp_from_k: float, temp_to_k: float,
                  heater_kw: float) -> float:
    """Estimate hours to heat charge, accounting for losses.

    Uses average loss over the temperature range.
    """
    if heater_kw <= 0.0 or temp_to_k <= temp_from_k or mass_kg <= 0.0:
        return 0.0
    avg_temp = (temp_from_k + temp_to_k) / 2.0
    avg_loss_kw = total_loss_kw(avg_temp)
    net_power_kw = heater_kw - avg_loss_kw
    if net_power_kw <= 0.0:
        return float("inf")  # heater can't overcome losses
    energy_needed = energy_to_heat(mass_kg, temp_from_k, temp_to_k)
    return energy_needed / net_power_kw


def cooling_hours(temp_from_k: float, temp_to_k: float) -> float:
    """Hours to cool from temp_from to temp_to at max safe cooling rate.

    Controlled cooling at MAX_COOLING_RATE_K_PER_MIN.
    """
    if temp_from_k <= temp_to_k:
        return 0.0
    delta = temp_from_k - temp_to_k
    minutes = delta / MAX_COOLING_RATE_K_PER_MIN
    return minutes / 60.0


def crack_probability(cooling_rate_k_per_min: float) -> float:
    """Probability a brick cracks based on cooling rate.

    Below max safe rate: base probability.
    Above: probability increases linearly with excess rate.
    """
    if cooling_rate_k_per_min <= 0.0:
        return 0.0
    if cooling_rate_k_per_min <= MAX_COOLING_RATE_K_PER_MIN:
        return CRACK_PROBABILITY_BASE
    excess = cooling_rate_k_per_min - MAX_COOLING_RATE_K_PER_MIN
    return min(1.0, CRACK_PROBABILITY_BASE + 0.1 * excess)


def bricks_from_charge(charge_mass_kg: float) -> int:
    """Number of bricks producible from a given charge mass."""
    if charge_mass_kg <= 0.0 or BRICK_MASS_KG <= 0.0:
        return 0
    return int(charge_mass_kg / BRICK_MASS_KG)


# ---------------------------------------------------------------------------
# Tick engine
# ---------------------------------------------------------------------------

@dataclass
class SolRecord:
    """Immutable record of one sol's kiln operations."""

    sol: int = 0
    phase: str = "idle"
    kiln_temp_k: float = MARS_AMBIENT_K
    charge_temp_k: float = MARS_AMBIENT_K
    energy_input_kwh: float = 0.0
    wall_loss_kwh: float = 0.0
    radiation_loss_kwh: float = 0.0
    density_fraction: float = 0.0
    strength_mpa: float = 0.0
    bricks_produced: int = 0
    bricks_cracked: int = 0
    batch_complete: bool = False


def load_charge(state: KilnState, mass_kg: float,
                iron_oxide_frac: float = REGOLITH_IRON_OXIDE_FRAC,
                sieved: bool = True) -> str | None:
    """Load regolith into the kiln for a new batch.

    Returns error string or None on success.
    """
    if state.phase != "idle" and state.phase != "done":
        return f"Cannot load: kiln is {state.phase}"
    if mass_kg <= 0.0:
        return "Charge mass must be positive"
    if mass_kg > KILN_CAPACITY_KG:
        return f"Charge {mass_kg} kg exceeds capacity {KILN_CAPACITY_KG} kg"

    state.charge_mass_kg = mass_kg
    state.charge_temp_k = MARS_AMBIENT_K
    state.kiln_temp_k = MARS_AMBIENT_K
    state.iron_oxide_frac = max(0.0, min(1.0, iron_oxide_frac))
    state.feedstock_sieved = sieved
    state.target_temp_k = sintering_temperature(state.iron_oxide_frac)
    state.phase = "heating"
    state.phase_hours = 0.0
    state.energy_input_kwh = 0.0
    state.density_fraction = 0.0
    state.compressive_strength_mpa = 0.0
    state.bricks_this_batch = 0
    state.bricks_cracked = 0
    return None


def tick(state: KilnState, available_power_kw: float = KILN_HEATER_MAX_KW,
         dt_hours: float = MARS_SOL_HOURS) -> SolRecord:
    """Advance the kiln by one time step (default = one sol).

    Returns a SolRecord describing what happened.
    """
    state.sol += 1
    record = SolRecord(sol=state.sol, phase=state.phase)

    heater_kw = min(available_power_kw, KILN_HEATER_MAX_KW)
    dt_seconds = dt_hours * 3600.0

    if state.phase == "idle":
        record.kiln_temp_k = state.kiln_temp_k
        record.charge_temp_k = state.charge_temp_k
        return record

    if state.phase == "heating":
        _tick_heating(state, record, heater_kw, dt_seconds, dt_hours)

    elif state.phase == "soaking":
        _tick_soaking(state, record, heater_kw, dt_seconds, dt_hours)

    elif state.phase == "cooling":
        _tick_cooling(state, record, dt_seconds, dt_hours)

    elif state.phase == "done":
        record.phase = "done"
        record.density_fraction = state.density_fraction
        record.strength_mpa = state.compressive_strength_mpa
        record.bricks_produced = state.bricks_this_batch
        record.bricks_cracked = state.bricks_cracked
        record.batch_complete = True

    record.kiln_temp_k = state.kiln_temp_k
    record.charge_temp_k = state.charge_temp_k
    return record


def _tick_heating(state: KilnState, record: SolRecord,
                  heater_kw: float, dt_s: float, dt_h: float) -> None:
    """Heat charge toward sintering temperature."""
    loss_kw = total_loss_kw(state.kiln_temp_k)
    net_kw = heater_kw - loss_kw

    if net_kw > 0.0 and state.charge_mass_kg > 0.0:
        energy_j = net_kw * 1000.0 * dt_s
        delta_t = energy_j / (state.charge_mass_kg * REGOLITH_CP_J_KG_K)
        state.charge_temp_k += delta_t
        state.kiln_temp_k = state.charge_temp_k
    else:
        # Not enough power — temperature drifts toward ambient
        drift = min(1.0, dt_s / (state.charge_mass_kg * REGOLITH_CP_J_KG_K
                                  / (loss_kw * 1000.0 + 1.0)))
        state.charge_temp_k = (state.charge_temp_k
                               + drift * (MARS_AMBIENT_K - state.charge_temp_k))
        state.kiln_temp_k = state.charge_temp_k

    energy_kwh = heater_kw * dt_h
    state.energy_input_kwh += energy_kwh
    state.phase_hours += dt_h

    record.energy_input_kwh = energy_kwh
    record.wall_loss_kwh = kiln_wall_loss_kw(state.kiln_temp_k) * dt_h
    record.radiation_loss_kwh = kiln_radiation_loss_kw(state.kiln_temp_k) * dt_h

    # Transition to soaking when target reached
    if state.charge_temp_k >= state.target_temp_k:
        state.charge_temp_k = state.target_temp_k
        state.kiln_temp_k = state.target_temp_k
        state.phase = "soaking"
        state.phase_hours = 0.0


def _tick_soaking(state: KilnState, record: SolRecord,
                  heater_kw: float, dt_s: float, dt_h: float) -> None:
    """Hold at sintering temperature while densification occurs."""
    # Maintain temperature — heater compensates losses
    loss_kw = total_loss_kw(state.kiln_temp_k)
    maintain_kw = min(loss_kw, heater_kw)
    energy_kwh = maintain_kw * dt_h
    state.energy_input_kwh += energy_kwh
    state.phase_hours += dt_h

    # If heater can't maintain, temperature drops
    if heater_kw < loss_kw:
        deficit_kw = loss_kw - heater_kw
        deficit_j = deficit_kw * 1000.0 * dt_s
        temp_drop = deficit_j / (state.charge_mass_kg * REGOLITH_CP_J_KG_K
                                 + 1.0)
        state.charge_temp_k = max(MARS_AMBIENT_K,
                                  state.charge_temp_k - temp_drop)
        state.kiln_temp_k = state.charge_temp_k

    # Densification via Arrhenius kinetics
    rate = sintering_rate(state.charge_temp_k)
    state.density_fraction = densification(rate, dt_s, state.density_fraction)

    record.energy_input_kwh = energy_kwh
    record.wall_loss_kwh = kiln_wall_loss_kw(state.kiln_temp_k) * dt_h
    record.radiation_loss_kwh = kiln_radiation_loss_kw(state.kiln_temp_k) * dt_h
    record.density_fraction = state.density_fraction

    # Transition to cooling after soak period
    if state.phase_hours >= SOAK_TIME_HOURS:
        state.phase = "cooling"
        state.phase_hours = 0.0


def _tick_cooling(state: KilnState, record: SolRecord,
                  dt_s: float, dt_h: float) -> None:
    """Cool the charge at a controlled rate."""
    state.phase_hours += dt_h

    # Controlled cooling: limit to safe rate
    max_drop_this_step = MAX_COOLING_RATE_K_PER_MIN * (dt_h * 60.0)
    target_after_cool = max(MARS_AMBIENT_K + 50.0,
                            state.charge_temp_k - max_drop_this_step)

    # Natural cooling rate (from losses) may be slower than max
    loss_kw = total_loss_kw(state.kiln_temp_k)
    natural_drop_j = loss_kw * 1000.0 * dt_s
    natural_drop_k = natural_drop_j / (state.charge_mass_kg
                                        * REGOLITH_CP_J_KG_K + 1.0)
    natural_target = state.charge_temp_k - natural_drop_k

    # Use the slower (more conservative) cooling
    new_temp = max(target_after_cool, natural_target, MARS_AMBIENT_K)
    actual_rate = (state.charge_temp_k - new_temp) / max(dt_h * 60.0, 1.0)

    state.charge_temp_k = new_temp
    state.kiln_temp_k = new_temp

    record.wall_loss_kwh = kiln_wall_loss_kw(state.kiln_temp_k) * dt_h
    record.radiation_loss_kwh = kiln_radiation_loss_kw(state.kiln_temp_k) * dt_h

    # Check if cooled enough to extract
    if state.charge_temp_k <= MARS_AMBIENT_K + 60.0:
        _finish_batch(state, record, actual_rate)


def _finish_batch(state: KilnState, record: SolRecord,
                  cooling_rate_k_per_min: float) -> None:
    """Finalize batch: compute strength, count good/cracked bricks."""
    state.compressive_strength_mpa = brick_strength(
        state.density_fraction, state.feedstock_sieved)

    total_bricks = bricks_from_charge(state.charge_mass_kg)
    p_crack = crack_probability(cooling_rate_k_per_min)

    # Deterministic crack count (expected value) for reproducibility
    cracked = int(round(total_bricks * p_crack))
    good = total_bricks - cracked

    state.bricks_this_batch = good
    state.bricks_cracked = cracked
    state.bricks_produced_total += good
    state.bricks_cracked_total += cracked
    state.batches_completed += 1
    state.energy_total_kwh += state.energy_input_kwh

    state.phase = "done"

    record.density_fraction = state.density_fraction
    record.strength_mpa = state.compressive_strength_mpa
    record.bricks_produced = good
    record.bricks_cracked = cracked
    record.batch_complete = True

    state.log.append({
        "sol": state.sol,
        "batch": state.batches_completed,
        "bricks_good": good,
        "bricks_cracked": cracked,
        "density_frac": round(state.density_fraction, 4),
        "strength_mpa": round(state.compressive_strength_mpa, 2),
        "energy_kwh": round(state.energy_input_kwh, 2),
        "sieved": state.feedstock_sieved,
        "iron_oxide_frac": round(state.iron_oxide_frac, 3),
    })


# ---------------------------------------------------------------------------
# Convenience: run a full batch from load to done
# ---------------------------------------------------------------------------

def run_batch(mass_kg: float = KILN_CAPACITY_KG,
              iron_oxide_frac: float = REGOLITH_IRON_OXIDE_FRAC,
              sieved: bool = True,
              power_kw: float = KILN_HEATER_MAX_KW,
              max_sols: int = 200) -> tuple[KilnState, list[SolRecord]]:
    """Run a complete sintering batch and return (final_state, records).

    Useful for testing and analysis.
    """
    state = KilnState()
    err = load_charge(state, mass_kg, iron_oxide_frac, sieved)
    if err:
        raise ValueError(err)

    records: list[SolRecord] = []
    for _ in range(max_sols):
        rec = tick(state, available_power_kw=power_kw)
        records.append(rec)
        if state.phase == "done":
            break
    return state, records
