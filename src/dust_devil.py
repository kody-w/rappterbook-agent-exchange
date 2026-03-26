"""dust_devil.py — Mars convective dust devil simulation.

Models localized thermally-driven vortices on the Martian surface.
Dust devils are NOT dust storms — they are convective columns driven
by surface-atmosphere temperature gradients.  On Mars they reach
heights of 8-10 km and diameters of 200+ m, dwarfing their Earth
cousins.

Key difference from dust_storm.py: dust devils are *beneficial* to
solar-powered assets.  Spirit rover's solar panels were repeatedly
cleaned by passing dust devils, extending its mission by YEARS.

Physics modelled
----------------
* **Convective Available Potential Energy (CAPE)** — Vortex intensity
  scales with (T_surface - T_air).  Mars has extreme surface heating
  (low thermal inertia), producing strong thermals.

* **Tangential velocity** — Cyclostrophic balance:
  v_tan = sqrt(dP / ρ), where dP is the core pressure drop.
  Mars dust devils reach 20-45 m/s tangential winds.

* **Height** — Proportional to boundary layer depth:
  h ≈ k × (T_surface - T_air)^0.5.  Mars boundary layer extends
  5-10 km during afternoon heating → tall vortices.

* **Diameter** — Empirical: D ∝ h^0.5 for Mars dust devils.
  Ranges 10-700 m (median ~200 m).

* **Panel cleaning** — Passing vortex lifts deposited dust via
  shear stress.  Cleaning efficiency depends on tangential velocity
  and passage distance.  Spirit saw 5-10% panel recovery per event.

* **Pressure drop** — Core pressure deficit 1-9 Pa on Mars
  (measured by InSight, Pathfinder, Phoenix).  Drives the vortex.

* **Diurnal cycle** — Peak activity 10:00-15:00 local Mars time
  when surface heating is strongest.  Zero activity at night.

* **Seasonal cycle** — Peak during southern spring/summer
  (Ls 180-360) when Mars is closest to Sun.

Conservation laws & invariants
------------------------------
- Tangential velocity ≥ 0
- Height ≥ 0, diameter ≥ 0
- Panel cleaning ∈ [0, 1] (fraction of dust removed)
- Pressure drop ≥ 0
- sol_count monotonically increasing
- All temperatures in K ≥ 0
- Energy dissipated ≥ 0

References
----------
- Balme & Greeley 2006: "Dust devils on Earth and Mars" (Annual Rev)
- Lorenz 2016: "Dust devil statistics" (Icarus)
- InSight pressure data: Banfield et al. 2020 (Nature Geoscience)
- Spirit panel cleaning: Lorenz & Reiss 2015 (Icarus)
- Mars boundary layer: Hinson et al. 2008 (JGR)
- Pathfinder thermopile data: Schofield et al. 1997 (Science)

One tick = one sol.  Velocities in m/s, heights in m, temperature in K.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ── Physical constants ──────────────────────────────────────────────

# Mars atmosphere
MARS_AIR_DENSITY_KG_M3 = 0.020         # ~0.02 kg/m³ (vs Earth ~1.2)
MARS_SURFACE_TEMP_K = 210.0             # average surface temperature
MARS_GRAVITY_M_S2 = 3.72               # Mars surface gravity

# Diurnal surface heating (afternoon peak)
SURFACE_HEATING_PEAK_DT_K = 40.0        # max ΔT surface-air in afternoon
SURFACE_HEATING_MIN_DT_K = 2.0          # pre-dawn minimum ΔT

# Dust devil physical parameters (from InSight/Pathfinder/MER)
PRESSURE_DROP_COEFF_PA_PER_K = 0.25     # core pressure drop per K of ΔT
MAX_CORE_PRESSURE_DROP_PA = 9.0         # InSight measured up to 9 Pa
MIN_CORE_PRESSURE_DROP_PA = 0.5         # threshold for detection

# Velocity: cyclostrophic balance v = sqrt(dP / ρ)
# on Mars with ρ = 0.02 → v = sqrt(dP / 0.02) ≈ 7 × sqrt(dP)
V_TAN_MAX_M_S = 45.0                   # observed upper bound
V_TAN_MIN_M_S = 2.0                    # below this, no vortex

# Geometry
HEIGHT_COEFF_M_PER_SQRT_K = 300.0      # h = coeff × sqrt(ΔT)
MAX_HEIGHT_M = 10_000.0                # tallest observed on Mars
MIN_HEIGHT_M = 10.0                    # smallest detectable
DIAMETER_COEFF = 0.07                  # D = coeff × h (empirical)
MAX_DIAMETER_M = 700.0                 # largest observed
MIN_DIAMETER_M = 5.0                   # smallest vortex

# Occurrence rates (per sol, during dust devil season)
BASE_DEVILS_PER_SOL = 1.5              # average encounters near a base
SEASON_PEAK_LS = 270.0                 # perihelion — max heating
SEASON_WIDTH_LS = 90.0                 # half-width of active season

# Solar panel cleaning
MAX_CLEANING_FRACTION = 0.10           # Spirit saw up to 10% recovery
CLEANING_VELOCITY_THRESHOLD_M_S = 8.0  # need decent wind to lift dust
CLEANING_DISTANCE_DECAY_M = 200.0      # effectiveness drops with distance

# Damage (very rare — only from large, direct-hit devils)
DAMAGE_VELOCITY_THRESHOLD_M_S = 30.0   # only strongest cause damage
DAMAGE_RATE_PER_HIT = 0.001            # 0.1% equipment health per hit

# Mars sol duration
SOL_HOURS = 24.66                      # Mars sol in Earth hours


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class DustDevil:
    """A single dust devil event within one sol."""

    pressure_drop_pa: float = 0.0
    tangential_velocity_m_s: float = 0.0
    height_m: float = 0.0
    diameter_m: float = 0.0
    distance_from_base_m: float = 500.0
    cleaning_fraction: float = 0.0
    caused_damage: bool = False

    def __post_init__(self) -> None:
        """Clamp values to physical bounds."""
        self.pressure_drop_pa = max(0.0, min(self.pressure_drop_pa,
                                              MAX_CORE_PRESSURE_DROP_PA))
        self.tangential_velocity_m_s = max(0.0, min(
            self.tangential_velocity_m_s, V_TAN_MAX_M_S))
        self.height_m = max(0.0, min(self.height_m, MAX_HEIGHT_M))
        self.diameter_m = max(0.0, min(self.diameter_m, MAX_DIAMETER_M))
        self.distance_from_base_m = max(0.0, self.distance_from_base_m)
        self.cleaning_fraction = max(0.0, min(1.0, self.cleaning_fraction))


@dataclass
class DustDevilState:
    """Colony-level dust devil tracking across sols."""

    sol: int = 0
    solar_longitude_deg: float = 0.0

    # Current sol statistics
    devils_today: list[DustDevil] = field(default_factory=list)
    total_cleaning_today: float = 0.0
    damage_today: float = 0.0

    # Cumulative statistics
    total_devils_observed: int = 0
    total_cleaning_cumulative: float = 0.0
    total_damage_cumulative: float = 0.0
    strongest_ever_pa: float = 0.0
    tallest_ever_m: float = 0.0

    # Colony state
    panel_dust_coverage: float = 0.3     # 30% initial dust
    equipment_health: float = 1.0        # 100% initial


# ── Physics functions ───────────────────────────────────────────────

def seasonal_activity_modifier(solar_longitude_deg: float) -> float:
    """Dust devil activity modifier based on Mars season.

    Peak activity near perihelion (Ls ~270°) when surface heating
    is strongest.  Returns a multiplier in [0.1, 1.0].

    Args:
        solar_longitude_deg: Mars solar longitude in degrees [0, 360).

    Returns:
        Activity multiplier.
    """
    ls = solar_longitude_deg % 360.0
    delta = min(abs(ls - SEASON_PEAK_LS),
                360.0 - abs(ls - SEASON_PEAK_LS))
    gaussian = math.exp(-0.5 * (delta / SEASON_WIDTH_LS) ** 2)
    return 0.1 + 0.9 * gaussian


def surface_temperature_delta(solar_longitude_deg: float,
                               local_hour: float = 13.0) -> float:
    """Surface-to-air temperature difference driving convection.

    Peaks at ~13:00 local time.  Varies with season (more heating
    near perihelion).

    Args:
        solar_longitude_deg: Mars Ls in degrees.
        local_hour: local Mars hour [0, 24.66).

    Returns:
        Temperature difference in K (always ≥ 0).
    """
    # Diurnal curve: peak at 13:00, trough at 02:00
    hour_frac = (local_hour - 13.0) / 12.33  # half-sol
    diurnal = math.exp(-2.0 * hour_frac ** 2)

    # Seasonal: more heating near perihelion
    seasonal = seasonal_activity_modifier(solar_longitude_deg)

    dt = SURFACE_HEATING_MIN_DT_K + (
        SURFACE_HEATING_PEAK_DT_K - SURFACE_HEATING_MIN_DT_K
    ) * diurnal * seasonal

    return max(0.0, dt)


def core_pressure_drop(delta_t_k: float) -> float:
    """Vortex core pressure deficit from thermal gradient.

    Args:
        delta_t_k: surface-air temperature difference in K.

    Returns:
        Core pressure drop in Pa, clamped to [0, MAX].
    """
    dp = PRESSURE_DROP_COEFF_PA_PER_K * max(0.0, delta_t_k)
    return min(dp, MAX_CORE_PRESSURE_DROP_PA)


def tangential_velocity(pressure_drop_pa: float) -> float:
    """Cyclostrophic balance: v = sqrt(dP / ρ).

    On Mars with ρ ≈ 0.02 kg/m³, even small pressure drops
    produce significant velocities.

    Args:
        pressure_drop_pa: core pressure drop in Pa.

    Returns:
        Tangential velocity in m/s.
    """
    if pressure_drop_pa <= 0.0:
        return 0.0
    v = math.sqrt(pressure_drop_pa / MARS_AIR_DENSITY_KG_M3)
    return min(v, V_TAN_MAX_M_S)


def vortex_height(delta_t_k: float) -> float:
    """Dust devil height from convective boundary layer depth.

    Args:
        delta_t_k: surface-air temperature difference in K.

    Returns:
        Vortex height in m.
    """
    if delta_t_k <= 0.0:
        return 0.0
    h = HEIGHT_COEFF_M_PER_SQRT_K * math.sqrt(delta_t_k)
    return max(MIN_HEIGHT_M, min(h, MAX_HEIGHT_M))


def vortex_diameter(height_m: float) -> float:
    """Dust devil diameter from empirical height scaling.

    Args:
        height_m: vortex height in m.

    Returns:
        Diameter in m.
    """
    if height_m <= 0.0:
        return 0.0
    d = DIAMETER_COEFF * height_m
    return max(MIN_DIAMETER_M, min(d, MAX_DIAMETER_M))


def panel_cleaning_efficiency(v_tan_m_s: float,
                                distance_m: float) -> float:
    """Fraction of panel dust removed by a passing dust devil.

    Spirit rover saw 5-10% panel recovery from nearby vortices.
    Efficiency drops with distance and requires minimum wind speed.

    Args:
        v_tan_m_s: tangential velocity in m/s.
        distance_m: closest approach to solar panels in m.

    Returns:
        Fraction of dust removed ∈ [0, MAX_CLEANING_FRACTION].
    """
    if v_tan_m_s < CLEANING_VELOCITY_THRESHOLD_M_S:
        return 0.0
    if distance_m < 0.0:
        distance_m = 0.0

    # Velocity factor: linear above threshold
    v_factor = min(1.0, (v_tan_m_s - CLEANING_VELOCITY_THRESHOLD_M_S) /
                   (V_TAN_MAX_M_S - CLEANING_VELOCITY_THRESHOLD_M_S))

    # Distance decay: exponential
    d_factor = math.exp(-distance_m / CLEANING_DISTANCE_DECAY_M)

    return MAX_CLEANING_FRACTION * v_factor * d_factor


def damage_from_devil(v_tan_m_s: float, distance_m: float,
                       diameter_m: float) -> float:
    """Equipment damage from a large, close dust devil.

    Only the strongest vortices at very close range cause damage.
    The vortex must essentially pass over the base (distance < diameter).

    Args:
        v_tan_m_s: tangential velocity in m/s.
        distance_m: closest approach in m.
        diameter_m: vortex diameter in m.

    Returns:
        Equipment health reduction ∈ [0, DAMAGE_RATE_PER_HIT].
    """
    if v_tan_m_s < DAMAGE_VELOCITY_THRESHOLD_M_S:
        return 0.0
    if distance_m > diameter_m:
        return 0.0

    # Strength factor
    v_factor = min(1.0, (v_tan_m_s - DAMAGE_VELOCITY_THRESHOLD_M_S) /
                   (V_TAN_MAX_M_S - DAMAGE_VELOCITY_THRESHOLD_M_S))

    return DAMAGE_RATE_PER_HIT * v_factor


def generate_devil(delta_t_k: float,
                    max_distance_m: float = 2000.0) -> DustDevil | None:
    """Generate a single dust devil from thermal conditions.

    Returns None if conditions are too weak.

    Args:
        delta_t_k: surface-air temperature difference in K.
        max_distance_m: maximum distance from base for detection.

    Returns:
        A DustDevil instance, or None if below threshold.
    """
    dp = core_pressure_drop(delta_t_k)
    if dp < MIN_CORE_PRESSURE_DROP_PA:
        return None

    # Add ±20% random variation
    dp *= random.uniform(0.8, 1.2)
    dp = min(dp, MAX_CORE_PRESSURE_DROP_PA)

    v_tan = tangential_velocity(dp)
    if v_tan < V_TAN_MIN_M_S:
        return None

    h = vortex_height(delta_t_k * random.uniform(0.8, 1.2))
    d = vortex_diameter(h)

    # Random distance from base
    dist = random.uniform(10.0, max_distance_m)

    cleaning = panel_cleaning_efficiency(v_tan, dist)
    dmg = damage_from_devil(v_tan, dist, d)

    return DustDevil(
        pressure_drop_pa=round(dp, 3),
        tangential_velocity_m_s=round(v_tan, 2),
        height_m=round(h, 1),
        diameter_m=round(d, 1),
        distance_from_base_m=round(dist, 1),
        cleaning_fraction=round(cleaning, 5),
        caused_damage=dmg > 0.0,
    )


# ── Tick function ───────────────────────────────────────────────────

def tick_dust_devils(state: DustDevilState,
                      rng: random.Random | None = None) -> DustDevilState:
    """Advance the dust devil simulation by one sol.

    Generates dust devils based on current season, applies panel
    cleaning and equipment damage, updates cumulative statistics.

    Args:
        state: current DustDevilState (mutated in place).
        rng: optional seeded Random instance for reproducibility.

    Returns:
        The mutated state.
    """
    if rng is None:
        rng = random.Random()

    state.sol += 1
    state.solar_longitude_deg = (state.solar_longitude_deg +
                                  0.524) % 360.0  # ~0.524°/sol

    # Reset daily counters
    state.devils_today = []
    state.total_cleaning_today = 0.0
    state.damage_today = 0.0

    # How many dust devils today?
    activity = seasonal_activity_modifier(state.solar_longitude_deg)
    expected_count = BASE_DEVILS_PER_SOL * activity
    # Poisson draw
    count = 0
    prob = math.exp(-expected_count)
    cumulative = prob
    r = rng.random()
    while cumulative < r and count < 20:
        count += 1
        prob *= expected_count / count
        cumulative += prob
    # count is our Poisson sample (capped at 20)

    # Generate each devil during peak hours (10:00-15:00)
    for _ in range(count):
        hour = rng.uniform(10.0, 15.0)
        dt = surface_temperature_delta(state.solar_longitude_deg, hour)
        # Add individual variation
        dt *= rng.uniform(0.7, 1.3)

        old_random_state = random.getstate()
        random.setstate(rng.getstate())
        devil = generate_devil(dt)
        rng.setstate(random.getstate())
        random.setstate(old_random_state)

        if devil is not None:
            state.devils_today.append(devil)

            # Apply cleaning (multiplicative removal)
            if devil.cleaning_fraction > 0.0:
                state.panel_dust_coverage *= (1.0 - devil.cleaning_fraction)
                state.panel_dust_coverage = max(0.0,
                                                 state.panel_dust_coverage)
                state.total_cleaning_today += devil.cleaning_fraction

            # Apply damage
            if devil.caused_damage:
                dmg = damage_from_devil(
                    devil.tangential_velocity_m_s,
                    devil.distance_from_base_m,
                    devil.diameter_m,
                )
                state.equipment_health = max(0.0,
                                              state.equipment_health - dmg)
                state.damage_today += dmg

            # Update records
            if devil.pressure_drop_pa > state.strongest_ever_pa:
                state.strongest_ever_pa = devil.pressure_drop_pa
            if devil.height_m > state.tallest_ever_m:
                state.tallest_ever_m = devil.height_m

    # Update cumulative stats
    state.total_devils_observed += len(state.devils_today)
    state.total_cleaning_cumulative += state.total_cleaning_today
    state.total_damage_cumulative += state.damage_today

    # Natural dust re-accumulation (ambient settling)
    state.panel_dust_coverage = min(
        1.0, state.panel_dust_coverage + 0.003)  # ~0.3%/sol baseline

    return state


# ── Factory ─────────────────────────────────────────────────────────

def create_dust_devil_state(initial_dust: float = 0.3,
                             solar_longitude_deg: float = 0.0
                             ) -> DustDevilState:
    """Create a fresh dust devil tracking state.

    Args:
        initial_dust: initial panel dust coverage [0, 1].
        solar_longitude_deg: starting Mars Ls.

    Returns:
        A new DustDevilState.
    """
    return DustDevilState(
        panel_dust_coverage=max(0.0, min(1.0, initial_dust)),
        solar_longitude_deg=solar_longitude_deg % 360.0,
    )
