"""eva_suit.py -- Mars EVA Suit Life-Support Simulation.

Models the self-contained life-support system inside a Mars EVA
(Extravehicular Activity) suit.  Each tick = 1 minute of EVA time.

Physics modelled
----------------
* **O2 consumption** -- metabolic rate scales with activity level
  (rest -> heavy exertion).  CO2 produced at respiratory quotient RQ=0.85.
* **CO2 scrubbing** -- LiOH canister absorbs CO2; canister capacity
  degrades over the EVA.  When canister saturates, CO2 rises fast.
* **Suit pressure** -- maintained by O2 feed valve.  Micro-leaks bleed
  pressure; a breach causes rapid depress.
* **Thermal regulation** -- liquid cooling garment + heater.  Metabolic
  heat must be rejected; Mars ambient is ~-60 C so the suit can also
  get too cold at low activity.
* **Battery** -- powers fans, pumps, heater, radio.  Depletes over time.
* **Radiation dose** -- accumulated from ambient + any solar particle
  events.  Suit shielding attenuates but doesn't eliminate.
* **Mobility** -- movement speed depends on suit pressure, fatigue, and
  terrain slope.

Every output is clamped to physical bounds.  Conservation laws
(O2 in = O2 consumed + O2 in suit, energy in = energy out + stored)
are tested.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# -- Physical constants -------------------------------------------------------
MARS_AMBIENT_PRESSURE_KPA = 0.636          # Mars surface atmospheric pressure
SUIT_NOMINAL_PRESSURE_KPA = 29.6           # NASA EMU-style low-pressure pure O2
SUIT_MIN_SAFE_PRESSURE_KPA = 20.0          # Below this, hypoxia risk
SUIT_VOLUME_L = 200.0                      # Internal free volume (litres)

# O2 / CO2
O2_REST_L_PER_MIN = 0.3                    # O2 consumption at rest (STP litres/min)
O2_MAX_L_PER_MIN = 2.5                     # O2 consumption at max exertion
RESPIRATORY_QUOTIENT = 0.85                # CO2 produced / O2 consumed
CO2_DANGER_PERCENT = 3.0                   # CO2 partial pressure danger threshold
LIOH_CAPACITY_L = 600.0                    # Total CO2 the LiOH canister can absorb (litres STP)

# Pressure
MICRO_LEAK_KPA_PER_MIN = 0.005             # Baseline micro-leak rate
BREACH_LEAK_KPA_PER_MIN = 5.0              # Rapid depress from suit breach
FEED_VALVE_KPA_PER_MIN = 2.0              # Max O2 feed rate to repressurize

# Thermal
METABOLIC_HEAT_REST_W = 80.0               # Resting metabolic heat (watts)
METABOLIC_HEAT_MAX_W = 600.0               # Max exertion metabolic heat
MARS_AMBIENT_TEMP_C = -60.0                # Average Mars surface temp
SUIT_TARGET_TEMP_C = 22.0                  # Comfort target
COOLING_CAPACITY_W = 500.0                 # Liquid cooling garment max rejection
HEATER_CAPACITY_W = 200.0                  # Electric heater max output
SUIT_THERMAL_MASS_J_PER_C = 5000.0         # Thermal inertia of suit + crew
SUIT_INSULATION_W_PER_C = 2.0              # Heat leak to environment per deg C delta

# Battery
BATTERY_CAPACITY_WH = 500.0               # Suit battery (watt-hours)
BASE_POWER_DRAW_W = 25.0                   # Fans, pumps, radio baseline
HEATER_EFFICIENCY = 0.95                   # Electric heater efficiency

# Radiation
AMBIENT_RAD_USVH = 25.0                    # Ambient Mars surface (uSv/hour)
SUIT_SHIELDING_FACTOR = 0.7                # Suit attenuates to 70% of ambient
SPE_RAD_USVH = 5000.0                      # Solar particle event peak rate
CAREER_LIMIT_MSV = 1000.0                  # Career dose limit (millisieverts)

# Mobility
BASE_SPEED_KMH = 2.5                       # Walking speed in suit (km/h)
FATIGUE_RATE_PER_MIN = 0.001               # Fatigue accumulation per minute
MAX_FATIGUE = 1.0                          # Fatigue ceiling
FATIGUE_SPEED_PENALTY = 0.6                # Speed multiplier at max fatigue


# -- Data structures ----------------------------------------------------------

@dataclass
class SuitState:
    """Mutable state of one EVA suit."""
    pressure_kpa: float = SUIT_NOMINAL_PRESSURE_KPA
    o2_reserve_l: float = 600.0            # Portable O2 supply (STP litres)
    co2_absorbed_l: float = 0.0            # LiOH canister CO2 absorbed so far
    co2_in_suit_pct: float = 0.04          # Current CO2 percentage in suit atmo
    temp_c: float = SUIT_TARGET_TEMP_C
    battery_wh: float = BATTERY_CAPACITY_WH
    radiation_dose_usv: float = 0.0        # Accumulated dose this EVA (uSv)
    fatigue: float = 0.0                   # [0, 1]
    breached: bool = False
    elapsed_min: int = 0
    distance_km: float = 0.0

    def __post_init__(self) -> None:
        self.pressure_kpa = max(0.0, self.pressure_kpa)
        self.o2_reserve_l = max(0.0, self.o2_reserve_l)
        self.co2_absorbed_l = max(0.0, min(self.co2_absorbed_l, LIOH_CAPACITY_L))
        self.co2_in_suit_pct = max(0.0, min(self.co2_in_suit_pct, 100.0))
        self.temp_c = max(-273.15, self.temp_c)
        self.battery_wh = max(0.0, min(self.battery_wh, BATTERY_CAPACITY_WH))
        self.radiation_dose_usv = max(0.0, self.radiation_dose_usv)
        self.fatigue = max(0.0, min(self.fatigue, MAX_FATIGUE))


@dataclass
class EvaTick:
    """Output of one minute of EVA simulation."""
    o2_consumed_l: float = 0.0
    co2_produced_l: float = 0.0
    co2_scrubbed_l: float = 0.0
    pressure_delta_kpa: float = 0.0
    temp_delta_c: float = 0.0
    power_used_wh: float = 0.0
    radiation_usv: float = 0.0
    speed_kmh: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class EvaSession:
    """Summary of a complete EVA session (multiple ticks)."""
    total_minutes: int = 0
    total_o2_consumed_l: float = 0.0
    total_distance_km: float = 0.0
    total_radiation_usv: float = 0.0
    total_power_used_wh: float = 0.0
    peak_co2_pct: float = 0.0
    min_pressure_kpa: float = SUIT_NOMINAL_PRESSURE_KPA
    min_temp_c: float = SUIT_TARGET_TEMP_C
    max_temp_c: float = SUIT_TARGET_TEMP_C
    abort_reason: str = ""
    warnings: List[str] = field(default_factory=list)


# -- Core simulation functions ------------------------------------------------

def metabolic_rate(activity: float) -> Tuple[float, float]:
    """Return (O2 consumption L/min, metabolic heat W) for activity in [0,1].

    activity=0 is rest, activity=1 is maximum exertion.
    Interpolation is non-linear -- exertion costs disproportionately more.
    """
    activity = max(0.0, min(1.0, activity))
    t = activity ** 1.5
    o2 = O2_REST_L_PER_MIN + t * (O2_MAX_L_PER_MIN - O2_REST_L_PER_MIN)
    heat = METABOLIC_HEAT_REST_W + t * (METABOLIC_HEAT_MAX_W - METABOLIC_HEAT_REST_W)
    return (o2, heat)


def scrub_co2(co2_produced_l: float, canister_used_l: float) -> Tuple[float, float]:
    """Scrub CO2 through LiOH canister.

    Returns (co2_actually_scrubbed_l, new_canister_used_l).
    Scrubbing efficiency degrades as canister fills.
    """
    co2_produced_l = max(0.0, co2_produced_l)
    canister_used_l = max(0.0, canister_used_l)
    remaining_capacity = max(0.0, LIOH_CAPACITY_L - canister_used_l)
    fill_fraction = canister_used_l / LIOH_CAPACITY_L if LIOH_CAPACITY_L > 0 else 1.0
    efficiency = max(0.0, 1.0 - 0.3 * fill_fraction)
    scrubbed = min(co2_produced_l * efficiency, remaining_capacity)
    return (scrubbed, canister_used_l + scrubbed)


def pressure_tick(
    suit: SuitState,
    o2_consumed_l: float,
    co2_net_l: float,
) -> float:
    """Update suit pressure for one minute.

    Accounts for: O2 feed valve, micro-leaks, breach, gas consumption.
    Returns pressure delta (kPa).
    """
    leak_rate = BREACH_LEAK_KPA_PER_MIN if suit.breached else MICRO_LEAK_KPA_PER_MIN
    pressure_deficit = max(0.0, SUIT_NOMINAL_PRESSURE_KPA - suit.pressure_kpa)
    feed = min(pressure_deficit, FEED_VALVE_KPA_PER_MIN)
    # PV = nRT conversion: litres STP to kPa in suit volume
    gas_pressure_loss = o2_consumed_l * 0.05 if suit.o2_reserve_l > 0 else 0.0
    co2_pressure_add = max(0.0, co2_net_l) * 0.05

    delta = feed - leak_rate - gas_pressure_loss + co2_pressure_add
    if suit.o2_reserve_l <= 0:
        delta = -leak_rate - gas_pressure_loss + co2_pressure_add

    return delta


def thermal_tick(
    suit: SuitState,
    metabolic_heat_w: float,
) -> Tuple[float, float]:
    """Update suit temperature for one minute.

    Returns (temp_delta_C, heater_power_used_W).
    """
    ambient_delta = suit.temp_c - MARS_AMBIENT_TEMP_C
    insulation_loss = SUIT_INSULATION_W_PER_C * ambient_delta

    cooling = 0.0
    if suit.temp_c > SUIT_TARGET_TEMP_C:
        excess = suit.temp_c - SUIT_TARGET_TEMP_C
        cooling = min(COOLING_CAPACITY_W, excess * 50.0)

    heater_w = 0.0
    if suit.temp_c < SUIT_TARGET_TEMP_C:
        deficit = SUIT_TARGET_TEMP_C - suit.temp_c
        heater_w = min(HEATER_CAPACITY_W, deficit * 30.0)

    net_heat = metabolic_heat_w + heater_w * HEATER_EFFICIENCY - cooling - insulation_loss
    dt_seconds = 60.0
    temp_delta = (net_heat * dt_seconds) / SUIT_THERMAL_MASS_J_PER_C

    return (temp_delta, heater_w)


def radiation_tick(
    ambient_usv_per_hour: float = AMBIENT_RAD_USVH,
    solar_particle_event: bool = False,
) -> float:
    """Calculate radiation dose for one minute of EVA.

    Returns dose in uSv.
    """
    rate = ambient_usv_per_hour
    if solar_particle_event:
        rate += SPE_RAD_USVH
    dose = rate * SUIT_SHIELDING_FACTOR / 60.0
    return max(0.0, dose)


def mobility_speed(
    suit: SuitState,
    activity: float,
    terrain_slope_deg: float = 0.0,
) -> float:
    """Calculate movement speed (km/h) given suit state and terrain.

    Speed is reduced by: fatigue, low pressure, steep terrain.
    """
    activity = max(0.0, min(1.0, activity))
    walking = activity * 0.8 + 0.2
    speed = BASE_SPEED_KMH * walking

    fatigue_mult = 1.0 - suit.fatigue * FATIGUE_SPEED_PENALTY
    speed *= max(0.1, fatigue_mult)

    if suit.pressure_kpa < SUIT_NOMINAL_PRESSURE_KPA:
        pressure_ratio = suit.pressure_kpa / SUIT_NOMINAL_PRESSURE_KPA
        speed *= max(0.3, pressure_ratio)

    slope = abs(terrain_slope_deg)
    if slope > 0:
        slope_factor = max(0.1, 1.0 - slope / 45.0)
        speed *= slope_factor

    return max(0.0, speed)


def check_abort_conditions(suit: SuitState) -> str:
    """Check if EVA should be aborted.

    Returns empty string if OK, or abort reason.
    """
    if suit.pressure_kpa < SUIT_MIN_SAFE_PRESSURE_KPA:
        return "ABORT: Suit pressure below safe minimum"
    if suit.co2_in_suit_pct > CO2_DANGER_PERCENT:
        return "ABORT: CO2 level dangerous"
    if suit.battery_wh <= 0:
        return "ABORT: Battery depleted"
    if suit.o2_reserve_l <= 0 and suit.pressure_kpa < SUIT_NOMINAL_PRESSURE_KPA:
        return "ABORT: O2 reserve exhausted"
    if suit.temp_c > 45.0:
        return "ABORT: Hyperthermia risk"
    if suit.temp_c < 5.0:
        return "ABORT: Hypothermia risk"
    return ""


def tick_eva(
    suit: SuitState,
    activity: float = 0.3,
    terrain_slope_deg: float = 0.0,
    solar_particle_event: bool = False,
    ambient_rad_usv_h: float = AMBIENT_RAD_USVH,
) -> EvaTick:
    """Advance EVA suit simulation by one minute.

    Parameters
    ----------
    suit : SuitState
        Mutable suit state (modified in place).
    activity : float
        Activity level [0=rest, 1=max exertion].
    terrain_slope_deg : float
        Terrain slope in degrees (0=flat).
    solar_particle_event : bool
        Whether a solar particle event is occurring.
    ambient_rad_usv_h : float
        Ambient radiation rate (uSv/hour).

    Returns
    -------
    EvaTick with deltas and warnings for this minute.
    """
    result = EvaTick()
    warnings: List[str] = []

    # -- Metabolic --
    o2_rate, heat_w = metabolic_rate(activity)
    o2_available = min(o2_rate, suit.o2_reserve_l)
    suit.o2_reserve_l = max(0.0, suit.o2_reserve_l - o2_available)
    result.o2_consumed_l = o2_available

    if o2_available < o2_rate:
        warnings.append("LOW_O2: Reserve insufficient for metabolic demand")

    # -- CO2 --
    co2_produced = o2_available * RESPIRATORY_QUOTIENT
    result.co2_produced_l = co2_produced
    scrubbed, new_canister = scrub_co2(co2_produced, suit.co2_absorbed_l)
    suit.co2_absorbed_l = new_canister
    result.co2_scrubbed_l = scrubbed
    co2_net = co2_produced - scrubbed

    if SUIT_VOLUME_L > 0:
        co2_volume_in_suit = suit.co2_in_suit_pct / 100.0 * SUIT_VOLUME_L
        co2_volume_in_suit += co2_net
        co2_volume_in_suit = max(0.0, co2_volume_in_suit)
        suit.co2_in_suit_pct = min(100.0, (co2_volume_in_suit / SUIT_VOLUME_L) * 100.0)

    if suit.co2_in_suit_pct > 2.0:
        warnings.append("HIGH_CO2: %.1f%%" % suit.co2_in_suit_pct)

    if suit.co2_absorbed_l >= LIOH_CAPACITY_L * 0.9:
        warnings.append("CANISTER_LOW: LiOH canister >90%% used")

    # -- Pressure --
    p_delta = pressure_tick(suit, o2_available, co2_net)
    suit.pressure_kpa = max(0.0, suit.pressure_kpa + p_delta)
    result.pressure_delta_kpa = p_delta

    if suit.pressure_kpa < SUIT_MIN_SAFE_PRESSURE_KPA + 5.0:
        warnings.append("LOW_PRESSURE: %.1f kPa" % suit.pressure_kpa)

    # -- Thermal --
    t_delta, heater_w = thermal_tick(suit, heat_w)
    suit.temp_c += t_delta
    result.temp_delta_c = t_delta

    if suit.temp_c > 35.0:
        warnings.append("HIGH_TEMP: %.1f C" % suit.temp_c)
    elif suit.temp_c < 10.0:
        warnings.append("LOW_TEMP: %.1f C" % suit.temp_c)

    # -- Power --
    power_w = BASE_POWER_DRAW_W + heater_w
    power_wh = power_w / 60.0
    suit.battery_wh = max(0.0, suit.battery_wh - power_wh)
    result.power_used_wh = power_wh

    if suit.battery_wh < BATTERY_CAPACITY_WH * 0.2:
        warnings.append("LOW_BATTERY: %.0f Wh remaining" % suit.battery_wh)

    # -- Radiation --
    rad = radiation_tick(ambient_rad_usv_h, solar_particle_event)
    suit.radiation_dose_usv += rad
    result.radiation_usv = rad

    if solar_particle_event:
        warnings.append("SPE_ACTIVE: Seek shelter immediately")

    if suit.radiation_dose_usv > 500.0:
        warnings.append("HIGH_DOSE: %.0f uSv accumulated" % suit.radiation_dose_usv)

    # -- Mobility / Fatigue --
    suit.fatigue = min(MAX_FATIGUE, suit.fatigue + FATIGUE_RATE_PER_MIN * activity)
    speed = mobility_speed(suit, activity, terrain_slope_deg)
    result.speed_kmh = speed
    suit.distance_km += speed / 60.0
    suit.elapsed_min += 1

    result.warnings = warnings
    return result


def run_eva(
    suit: SuitState,
    duration_min: int = 240,
    activity: float = 0.3,
    terrain_slope_deg: float = 0.0,
    solar_particle_event_start: int = -1,
    solar_particle_event_duration: int = 0,
) -> EvaSession:
    """Run a complete EVA session for the given duration.

    Returns an EvaSession summary.  Aborts early if safety limits hit.
    """
    session = EvaSession()
    all_warnings: List[str] = []

    for minute in range(duration_min):
        abort = check_abort_conditions(suit)
        if abort:
            session.abort_reason = abort
            break

        spe = (
            solar_particle_event_start >= 0
            and solar_particle_event_start <= minute
            < solar_particle_event_start + solar_particle_event_duration
        )

        tick = tick_eva(suit, activity, terrain_slope_deg, spe)

        session.total_minutes += 1
        session.total_o2_consumed_l += tick.o2_consumed_l
        session.total_power_used_wh += tick.power_used_wh
        session.total_radiation_usv += tick.radiation_usv
        session.peak_co2_pct = max(session.peak_co2_pct, suit.co2_in_suit_pct)
        session.min_pressure_kpa = min(session.min_pressure_kpa, suit.pressure_kpa)
        session.min_temp_c = min(session.min_temp_c, suit.temp_c)
        session.max_temp_c = max(session.max_temp_c, suit.temp_c)

        all_warnings.extend(tick.warnings)

    session.total_distance_km = suit.distance_km
    session.warnings = sorted(set(all_warnings))
    return session
