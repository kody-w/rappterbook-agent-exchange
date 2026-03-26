"""fire_suppression.py -- Mars Habitat Fire Detection and Suppression.

Models the fire detection and suppression system for a pressurized Mars
habitat.  Each tick = 1 minute of real time during a fire event.

Physics modelled
----------------
* **Fire growth** -- heat release rate (HRR) follows a t-squared growth
  curve in enclosed volumes.  Growth rate depends on fuel type and O2
  concentration.  Mars habitats run 21-26% O2 at ~70 kPa total pressure;
  elevated O2 accelerates combustion dramatically.
* **Smoke production** -- smoke optical density scales with HRR and
  fuel type.  Smoke is the primary killer in enclosed habitat fires.
* **O2 depletion** -- fire consumes O2 proportional to HRR.
  Stoichiometric ratio ~13 MJ per kg O2 consumed.
* **CO production** -- incomplete combustion yields CO at a fraction
  that increases as O2 drops.  CO is lethal above 1200 ppm.
* **Temperature rise** -- compartment temperature via energy balance:
  fire input vs heat loss through habitat walls.
  Flashover occurs at ~500 C ceiling temperature.
* **Detection** -- smoke detectors trigger at configurable optical
  density.  Heat detectors trigger at rate-of-rise or fixed temp.
* **Suppression** -- CO2 flooding agent displaces O2 to extinguish.
  Agent mass required depends on compartment volume and O2 level.
* **Compartment isolation** -- sealing ventilation prevents O2
  resupply, starving the fire.

Reference incidents:
  - Apollo 1 (1967): pure O2 at 110 kPa, catastrophic fire in 17 seconds
  - Mir (1997): solid-fuel O2 generator fire, crew survived
  - ISS guidelines: 30 seconds to begin suppression after alarm

Conservation laws:
  - O2 consumed by fire + O2 remaining = initial O2 (mass balance)
  - Suppressant deployed + remaining = initial supply
  - All concentrations non-negative, O2 fraction in [0, 1]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


# -- Physical constants -------------------------------------------------------

HABITAT_PRESSURE_KPA = 70.0
NOMINAL_O2_FRACTION = 0.23
FLASHOVER_TEMP_C = 500.0
AMBIENT_TEMP_C = 22.0

# Fire growth (t-squared model alpha values, kW/s^2)
T_SQUARED_ALPHA_SLOW = 0.003
T_SQUARED_ALPHA_MEDIUM = 0.012
T_SQUARED_ALPHA_FAST = 0.047
O2_ENHANCEMENT_FACTOR = 2.5
O2_EXTINCTION_THRESHOLD = 0.14
HEAT_OF_COMBUSTION_MJ_PER_KG_O2 = 13.0
AIR_DENSITY_KG_PER_M3 = 1.0
SPECIFIC_HEAT_AIR_KJ_PER_KG_C = 1.005

# Smoke and CO
SMOKE_YIELD_PER_MW = 0.5
SMOKE_LETHAL_OD = 2.0
CO_BASE_YIELD_PPM_PER_KW = 0.1
CO_LOW_O2_MULTIPLIER = 5.0
CO_LETHAL_PPM = 1200.0

# Detection thresholds
SMOKE_DETECTOR_THRESHOLD_OD = 0.1
HEAT_DETECTOR_RATE_C_PER_MIN = 8.0
HEAT_DETECTOR_FIXED_TEMP_C = 57.0

# Suppression
CO2_AGENT_DENSITY_KG_PER_M3 = 1.98
AGENT_DISCHARGE_RATE_KG_PER_MIN = 20.0

# Compartment defaults
DEFAULT_COMPARTMENT_VOLUME_M3 = 50.0
WALL_HEAT_LOSS_KW_PER_C = 0.5
VENT_O2_SUPPLY_KG_PER_MIN = 0.1


# -- Data structures ----------------------------------------------------------

@dataclass
class FireState:
    """Mutable state of the fire suppression system and active fire."""

    minute: int = 0
    fire_active: bool = False
    fire_hrr_kw: float = 0.0
    fire_growth_alpha: float = T_SQUARED_ALPHA_MEDIUM
    fire_start_minute: int = 0

    o2_fraction: float = NOMINAL_O2_FRACTION
    o2_mass_kg: float = 0.0
    co_ppm: float = 0.0
    smoke_od: float = 0.0
    compartment_temp_c: float = AMBIENT_TEMP_C

    volume_m3: float = DEFAULT_COMPARTMENT_VOLUME_M3
    ventilation_open: bool = True

    agent_remaining_kg: float = 50.0
    agent_deployed_kg: float = 0.0
    suppression_active: bool = False

    smoke_alarm: bool = False
    heat_alarm: bool = False
    alarm_minute: int = -1

    fire_suppressed: bool = False
    flashover: bool = False
    casualties: bool = False
    total_o2_consumed_kg: float = 0.0
    peak_hrr_kw: float = 0.0
    peak_temp_c: float = AMBIENT_TEMP_C

    def __post_init__(self) -> None:
        self.o2_fraction = max(0.0, min(1.0, self.o2_fraction))
        self.co_ppm = max(0.0, self.co_ppm)
        self.smoke_od = max(0.0, self.smoke_od)
        self.compartment_temp_c = max(-60.0, self.compartment_temp_c)
        self.agent_remaining_kg = max(0.0, self.agent_remaining_kg)
        self.agent_deployed_kg = max(0.0, self.agent_deployed_kg)
        self.fire_hrr_kw = max(0.0, self.fire_hrr_kw)
        self.volume_m3 = max(1.0, self.volume_m3)
        if self.o2_mass_kg == 0.0 and self.o2_fraction > 0:
            air_mass = self.volume_m3 * AIR_DENSITY_KG_PER_M3
            self.o2_mass_kg = air_mass * self.o2_fraction


@dataclass
class FireMinute:
    """Output of one minute of fire simulation."""

    minute: int = 0
    hrr_kw: float = 0.0
    o2_fraction: float = NOMINAL_O2_FRACTION
    co_ppm: float = 0.0
    smoke_od: float = 0.0
    temp_c: float = AMBIENT_TEMP_C
    smoke_alarm: bool = False
    heat_alarm: bool = False
    suppression_active: bool = False
    agent_deployed_kg: float = 0.0
    fire_active: bool = False
    fire_suppressed: bool = False
    flashover: bool = False
    warnings: List[str] = field(default_factory=list)


# -- Pure physics functions ---------------------------------------------------

def t_squared_hrr(elapsed_min: float, alpha: float,
                  o2_fraction: float) -> float:
    """Compute fire heat release rate using t-squared model.

    HRR = alpha * t^2 (seconds), enhanced by elevated O2.
    Returns kW.
    """
    if elapsed_min <= 0 or alpha <= 0:
        return 0.0
    elapsed_s = elapsed_min * 60.0
    base_hrr = alpha * elapsed_s ** 2
    o2_excess = max(0.0, o2_fraction - 0.21)
    enhancement = 1.0 + (o2_excess / 0.05) * (O2_ENHANCEMENT_FACTOR - 1.0)
    return base_hrr * enhancement


def o2_consumed_kg(hrr_kw: float, duration_min: float) -> float:
    """Compute O2 mass consumed by fire in given time period."""
    if hrr_kw <= 0 or duration_min <= 0:
        return 0.0
    energy_mj = hrr_kw * duration_min * 60.0 / 1000.0
    return energy_mj / HEAT_OF_COMBUSTION_MJ_PER_KG_O2


def smoke_density(hrr_kw: float) -> float:
    """Compute smoke optical density (OD/m) from HRR."""
    return max(0.0, SMOKE_YIELD_PER_MW * hrr_kw / 1000.0)


def co_concentration(hrr_kw: float, o2_fraction: float,
                     volume_m3: float) -> float:
    """Compute CO concentration increment in ppm from fire output."""
    if hrr_kw <= 0 or volume_m3 <= 0:
        return 0.0
    base_co = CO_BASE_YIELD_PPM_PER_KW * hrr_kw
    if o2_fraction < 0.16:
        base_co *= CO_LOW_O2_MULTIPLIER
    return base_co * (DEFAULT_COMPARTMENT_VOLUME_M3 / volume_m3)


def compartment_temp_rise(hrr_kw: float, current_temp_c: float,
                          volume_m3: float) -> float:
    """Compute temperature change per minute from fire and heat losses.

    Returns delta-T in degrees C per minute.
    """
    air_mass = volume_m3 * AIR_DENSITY_KG_PER_M3
    thermal_cap = air_mass * SPECIFIC_HEAT_AIR_KJ_PER_KG_C
    if thermal_cap <= 0:
        return 0.0
    heat_in = hrr_kw * 60.0
    heat_loss = WALL_HEAT_LOSS_KW_PER_C * (current_temp_c - AMBIENT_TEMP_C) * 60.0
    net_heat = heat_in - heat_loss
    return net_heat / thermal_cap


def agent_needed_kg(volume_m3: float, o2_fraction: float) -> float:
    """Compute CO2 agent mass needed to reduce O2 below extinction."""
    if o2_fraction <= O2_EXTINCTION_THRESHOLD:
        return 0.0
    target_dilution = 1.0 - (O2_EXTINCTION_THRESHOLD / o2_fraction)
    volume_to_displace = volume_m3 * target_dilution
    return volume_to_displace * CO2_AGENT_DENSITY_KG_PER_M3


def check_smoke_detector(smoke_od: float) -> bool:
    """Check if smoke detector triggers."""
    return smoke_od >= SMOKE_DETECTOR_THRESHOLD_OD


def check_heat_detector(temp_c: float, prev_temp_c: float) -> bool:
    """Check if heat detector triggers (rate-of-rise or fixed temp)."""
    rate = temp_c - prev_temp_c
    return (rate >= HEAT_DETECTOR_RATE_C_PER_MIN or
            temp_c >= HEAT_DETECTOR_FIXED_TEMP_C)


# -- Tick function (one minute) -----------------------------------------------

def tick_fire(state: FireState,
              ignite: bool = False,
              activate_suppression: bool = False,
              seal_compartment: bool = False) -> FireMinute:
    """Advance the fire simulation by one minute.

    Parameters
    ----------
    state : FireState
        Mutable state -- modified in place.
    ignite : bool
        Start a fire this minute.
    activate_suppression : bool
        Manually trigger suppression system.
    seal_compartment : bool
        Close ventilation to starve fire.

    Returns
    -------
    FireMinute with this minute's results.
    """
    result = FireMinute(minute=state.minute)
    warnings: List[str] = []
    prev_temp = state.compartment_temp_c
    state.minute += 1

    if ignite and not state.fire_active and not state.fire_suppressed:
        state.fire_active = True
        state.fire_start_minute = state.minute - 1
        warnings.append("FIRE_IGNITED: Combustion detected")

    if seal_compartment:
        state.ventilation_open = False
        warnings.append("COMPARTMENT_SEALED: Ventilation closed")

    # Fire growth
    if state.fire_active and not state.fire_suppressed:
        elapsed = state.minute - state.fire_start_minute
        if state.o2_fraction > O2_EXTINCTION_THRESHOLD:
            state.fire_hrr_kw = t_squared_hrr(
                float(elapsed), state.fire_growth_alpha, state.o2_fraction
            )
        else:
            state.fire_hrr_kw = 0.0
            state.fire_active = False
            state.fire_suppressed = True
            warnings.append("FIRE_OUT: O2 below extinction threshold")

        consumed = o2_consumed_kg(state.fire_hrr_kw, 1.0)
        state.o2_mass_kg = max(0.0, state.o2_mass_kg - consumed)
        state.total_o2_consumed_kg += consumed

        if state.ventilation_open:
            state.o2_mass_kg += VENT_O2_SUPPLY_KG_PER_MIN

        air_mass = state.volume_m3 * AIR_DENSITY_KG_PER_M3
        if air_mass > 0:
            state.o2_fraction = max(0.0, min(1.0, state.o2_mass_kg / air_mass))

        state.smoke_od += smoke_density(state.fire_hrr_kw)
        state.co_ppm += co_concentration(
            state.fire_hrr_kw, state.o2_fraction, state.volume_m3
        )

        delta_t = compartment_temp_rise(
            state.fire_hrr_kw, state.compartment_temp_c, state.volume_m3
        )
        state.compartment_temp_c += delta_t

        state.peak_hrr_kw = max(state.peak_hrr_kw, state.fire_hrr_kw)
        state.peak_temp_c = max(state.peak_temp_c, state.compartment_temp_c)

        if state.compartment_temp_c >= FLASHOVER_TEMP_C and not state.flashover:
            state.flashover = True
            warnings.append(
                "FLASHOVER: Compartment fully involved at %.0f C"
                % state.compartment_temp_c
            )

    # Detection
    if not state.smoke_alarm and check_smoke_detector(state.smoke_od):
        state.smoke_alarm = True
        if state.alarm_minute < 0:
            state.alarm_minute = state.minute
        warnings.append("SMOKE_ALARM: OD %.2f" % state.smoke_od)

    if not state.heat_alarm and check_heat_detector(
        state.compartment_temp_c, prev_temp
    ):
        state.heat_alarm = True
        if state.alarm_minute < 0:
            state.alarm_minute = state.minute
        warnings.append("HEAT_ALARM: %.1f C" % state.compartment_temp_c)

    # Suppression
    if activate_suppression or (state.smoke_alarm and state.heat_alarm):
        state.suppression_active = True

    if (state.suppression_active and state.fire_active
            and state.agent_remaining_kg > 0):
        discharge = min(AGENT_DISCHARGE_RATE_KG_PER_MIN,
                        state.agent_remaining_kg)
        state.agent_remaining_kg -= discharge
        state.agent_deployed_kg += discharge
        displaced_o2 = (discharge / CO2_AGENT_DENSITY_KG_PER_M3
                        * AIR_DENSITY_KG_PER_M3 * state.o2_fraction)
        state.o2_mass_kg = max(0.0, state.o2_mass_kg - displaced_o2)
        air_mass = state.volume_m3 * AIR_DENSITY_KG_PER_M3
        if air_mass > 0:
            state.o2_fraction = max(0.0, min(1.0,
                                             state.o2_mass_kg / air_mass))
        result.agent_deployed_kg = discharge

        if state.agent_remaining_kg <= 0:
            warnings.append("AGENT_EXHAUSTED: Suppression depleted")

    # Casualty assessment
    if state.co_ppm >= CO_LETHAL_PPM and not state.casualties:
        state.casualties = True
        warnings.append("CASUALTIES: Lethal CO at %.0f ppm" % state.co_ppm)
    if state.flashover and not state.casualties:
        state.casualties = True
        warnings.append("CASUALTIES: Flashover — unsurvivable")

    # Status warnings
    if state.fire_active and state.fire_hrr_kw > 100:
        warnings.append("FIRE_GROWING: HRR %.0f kW" % state.fire_hrr_kw)
    if state.o2_fraction < 0.18 and state.fire_active:
        warnings.append("LOW_O2: %.1f%%" % (state.o2_fraction * 100))
    if state.smoke_od > 1.0:
        warnings.append("HEAVY_SMOKE: OD %.2f" % state.smoke_od)

    # Populate result
    result.hrr_kw = state.fire_hrr_kw
    result.o2_fraction = state.o2_fraction
    result.co_ppm = state.co_ppm
    result.smoke_od = state.smoke_od
    result.temp_c = state.compartment_temp_c
    result.smoke_alarm = state.smoke_alarm
    result.heat_alarm = state.heat_alarm
    result.suppression_active = state.suppression_active
    result.fire_active = state.fire_active
    result.fire_suppressed = state.fire_suppressed
    result.flashover = state.flashover
    result.warnings = warnings
    return result


def create_fire_system(scenario: str = "standard") -> FireState:
    """Create a fire suppression system for a given habitat scenario.

    Scenarios:
    - 'standard': normal habitat module, medium fire growth
    - 'enriched_o2': elevated O2 for medical bay, fast growth
    - 'storage': large volume, slow growth, heavy fuel load
    """
    configs = {
        "standard": FireState(),
        "enriched_o2": FireState(
            o2_fraction=0.26,
            fire_growth_alpha=T_SQUARED_ALPHA_FAST,
            volume_m3=30.0,
            agent_remaining_kg=40.0,
        ),
        "storage": FireState(
            fire_growth_alpha=T_SQUARED_ALPHA_SLOW,
            volume_m3=100.0,
            agent_remaining_kg=80.0,
        ),
    }
    return configs.get(scenario, configs["standard"])
