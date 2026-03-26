"""landing_pad.py -- Mars Colony Landing Pad & Supply Reception.

Models a reinforced landing pad for receiving supply pods from orbit.
Each tick = 1 sol of pad operations.

Physics modelled
----------------
* **Rocket equation** -- Tsiolkovsky fuel requirement for powered Mars
  landing.  Fuel = m_f * (exp(dv / (Isp * g0)) - 1).  Δv ≈ 1200 m/s
  for Mars orbit-to-surface powered descent.
* **Surface degradation** -- each landing erodes the pad surface from
  exhaust blast.  Below a threshold, FOD (Foreign Object Debris) risk
  triggers mandatory resurfacing.
* **Approach Go/No-Go** -- five independent criteria must pass before
  a pod is cleared to land: wind, dust opacity (tau), visibility,
  pad availability, and beacon health.
* **Beacon system** -- radio/optical beacons guide pods to the pad.
  Degrade each sol and with each landing.  Below minimum health,
  landings are unsafe.
* **Thermal** -- rocket exhaust heats the pad.  Mars cold provides
  passive cooling between landings.
* **Cargo unloading** -- pods carry up to 5 tonnes.  Unloading rate
  limited by EVA crew and rover capacity (~2 tonnes/sol).
* **Pad clearing** -- landed pod must be removed before next landing.

Reference systems:
  - SpaceX Starship: ~100 t payload, Raptor engines Isp ≈ 350 s
  - Mars Supply Pod (this model): 5 t payload, Isp 350 s, Δv 1200 m/s
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# -- Physical constants -------------------------------------------------------

MARS_GRAVITY_M_S2 = 3.72
EARTH_GRAVITY_M_S2 = 9.81

# Rocket
LANDING_DELTA_V_M_S = 1200.0
ENGINE_ISP_S = 350.0
DEFAULT_POD_DRY_MASS_KG = 2000.0
MAX_PAYLOAD_KG = 5000.0

# Pad geometry
PAD_RADIUS_M = 25.0

# Surface
SURFACE_STRENGTH_NEW = 1.0
SURFACE_DEGRADATION_PER_LANDING = 0.05
SURFACE_FOD_THRESHOLD = 0.4
SURFACE_MIN = 0.1
RESURFACE_RESTORE = 0.6

# Weather limits
WIND_SAFE_LIMIT_M_S = 25.0
DUST_TAU_SAFE_LIMIT = 3.0
VISIBILITY_MIN_M = 500.0
CLEAR_SKY_TAU = 0.3

# Thermal
PAD_AMBIENT_TEMP_K = 210.0
PAD_TEMP_SPIKE_PER_LANDING_K = 150.0
PAD_COOLING_RATE = 0.3  # fraction of excess temp shed per sol

# Beacons
BEACON_HEALTH_NEW = 1.0
BEACON_MIN_HEALTH = 0.1
BEACON_DEGRADATION_PER_SOL = 0.001
BEACON_DEGRADATION_PER_LANDING = 0.02
BEACON_MAINTENANCE_RESTORE = 0.6

# Cargo
UNLOAD_RATE_KG_PER_SOL = 2000.0


# -- Data structures ----------------------------------------------------------

@dataclass
class SupplyPod:
    """A supply pod descending from Mars orbit."""
    payload_kg: float = 0.0
    landed: bool = False
    unloaded_kg: float = 0.0
    manifest: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.payload_kg = max(0.0, min(self.payload_kg, MAX_PAYLOAD_KG))
        self.unloaded_kg = max(0.0, min(self.unloaded_kg, self.payload_kg))

    @property
    def remaining_kg(self):
        """Cargo still on the pod."""
        return max(0.0, self.payload_kg - self.unloaded_kg)

    @property
    def fully_unloaded(self):
        """True when all cargo has been removed."""
        return self.remaining_kg <= 0.0


@dataclass
class LandingPadState:
    """Mutable state of a Mars landing pad."""
    sol: int = 0
    surface_strength: float = SURFACE_STRENGTH_NEW
    beacon_health: float = BEACON_HEALTH_NEW
    active_pod: Optional[SupplyPod] = None
    wind_speed_m_s: float = 0.0
    dust_tau: float = 0.0
    visibility_m: float = 10000.0
    pad_temp_k: float = PAD_AMBIENT_TEMP_K
    total_landings: int = 0
    total_fuel_spent_kg: float = 0.0
    total_cargo_received_kg: float = 0.0

    def __post_init__(self):
        self.surface_strength = max(SURFACE_MIN, min(self.surface_strength, 1.0))
        self.beacon_health = max(BEACON_MIN_HEALTH, min(self.beacon_health, 1.0))
        self.pad_temp_k = max(PAD_AMBIENT_TEMP_K, self.pad_temp_k)
        self.total_landings = max(0, self.total_landings)
        self.total_fuel_spent_kg = max(0.0, self.total_fuel_spent_kg)
        self.total_cargo_received_kg = max(0.0, self.total_cargo_received_kg)

    @property
    def pad_clear(self):
        """True when no pod is on the pad."""
        return self.active_pod is None

    @property
    def fod_risk(self):
        """True when surface is degraded enough for debris risk."""
        return self.surface_strength < SURFACE_FOD_THRESHOLD

    @property
    def landing_safe(self):
        """True when weather and beacon conditions allow landing."""
        return (self.wind_speed_m_s <= WIND_SAFE_LIMIT_M_S
                and self.dust_tau <= DUST_TAU_SAFE_LIMIT
                and self.visibility_m >= VISIBILITY_MIN_M
                and self.beacon_health > BEACON_MIN_HEALTH)


@dataclass
class PadSol:
    """Input parameters for one sol of pad operations."""
    sol: int = 0
    incoming_pod: Optional[SupplyPod] = None
    unload: bool = False
    clear_pod: bool = False
    resurface: bool = False
    maintain_beacons: bool = False
    wind_m_s: float = 5.0
    dust_tau: float = 0.3
    visibility_m: float = 10000.0


# -- Physics functions --------------------------------------------------------

def fuel_required_kg(payload_kg, delta_v_m_s=LANDING_DELTA_V_M_S,
                     isp_s=ENGINE_ISP_S):
    """Fuel required for powered Mars landing via Tsiolkovsky equation.

    Returns fuel mass in kg. Returns 0 for invalid inputs.
    """
    if payload_kg < 0:
        return 0.0
    if delta_v_m_s <= 0 or isp_s <= 0:
        return 0.0
    m_f = DEFAULT_POD_DRY_MASS_KG + max(0.0, payload_kg)
    exhaust_velocity = isp_s * EARTH_GRAVITY_M_S2
    mass_ratio = math.exp(delta_v_m_s / exhaust_velocity)
    fuel = m_f * (mass_ratio - 1.0)
    return max(0.0, fuel)


def approach_go_nogo(wind_m_s, dust_tau, visibility_m,
                     pad_clear, beacon_health):
    """Evaluate five independent landing criteria.

    Returns dict with 'go' (bool) and 'criteria' (dict of individual checks).
    """
    criteria = {
        "wind": wind_m_s <= WIND_SAFE_LIMIT_M_S,
        "dust": dust_tau <= DUST_TAU_SAFE_LIMIT,
        "visibility": visibility_m >= VISIBILITY_MIN_M,
        "pad_clear": bool(pad_clear),
        "beacons": beacon_health > BEACON_MIN_HEALTH,
    }
    return {"go": all(criteria.values()), "criteria": criteria}


def land_pod(state, pod):
    """Attempt to land a supply pod on the pad.

    Mutates state and pod in place. Returns result dict.
    """
    # Go/no-go check
    decision = approach_go_nogo(
        state.wind_speed_m_s, state.dust_tau, state.visibility_m,
        state.pad_clear, state.beacon_health)

    if not decision["go"]:
        return {"success": False, "reason": "no_go", "criteria": decision["criteria"]}

    # Calculate fuel
    fuel = fuel_required_kg(pod.payload_kg)

    # Land the pod
    pod.landed = True
    state.active_pod = pod

    # Surface degradation
    state.surface_strength = max(SURFACE_MIN,
                                  state.surface_strength - SURFACE_DEGRADATION_PER_LANDING)

    # Thermal spike
    state.pad_temp_k += PAD_TEMP_SPIKE_PER_LANDING_K

    # Beacon degradation from landing vibration
    state.beacon_health = max(BEACON_MIN_HEALTH,
                               state.beacon_health - BEACON_DEGRADATION_PER_LANDING)

    # Accounting
    state.total_landings += 1
    state.total_fuel_spent_kg += fuel

    return {"success": True, "fuel_spent_kg": fuel}


def unload_cargo(state, kg_to_unload=None):
    """Unload cargo from the active pod.

    Mutates state and pod in place. Returns result dict.
    """
    if state.active_pod is None:
        return {"success": False, "reason": "no_pod_on_pad",
                "unloaded_kg": 0.0, "remaining_kg": 0.0, "fully_unloaded": False}

    pod = state.active_pod
    remaining = pod.remaining_kg

    if remaining <= 0:
        return {"success": True, "unloaded_kg": 0.0,
                "remaining_kg": 0.0, "fully_unloaded": True}

    if kg_to_unload is not None:
        to_unload = min(max(0.0, kg_to_unload), remaining)
    else:
        to_unload = min(UNLOAD_RATE_KG_PER_SOL, remaining)

    pod.unloaded_kg += to_unload
    state.total_cargo_received_kg += to_unload

    return {
        "success": True,
        "unloaded_kg": to_unload,
        "remaining_kg": pod.remaining_kg,
        "fully_unloaded": pod.fully_unloaded,
    }


def clear_pad(state):
    """Remove the landed pod from the pad.

    Returns result dict with remaining cargo info.
    """
    if state.active_pod is None:
        return {"success": False, "cargo_remaining_kg": 0.0}

    remaining = state.active_pod.remaining_kg
    state.active_pod = None
    return {"success": True, "cargo_remaining_kg": remaining}


def resurface_pad(state):
    """Resurface the landing pad to restore surface strength.

    Pad must be clear. Restores a fraction of the gap to perfect.
    """
    if not state.pad_clear:
        return {"success": False, "reason": "pad_occupied"}

    gap = 1.0 - state.surface_strength
    state.surface_strength = min(1.0,
                                  state.surface_strength + gap * RESURFACE_RESTORE)
    return {"success": True, "surface_strength": state.surface_strength}


def maintain_beacons(state):
    """Service the beacon array to restore health.

    Restores a fraction of the gap to perfect.
    """
    gap = 1.0 - state.beacon_health
    state.beacon_health = min(1.0,
                               state.beacon_health + gap * BEACON_MAINTENANCE_RESTORE)
    return {"success": True, "beacon_health": state.beacon_health}


# -- Main tick ----------------------------------------------------------------

def tick_pad(state, sol):
    """Advance the landing pad by one sol.

    Parameters
    ----------
    state : LandingPadState
        Current pad state (mutated in place).
    sol : PadSol
        Sol parameters: weather, incoming pod, operations.

    Returns
    -------
    dict
        Results of all operations this sol.
    """
    state.sol = sol.sol
    result = {"sol": sol.sol}

    # Update weather
    state.wind_speed_m_s = sol.wind_m_s
    state.dust_tau = sol.dust_tau
    state.visibility_m = sol.visibility_m

    # Landing
    if sol.incoming_pod is not None:
        result["landing"] = land_pod(state, sol.incoming_pod)
    else:
        result["landing"] = None

    # Unloading
    if sol.unload:
        result["unload"] = unload_cargo(state)
    else:
        result["unload"] = None

    # Clear pod
    if sol.clear_pod:
        result["clear"] = clear_pad(state)
    else:
        result["clear"] = None

    # Resurfacing
    if sol.resurface:
        result["resurface"] = resurface_pad(state)
    else:
        result["resurface"] = None

    # Beacon maintenance
    if sol.maintain_beacons:
        result["beacon_maintenance"] = maintain_beacons(state)
    else:
        result["beacon_maintenance"] = None

    # Daily beacon degradation
    state.beacon_health = max(BEACON_MIN_HEALTH,
                               state.beacon_health - BEACON_DEGRADATION_PER_SOL)

    # Thermal: passive cooling toward ambient
    if state.pad_temp_k > PAD_AMBIENT_TEMP_K:
        excess = state.pad_temp_k - PAD_AMBIENT_TEMP_K
        state.pad_temp_k = PAD_AMBIENT_TEMP_K + excess * (1.0 - PAD_COOLING_RATE)

    # Summary
    result["pad_clear"] = state.pad_clear
    result["surface_after"] = state.surface_strength
    result["beacon_after"] = state.beacon_health
    result["total_landings"] = state.total_landings
    result["total_cargo_kg"] = state.total_cargo_received_kg

    return result


def create_landing_pad():
    """Create a fresh landing pad with default configuration."""
    return LandingPadState()
