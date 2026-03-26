"""
habitat_pressure.py — Mars habitat pressurization model.

The pressure vessel IS the colony. Mars surface is ~0.6 kPa; humans need
~101.3 kPa inside. That 100 kPa differential is one hull breach away from
killing everyone. This module models the physics of keeping air inside.

Subsystems modelled:
  - Structural stress from internal/external pressure differential
  - Micro-leak accumulation (seals degrade over time, dust intrusion)
  - Airlock cycling losses (each EVA bleeds a fixed volume of gas)
  - Emergency blowout events (meteorite strike, seal failure)
  - Gas replenishment from reserve tanks
  - Pressure regulation (target maintenance with dead-band control)

Physical references:
  - ISS operates at 101.3 kPa (1 atm), leak rate ~0.5 kg/day
  - Mars surface pressure: ~0.6 kPa (varies seasonally 0.4-0.87 kPa)
  - ISS airlock loss per EVA: ~0.14 m3 of air at cabin pressure
  - Habitat hull stress: sigma = P*r / (2*t) for spherical pressure vessel
  - Ideal gas law: PV = nRT for gas inventory tracking

One tick = one sol.  Pressure in kPa, volume in m3, mass in kg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MARS_SURFACE_KPA = 0.636              # average surface pressure
TARGET_INTERNAL_KPA = 101.3           # nominal cabin pressure (1 atm)
DEADBAND_KPA = 0.5                    # +/-0.5 kPa regulation band
MIN_SAFE_KPA = 52.0                   # minimum survivable (pure O2 at ~0.5 atm)
CRITICAL_KPA = 30.0                   # loss of consciousness threshold

# Gas properties (simplified dry air at cabin temp)
AIR_MOLAR_MASS_KG = 0.029             # kg/mol (N2/O2 mix)
GAS_CONSTANT_J = 8.314                # J/(mol*K)
CABIN_TEMP_K = 293.15                 # 20 C nominal

# Leak parameters
BASE_LEAK_RATE_KG_SOL = 0.5           # ISS-class micro-leak (kg of air per sol)
SEAL_DEGRADATION_PER_SOL = 0.00005    # seal quality decay rate
DUST_SEAL_DAMAGE_FACTOR = 2.0         # dust storms double seal degradation
MIN_SEAL_QUALITY = 0.05               # seals never go to zero (still a hull)

# Airlock
AIRLOCK_VOLUME_M3 = 3.0               # volume lost per airlock cycle
AIRLOCK_RECOVERY_FRACTION = 0.85      # pump-back recovers 85% of airlock air

# Reserve tanks
RESERVE_DELIVERY_KG_SOL = 50.0        # max replenishment rate per sol

# Structural stress (thin-wall sphere approximation)
# sigma = dP * r / (2 * t)
HULL_RADIUS_M = 6.0                   # 12m diameter habitat module
HULL_THICKNESS_M = 0.012              # 12mm composite hull
YIELD_STRENGTH_MPA = 500.0            # carbon-fiber composite yield strength
SAFETY_FACTOR = 4.0                   # structural safety factor


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Habitat:
    """A pressurized habitat module.

    volume_m3: internal pressurized volume
    pressure_kpa: current internal pressure
    seal_quality: seal integrity [0.05, 1.0] -- 1.0 = factory fresh
    air_mass_kg: total mass of air inside
    """
    volume_m3: float
    pressure_kpa: float = TARGET_INTERNAL_KPA
    seal_quality: float = 1.0
    air_mass_kg: float = 0.0

    def __post_init__(self) -> None:
        self.volume_m3 = max(1.0, self.volume_m3)
        self.pressure_kpa = max(0.0, self.pressure_kpa)
        self.seal_quality = max(MIN_SEAL_QUALITY, min(1.0, self.seal_quality))
        if self.air_mass_kg <= 0:
            self.air_mass_kg = pressure_to_mass(self.pressure_kpa, self.volume_m3)


@dataclass
class ReserveTank:
    """Compressed gas reserve for replenishment.

    capacity_kg: maximum gas storage
    stored_kg: current gas in reserve
    """
    capacity_kg: float
    stored_kg: float

    def __post_init__(self) -> None:
        self.capacity_kg = max(0.0, self.capacity_kg)
        self.stored_kg = max(0.0, min(self.stored_kg, self.capacity_kg))

    def available(self) -> float:
        """Gas available for delivery (kg)."""
        return self.stored_kg

    def withdraw(self, kg: float) -> float:
        """Withdraw gas from reserve. Returns actual kg withdrawn."""
        actual = min(max(0.0, kg), self.stored_kg)
        self.stored_kg -= actual
        return actual

    def deposit(self, kg: float) -> float:
        """Deposit gas into reserve (from ISRU production). Returns actual kg stored."""
        headroom = self.capacity_kg - self.stored_kg
        actual = min(max(0.0, kg), headroom)
        self.stored_kg += actual
        return actual


# ---------------------------------------------------------------------------
# Gas physics
# ---------------------------------------------------------------------------

def pressure_to_mass(pressure_kpa: float, volume_m3: float) -> float:
    """Convert pressure + volume to gas mass using ideal gas law.

    PV = nRT -> n = PV / RT -> m = n * M
    P in Pa (kPa * 1000), V in m3, T in K.
    """
    if pressure_kpa <= 0 or volume_m3 <= 0:
        return 0.0
    pressure_pa = pressure_kpa * 1000.0
    n_mol = (pressure_pa * volume_m3) / (GAS_CONSTANT_J * CABIN_TEMP_K)
    return n_mol * AIR_MOLAR_MASS_KG


def mass_to_pressure(mass_kg: float, volume_m3: float) -> float:
    """Convert gas mass + volume to pressure using ideal gas law.

    m = n * M -> n = m / M -> P = nRT / V
    Returns pressure in kPa.
    """
    if mass_kg <= 0 or volume_m3 <= 0:
        return 0.0
    n_mol = mass_kg / AIR_MOLAR_MASS_KG
    pressure_pa = (n_mol * GAS_CONSTANT_J * CABIN_TEMP_K) / volume_m3
    return pressure_pa / 1000.0


def hull_stress_mpa(internal_kpa: float, external_kpa: float = MARS_SURFACE_KPA) -> float:
    """Hoop stress on the habitat hull (thin-wall sphere).

    sigma = dP * r / (2 * t)
    Returns stress in MPa.
    """
    delta_p_pa = max(0.0, internal_kpa - external_kpa) * 1000.0
    stress_pa = (delta_p_pa * HULL_RADIUS_M) / (2.0 * HULL_THICKNESS_M)
    return stress_pa / 1e6


def structural_safety_ratio(internal_kpa: float) -> float:
    """Ratio of yield strength to actual hull stress.

    > SAFETY_FACTOR means within design margins.
    < 1.0 means structural failure imminent.
    """
    stress = hull_stress_mpa(internal_kpa)
    if stress <= 0:
        return float('inf')
    return YIELD_STRENGTH_MPA / stress


# ---------------------------------------------------------------------------
# Leak model
# ---------------------------------------------------------------------------

def compute_leak_kg(
    habitat: Habitat,
    in_dust_storm: bool = False,
) -> float:
    """Gas lost to micro-leaks this sol (kg).

    Leak rate scales inversely with seal quality and with pressure differential.
    Dust storms accelerate seal degradation.
    """
    pressure_ratio = habitat.pressure_kpa / TARGET_INTERNAL_KPA
    seal_factor = 1.0 / max(habitat.seal_quality, MIN_SEAL_QUALITY)
    leak = BASE_LEAK_RATE_KG_SOL * pressure_ratio * seal_factor
    return max(0.0, leak)


def degrade_seals(
    habitat: Habitat,
    in_dust_storm: bool = False,
) -> float:
    """Apply one sol of seal degradation. Returns quality delta."""
    rate = SEAL_DEGRADATION_PER_SOL
    if in_dust_storm:
        rate *= DUST_SEAL_DAMAGE_FACTOR
    old = habitat.seal_quality
    habitat.seal_quality = max(MIN_SEAL_QUALITY, habitat.seal_quality - rate)
    return old - habitat.seal_quality


# ---------------------------------------------------------------------------
# Airlock
# ---------------------------------------------------------------------------

def airlock_cycle_loss_kg(habitat: Habitat) -> float:
    """Gas lost per airlock cycle (EVA egress/ingress).

    The airlock is pumped down to recover most air before opening,
    but some is always lost.
    """
    loss_volume = AIRLOCK_VOLUME_M3 * (1.0 - AIRLOCK_RECOVERY_FRACTION)
    return pressure_to_mass(habitat.pressure_kpa, loss_volume)


def perform_airlock_cycle(habitat: Habitat) -> float:
    """Execute one airlock cycle. Returns gas lost (kg).

    Mutates habitat: reduces air_mass_kg and recalculates pressure.
    """
    loss = airlock_cycle_loss_kg(habitat)
    loss = min(loss, habitat.air_mass_kg)
    habitat.air_mass_kg -= loss
    habitat.pressure_kpa = mass_to_pressure(habitat.air_mass_kg, habitat.volume_m3)
    return loss


# ---------------------------------------------------------------------------
# Emergency events
# ---------------------------------------------------------------------------

def blowout_event(
    habitat: Habitat,
    breach_area_cm2: float,
    duration_seconds: float,
) -> dict:
    """Model a hull breach (meteorite strike, seal failure).

    Uses orifice flow approximation for choked flow through a hole:
    mdot = C_d * A * P * sqrt(M / (R*T)) * f(gamma)
    Simplified: mass flow proportional to area * pressure * sqrt(molar_mass / temperature)

    Returns dict with gas_lost_kg, pressure_after, survivable (bool).
    """
    if breach_area_cm2 <= 0 or duration_seconds <= 0:
        return {
            "gas_lost_kg": 0.0,
            "pressure_before_kpa": habitat.pressure_kpa,
            "pressure_after_kpa": habitat.pressure_kpa,
            "survivable": True,
        }

    area_m2 = breach_area_cm2 / 1e4
    discharge_coeff = 0.65
    gamma_factor = 0.685  # f(gamma) for air, gamma=1.4

    pressure_pa = habitat.pressure_kpa * 1000.0
    flow_rate_kg_s = (discharge_coeff * area_m2 * pressure_pa *
                      math.sqrt(AIR_MOLAR_MASS_KG / (GAS_CONSTANT_J * CABIN_TEMP_K)) *
                      gamma_factor)

    gas_lost = flow_rate_kg_s * duration_seconds
    gas_lost = min(gas_lost, habitat.air_mass_kg * 0.95)  # can't lose more than 95%

    pressure_before = habitat.pressure_kpa
    habitat.air_mass_kg -= gas_lost
    habitat.air_mass_kg = max(0.0, habitat.air_mass_kg)
    habitat.pressure_kpa = mass_to_pressure(habitat.air_mass_kg, habitat.volume_m3)

    return {
        "gas_lost_kg": round(gas_lost, 4),
        "pressure_before_kpa": round(pressure_before, 4),
        "pressure_after_kpa": round(habitat.pressure_kpa, 4),
        "survivable": habitat.pressure_kpa >= CRITICAL_KPA,
    }


# ---------------------------------------------------------------------------
# Replenishment
# ---------------------------------------------------------------------------

def replenish_from_reserve(
    habitat: Habitat,
    reserve: ReserveTank,
    target_kpa: float = TARGET_INTERNAL_KPA,
) -> float:
    """Pump gas from reserve into habitat to reach target pressure.

    Rate-limited to RESERVE_DELIVERY_KG_SOL per sol.
    Returns kg of gas transferred.
    """
    current_mass = habitat.air_mass_kg
    target_mass = pressure_to_mass(target_kpa, habitat.volume_m3)
    deficit_kg = max(0.0, target_mass - current_mass)

    if deficit_kg <= 0:
        return 0.0

    transfer = min(deficit_kg, RESERVE_DELIVERY_KG_SOL, reserve.available())
    actual = reserve.withdraw(transfer)
    habitat.air_mass_kg += actual
    habitat.pressure_kpa = mass_to_pressure(habitat.air_mass_kg, habitat.volume_m3)
    return actual


# ---------------------------------------------------------------------------
# Per-sol tick
# ---------------------------------------------------------------------------

def tick_pressure(
    habitat: Habitat,
    reserve: ReserveTank,
    eva_count: int = 0,
    in_dust_storm: bool = False,
    breach_cm2: float = 0.0,
    breach_seconds: float = 0.0,
) -> dict:
    """Advance the habitat pressure system by one sol.

    Sequence:
      1. Degrade seals
      2. Apply micro-leaks
      3. Process airlock cycles (EVAs)
      4. Handle any blowout event
      5. Replenish from reserve tanks
      6. Calculate structural safety

    Returns a snapshot dict with all pressure metrics.

    Conservation law: air_mass change = replenished - leaked - airlock_lost - blowout_lost
    """
    pressure_start = habitat.pressure_kpa
    mass_start = habitat.air_mass_kg

    # 1. Seal degradation
    seal_delta = degrade_seals(habitat, in_dust_storm)

    # 2. Micro-leaks
    leak_kg = compute_leak_kg(habitat, in_dust_storm)
    leak_kg = min(leak_kg, habitat.air_mass_kg)
    habitat.air_mass_kg -= leak_kg
    habitat.pressure_kpa = mass_to_pressure(habitat.air_mass_kg, habitat.volume_m3)

    # 3. Airlock cycles
    airlock_total_kg = 0.0
    for _ in range(max(0, eva_count)):
        airlock_total_kg += perform_airlock_cycle(habitat)

    # 4. Blowout event
    blowout_info = blowout_event(habitat, breach_cm2, breach_seconds)

    # 5. Replenish
    replenished_kg = replenish_from_reserve(habitat, reserve)

    # 6. Structural safety
    safety = structural_safety_ratio(habitat.pressure_kpa)

    # Pressure classification
    if habitat.pressure_kpa >= TARGET_INTERNAL_KPA - DEADBAND_KPA:
        status = "nominal"
    elif habitat.pressure_kpa >= MIN_SAFE_KPA:
        status = "low"
    elif habitat.pressure_kpa >= CRITICAL_KPA:
        status = "critical"
    else:
        status = "fatal"

    # Conservation check
    mass_end = habitat.air_mass_kg
    mass_delta = mass_end - mass_start
    expected_delta = replenished_kg - leak_kg - airlock_total_kg - blowout_info["gas_lost_kg"]

    return {
        "pressure_start_kpa": round(pressure_start, 4),
        "pressure_end_kpa": round(habitat.pressure_kpa, 4),
        "pressure_status": status,
        "air_mass_kg": round(habitat.air_mass_kg, 4),
        "leak_kg": round(leak_kg, 4),
        "airlock_loss_kg": round(airlock_total_kg, 4),
        "blowout_loss_kg": blowout_info["gas_lost_kg"],
        "replenished_kg": round(replenished_kg, 4),
        "mass_balance_error_kg": round(mass_delta - expected_delta, 6),
        "seal_quality": round(habitat.seal_quality, 6),
        "seal_degradation": round(seal_delta, 6),
        "reserve_kg": round(reserve.stored_kg, 4),
        "structural_safety_ratio": round(safety, 4) if safety != float('inf') else 9999.0,
        "hull_stress_mpa": round(hull_stress_mpa(habitat.pressure_kpa), 4),
        "eva_count": eva_count,
        "survivable": blowout_info["survivable"] and habitat.pressure_kpa >= CRITICAL_KPA,
    }
