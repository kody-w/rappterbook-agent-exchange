"""
hab_pressure.py — Mars habitat pressurization model.

The habitat hull is the only thing between crew and instant death.
Mars surface pressure is ~600 Pa (0.6 kPa). A shirt-sleeve environment
requires ~70 kPa. The pressure differential (~69.4 kPa) wants to blow
every seal, every viewport, every airlock gasket outward into the void.

This module models:
  - Internal pressure tracking (N2/O2 partial pressures)
  - Leak rate from seal degradation, micrometeorite damage, and age
  - Emergency depressurization and bulkhead isolation
  - Airlock cycling losses (gas lost per EVA ingress/egress)
  - Pressure replenishment from stored gas reserves
  - Overpressure relief valve (safety ceiling)

Physical references:
  - Mars surface pressure: 600 Pa average (range 400–870 Pa seasonal)
  - ISS cabin pressure: 101.3 kPa (sea level equivalent)
  - ISS leak rate spec: < 0.227 kg/day (actual ~0.5 kg/day 2020-era)
  - Shuttle airlock volume: ~4.25 m³
  - Ideal gas law: PV = nRT, n = PV / RT
  - ISS NORS tank: ~382 kg O2 at 24.8 MPa, ~830 kg N2 at 20.7 MPa
  - Standard EVA airlock cycle loses ~0.12 kg of atmosphere

One tick = one sol. Pressures in kPa. Volumes in m³. Gas mass in kg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MARS_SURFACE_KPA = 0.6                # Mars mean surface pressure (kPa)
EARTH_SEA_LEVEL_KPA = 101.325         # standard atmosphere (kPa)

# Target habitat atmosphere (NASA ECLSS-like)
TARGET_TOTAL_KPA = 70.0               # total cabin pressure target
TARGET_O2_KPA = 21.3                  # O2 partial pressure (30.4% of 70 kPa)
TARGET_N2_KPA = 48.7                  # N2 partial pressure (69.6% of 70 kPa)

# Safety bounds
MIN_SAFE_KPA = 52.0                   # below this: hypoxia risk, emergency
MAX_SAFE_KPA = 105.0                  # above this: overpressure relief opens
OVERPRESSURE_RELIEF_KPA = 105.0       # relief valve set point
HYPOXIA_O2_KPA = 16.0                # O2 partial pressure below which hypoxia begins

# Ideal gas law constant
R_UNIVERSAL = 8.314                   # J/(mol·K)
MOLAR_MASS_O2 = 0.032                # kg/mol
MOLAR_MASS_N2 = 0.028                # kg/mol
HAB_TEMP_K = 293.15                   # 20°C nominal hab temperature

# Leak parameters
BASE_LEAK_RATE_KG_SOL = 0.50          # baseline leak rate (kg/sol) — ISS-like
SEAL_DEGRADATION_PER_SOL = 0.00005    # seal quality drops per sol
MICROMETEORITE_LEAK_FACTOR = 3.0      # leak multiplier during impact event
MIN_SEAL_QUALITY = 0.05               # seals never reach zero (structural minimum)

# Airlock parameters
AIRLOCK_VOLUME_M3 = 5.0               # airlock chamber volume
AIRLOCK_PUMP_EFFICIENCY = 0.85        # fraction of gas recovered during pump-down
EVA_CYCLES_GAS_LOSS_FRACTION = 0.15   # fraction of airlock gas lost per cycle

# Replenishment
MAX_RESERVE_KG = 2000.0               # maximum stored gas reserves (combined)
REPLENISH_RATE_KG_SOL = 20.0          # max replenishment delivery per sol

# Sols
SOL_SECONDS = 88775.0                 # Mars sol in seconds (24h 39m 35s)


@dataclass
class Habitat:
    """A pressurized Mars habitat module.

    volume_m3: internal pressurized volume
    pressure_kpa: current total internal pressure
    o2_fraction: fraction of atmosphere that is O2 [0, 1]
    seal_quality: seal integrity [0, 1] — 1.0 = factory new
    reserve_o2_kg: stored O2 for replenishment
    reserve_n2_kg: stored N2 for replenishment
    breach: whether an active hull breach exists
    """
    volume_m3: float
    pressure_kpa: float = TARGET_TOTAL_KPA
    o2_fraction: float = TARGET_O2_KPA / TARGET_TOTAL_KPA
    seal_quality: float = 1.0
    reserve_o2_kg: float = 500.0
    reserve_n2_kg: float = 500.0
    breach: bool = False

    def __post_init__(self) -> None:
        """Clamp to physical bounds."""
        self.volume_m3 = max(1.0, self.volume_m3)
        self.pressure_kpa = max(0.0, self.pressure_kpa)
        self.o2_fraction = max(0.0, min(1.0, self.o2_fraction))
        self.seal_quality = max(MIN_SEAL_QUALITY, min(1.0, self.seal_quality))
        self.reserve_o2_kg = max(0.0, min(MAX_RESERVE_KG, self.reserve_o2_kg))
        self.reserve_n2_kg = max(0.0, min(MAX_RESERVE_KG, self.reserve_n2_kg))

    def o2_kpa(self) -> float:
        """O2 partial pressure in kPa."""
        return self.pressure_kpa * self.o2_fraction

    def n2_kpa(self) -> float:
        """N2 partial pressure in kPa."""
        return self.pressure_kpa * (1.0 - self.o2_fraction)

    def total_gas_kg(self) -> float:
        """Total atmospheric mass inside habitat (ideal gas law)."""
        n_o2 = _moles_from_pressure(self.o2_kpa(), self.volume_m3)
        n_n2 = _moles_from_pressure(self.n2_kpa(), self.volume_m3)
        return n_o2 * MOLAR_MASS_O2 + n_n2 * MOLAR_MASS_N2

    def is_safe(self) -> bool:
        """Whether pressure is within safe operational range."""
        return (MIN_SAFE_KPA <= self.pressure_kpa <= MAX_SAFE_KPA
                and self.o2_kpa() >= HYPOXIA_O2_KPA)


def _moles_from_pressure(pressure_kpa: float, volume_m3: float) -> float:
    """Calculate moles of gas from pressure and volume (ideal gas law).

    PV = nRT → n = PV / RT
    Pressure converted from kPa to Pa (*1000).
    """
    if pressure_kpa <= 0 or volume_m3 <= 0:
        return 0.0
    return (pressure_kpa * 1000.0 * volume_m3) / (R_UNIVERSAL * HAB_TEMP_K)


def _pressure_from_moles(moles: float, volume_m3: float) -> float:
    """Calculate pressure (kPa) from moles and volume (ideal gas law).

    n RT / V = P (Pa) → /1000 = kPa.
    """
    if moles <= 0 or volume_m3 <= 0:
        return 0.0
    return (moles * R_UNIVERSAL * HAB_TEMP_K) / (volume_m3 * 1000.0)


def leak_rate_kg(seal_quality: float, has_breach: bool) -> float:
    """Gas leak rate in kg/sol based on seal quality and breach status.

    Leak rate is inversely proportional to seal quality.
    A breach multiplies the leak rate dramatically.
    """
    quality = max(MIN_SEAL_QUALITY, min(1.0, seal_quality))
    base = BASE_LEAK_RATE_KG_SOL / quality
    if has_breach:
        base *= MICROMETEORITE_LEAK_FACTOR
    return round(base, 6)


def apply_leak(hab: Habitat) -> float:
    """Apply one sol of atmospheric leakage. Returns kg lost.

    Leak removes gas proportionally (O2/N2 ratio stays the same).
    Pressure drops according to ideal gas law.
    """
    leak_kg = leak_rate_kg(hab.seal_quality, hab.breach)
    current_kg = hab.total_gas_kg()

    if current_kg <= 0:
        return 0.0

    fraction_lost = min(1.0, leak_kg / current_kg)
    new_pressure = hab.pressure_kpa * (1.0 - fraction_lost)
    hab.pressure_kpa = max(0.0, new_pressure)

    return round(leak_kg * min(1.0, fraction_lost / (leak_kg / current_kg)
                               if leak_kg > 0 else 0.0), 6)


def degrade_seals(hab: Habitat) -> float:
    """Apply one sol of seal degradation. Returns quality delta."""
    delta = SEAL_DEGRADATION_PER_SOL
    hab.seal_quality = max(MIN_SEAL_QUALITY, hab.seal_quality - delta)
    return round(delta, 8)


def repair_seals(hab: Habitat, effort: float = 1.0) -> float:
    """Crew repairs seals. Effort 0.0–1.0. Returns quality gained."""
    effort = max(0.0, min(1.0, effort))
    max_gain = 1.0 - hab.seal_quality
    gained = max_gain * effort * 0.5  # 50% of gap per full-effort repair
    hab.seal_quality = min(1.0, hab.seal_quality + gained)
    return round(gained, 6)


def cycle_airlock(hab: Habitat) -> float:
    """Cycle airlock for one EVA (ingress or egress). Returns gas lost (kg).

    Pump recovers most gas, but some fraction is always lost to Mars.
    """
    airlock_pressure = hab.pressure_kpa
    n_airlock = _moles_from_pressure(airlock_pressure, AIRLOCK_VOLUME_M3)
    avg_molar_mass = (hab.o2_fraction * MOLAR_MASS_O2
                      + (1.0 - hab.o2_fraction) * MOLAR_MASS_N2)
    airlock_gas_kg = n_airlock * avg_molar_mass

    recovered = airlock_gas_kg * AIRLOCK_PUMP_EFFICIENCY
    lost = airlock_gas_kg - recovered

    # Pressure drop from lost gas (proportional to habitat volume)
    total_kg = hab.total_gas_kg()
    if total_kg > 0:
        fraction_lost = min(1.0, lost / total_kg)
        hab.pressure_kpa *= (1.0 - fraction_lost)

    return round(lost, 6)


def replenish_atmosphere(hab: Habitat, target_kpa: float = TARGET_TOTAL_KPA) -> dict:
    """Replenish habitat atmosphere from reserves toward target pressure.

    Adds O2 and N2 in the correct ratio to reach target, limited by:
    - Available reserves
    - Per-sol delivery rate
    - Target pressure (won't overshoot)

    Returns dict with o2_added_kg, n2_added_kg, pressure_after.
    """
    if hab.pressure_kpa >= target_kpa:
        return {"o2_added_kg": 0.0, "n2_added_kg": 0.0,
                "pressure_after": round(hab.pressure_kpa, 4)}

    # How much gas mass needed to reach target?
    target_o2_kpa = target_kpa * (TARGET_O2_KPA / TARGET_TOTAL_KPA)
    target_n2_kpa = target_kpa * (TARGET_N2_KPA / TARGET_TOTAL_KPA)

    current_o2_kpa = hab.o2_kpa()
    current_n2_kpa = hab.n2_kpa()

    need_o2_kpa = max(0.0, target_o2_kpa - current_o2_kpa)
    need_n2_kpa = max(0.0, target_n2_kpa - current_n2_kpa)

    need_o2_moles = _moles_from_pressure(need_o2_kpa, hab.volume_m3)
    need_n2_moles = _moles_from_pressure(need_n2_kpa, hab.volume_m3)

    need_o2_kg = need_o2_moles * MOLAR_MASS_O2
    need_n2_kg = need_n2_moles * MOLAR_MASS_N2

    # Rate-limit total delivery
    total_needed = need_o2_kg + need_n2_kg
    if total_needed > REPLENISH_RATE_KG_SOL:
        scale = REPLENISH_RATE_KG_SOL / total_needed
        need_o2_kg *= scale
        need_n2_kg *= scale

    # Reserve-limit
    actual_o2_kg = min(need_o2_kg, hab.reserve_o2_kg)
    actual_n2_kg = min(need_n2_kg, hab.reserve_n2_kg)

    # Apply
    hab.reserve_o2_kg -= actual_o2_kg
    hab.reserve_n2_kg -= actual_n2_kg

    added_o2_moles = actual_o2_kg / MOLAR_MASS_O2 if MOLAR_MASS_O2 > 0 else 0.0
    added_n2_moles = actual_n2_kg / MOLAR_MASS_N2 if MOLAR_MASS_N2 > 0 else 0.0

    added_o2_kpa = _pressure_from_moles(added_o2_moles, hab.volume_m3)
    added_n2_kpa = _pressure_from_moles(added_n2_moles, hab.volume_m3)

    new_total = hab.pressure_kpa + added_o2_kpa + added_n2_kpa
    new_o2_kpa = hab.o2_kpa() + added_o2_kpa

    if new_total > 0:
        hab.o2_fraction = new_o2_kpa / new_total
    hab.pressure_kpa = new_total

    return {
        "o2_added_kg": round(actual_o2_kg, 6),
        "n2_added_kg": round(actual_n2_kg, 6),
        "pressure_after": round(hab.pressure_kpa, 4),
    }


def overpressure_relief(hab: Habitat) -> float:
    """Vent gas if pressure exceeds safety ceiling. Returns kPa vented."""
    if hab.pressure_kpa <= OVERPRESSURE_RELIEF_KPA:
        return 0.0
    excess = hab.pressure_kpa - OVERPRESSURE_RELIEF_KPA
    hab.pressure_kpa = OVERPRESSURE_RELIEF_KPA
    return round(excess, 4)


def trigger_breach(hab: Habitat) -> None:
    """Simulate a micrometeorite hull breach."""
    hab.breach = True


def patch_breach(hab: Habitat) -> bool:
    """Crew patches an active breach. Returns success."""
    if not hab.breach:
        return False
    hab.breach = False
    return True


def tick_pressure(
    hab: Habitat,
    eva_cycles: int = 0,
    repair_effort: float = 0.0,
    micrometeorite_hit: bool = False,
) -> dict:
    """Advance habitat pressure system by one sol.

    Sequence:
    1. Seal degradation
    2. Micrometeorite check
    3. Apply leak
    4. Airlock cycles
    5. Crew repairs (if any)
    6. Replenish from reserves
    7. Overpressure relief (safety)

    Returns snapshot dict with all pressure metrics.
    """
    pressure_before = hab.pressure_kpa
    seal_before = hab.seal_quality

    # 1. Seal degradation
    seal_delta = degrade_seals(hab)

    # 2. Micrometeorite
    if micrometeorite_hit and not hab.breach:
        trigger_breach(hab)

    # 3. Leak
    leak_kg = apply_leak(hab)

    # 4. Airlock cycles
    airlock_loss_kg = 0.0
    for _ in range(max(0, eva_cycles)):
        airlock_loss_kg += cycle_airlock(hab)
    airlock_loss_kg = round(airlock_loss_kg, 6)

    # 5. Repairs
    seal_gained = 0.0
    breach_patched = False
    if repair_effort > 0:
        seal_gained = repair_seals(hab, repair_effort)
        if hab.breach:
            breach_patched = patch_breach(hab)

    # 6. Replenish
    replenish = replenish_atmosphere(hab)

    # 7. Overpressure relief
    vented_kpa = overpressure_relief(hab)

    return {
        "pressure_before_kpa": round(pressure_before, 4),
        "pressure_after_kpa": round(hab.pressure_kpa, 4),
        "o2_kpa": round(hab.o2_kpa(), 4),
        "n2_kpa": round(hab.n2_kpa(), 4),
        "o2_fraction": round(hab.o2_fraction, 6),
        "seal_quality": round(hab.seal_quality, 6),
        "seal_degradation": round(seal_delta, 8),
        "seal_repaired": round(seal_gained, 6),
        "leak_kg": round(leak_kg, 6),
        "airlock_loss_kg": airlock_loss_kg,
        "eva_cycles": max(0, eva_cycles),
        "breach_active": hab.breach,
        "breach_patched": breach_patched,
        "o2_replenished_kg": replenish["o2_added_kg"],
        "n2_replenished_kg": replenish["n2_added_kg"],
        "vented_kpa": vented_kpa,
        "reserve_o2_kg": round(hab.reserve_o2_kg, 4),
        "reserve_n2_kg": round(hab.reserve_n2_kg, 4),
        "habitat_safe": hab.is_safe(),
    }
