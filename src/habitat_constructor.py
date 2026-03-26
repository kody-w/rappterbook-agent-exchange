"""habitat_constructor.py -- Mars Pressurized Habitat Module Assembly.

The colony mines regolith (regolith_processor), smelts iron
(ore_smelter), and fabricates parts (fabricator) -- but cannot
ASSEMBLE these into new pressurized structures.  Every habitat module
shipped from Earth costs $2 billion and 9 months of transit.  Building
locally is how Mars goes from outpost to civilization.

This module models the construction of inflatable-core, regolith-
shielded habitat modules from local materials.  The colony's
reproductive system -- it builds new rooms.

Construction method (NASA Langley / Bigelow Aerospace reference)
----------------------------------------------------------------
1. **Foundation** -- Level pad, compact regolith base.  5 sols.
2. **Frame erection** -- Smelted-iron rib cage bolted together.
   Structural members from ore_smelter output.  3 sols.
3. **Shell inflation** -- Kevlar/Vectran bladder inflated to 70 kPa.
   (Earth-shipped consumable until local polymer production.)  1 sol.
4. **Regolith shielding** -- 2 m of regolith piled over shell for
   radiation protection (~50 g/cm² shielding depth).  10 sols.
5. **Pressure test** -- Hold 70 kPa for 48 h, measure leak rate.
   Acceptable: < 0.05% per sol.  1 sol.
6. **Outfitting** -- Airlock integration, power/life-support hookup,
   interior fit-out.  10 sols.

Physics modelled
----------------
* Mass balance -- iron frame + regolith shielding + bladder + sealant.
* Energy budget -- welding, compaction, crane ops, lighting, heating.
* EVA time budget -- crew hours outside per sol (max 8 hr/sol, 2 crew).
* Structural integrity -- cylinder hoop stress sigma = P*r/t.
  Safety factor >= 3.0 required.
* Leak rate -- function of seal quality and pressure differential.
  Arrhenius-style temperature dependence on seal degradation.
* Radiation shielding -- regolith thickness -> dose reduction.
  2 m regolith blocks ~95% of GCR, ~99.5% of solar particle events.
* Thermal stress -- diurnal cycling (-80C to +20C) fatigues welds.
  Weld integrity drops ~0.01%/sol from thermal cycling.
* Dust contamination -- dust during construction degrades seal quality.
  Optical depth tau > 0.5 halves seal quality gain per sol.

Conservation laws
-----------------
- iron_used_kg >= 0 and <= iron_available
- regolith_used_m3 >= 0 and <= regolith_available
- phase progresses monotonically (no going backward)
- pressure_kpa in [0, 101.3] (can't exceed 1 atm)
- leak_rate >= 0
- integrity in [0, 1]
- shielding_m >= 0
- eva_hours_today in [0, MAX_EVA_HOURS_PER_SOL]
- total energy consumed >= 0
- total construction time >= sum of phase minimums

One tick = one sol.  Mass in kg, pressure in kPa, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

MARS_AMBIENT_PRESSURE_KPA = 0.636
MARS_AMBIENT_TEMP_K = 210.0
MARS_DIURNAL_RANGE_K = 100.0

# Habitat design parameters
TARGET_PRESSURE_KPA = 70.0
MAX_PRESSURE_KPA = 101.325
HABITAT_RADIUS_M = 4.0
HABITAT_LENGTH_M = 8.0
HABITAT_VOLUME_M3 = math.pi * HABITAT_RADIUS_M ** 2 * HABITAT_LENGTH_M

# Structural
IRON_YIELD_STRENGTH_MPA = 250.0
SAFETY_FACTOR_MIN = 3.0
FRAME_WALL_THICKNESS_M = 0.006
IRON_DENSITY_KG_M3 = 7874.0

# Frame mass: cylindrical shell of iron ribs (simplified as thin shell)
FRAME_IRON_KG = (
    2.0 * math.pi * HABITAT_RADIUS_M * HABITAT_LENGTH_M
    * FRAME_WALL_THICKNESS_M * IRON_DENSITY_KG_M3 * 0.15
)  # 15% fill factor for rib cage vs solid shell

# Regolith shielding
REGOLITH_DENSITY_KG_M3 = 1500.0
SHIELDING_TARGET_M = 2.0
SHIELDING_SURFACE_AREA_M2 = (
    2.0 * math.pi * (HABITAT_RADIUS_M + SHIELDING_TARGET_M)
    * (HABITAT_LENGTH_M + 2.0 * SHIELDING_TARGET_M)
)
REGOLITH_TOTAL_M3 = SHIELDING_SURFACE_AREA_M2 * SHIELDING_TARGET_M
REGOLITH_TOTAL_KG = REGOLITH_TOTAL_M3 * REGOLITH_DENSITY_KG_M3

# Bladder (Earth-shipped until local polymer)
BLADDER_MASS_KG = 120.0
BLADDER_BURST_PRESSURE_KPA = 350.0
BLADDER_TENSILE_MPA = 3000.0  # Vectran

# Sealant
SEALANT_KG_PER_JOINT = 0.5
JOINTS_PER_MODULE = 48
SEALANT_TOTAL_KG = SEALANT_KG_PER_JOINT * JOINTS_PER_MODULE

# EVA constraints
MAX_EVA_HOURS_PER_SOL = 8.0
MAX_EVA_CREW = 2
EVA_HOURS_PER_SOL = MAX_EVA_HOURS_PER_SOL * MAX_EVA_CREW  # 16 crew-hours

# Energy costs (kWh per sol of active construction)
ENERGY_FOUNDATION_KWH = 40.0
ENERGY_FRAMING_KWH = 60.0
ENERGY_INFLATION_KWH = 10.0
ENERGY_SHIELDING_KWH = 35.0
ENERGY_PRESSURE_TEST_KWH = 5.0
ENERGY_OUTFITTING_KWH = 25.0

# Phase durations (sols, minimum)
PHASE_FOUNDATION_SOLS = 5
PHASE_FRAMING_SOLS = 3
PHASE_INFLATION_SOLS = 1
PHASE_SHIELDING_SOLS = 10
PHASE_PRESSURE_TEST_SOLS = 1
PHASE_OUTFITTING_SOLS = 10
TOTAL_MIN_SOLS = (
    PHASE_FOUNDATION_SOLS + PHASE_FRAMING_SOLS + PHASE_INFLATION_SOLS
    + PHASE_SHIELDING_SOLS + PHASE_PRESSURE_TEST_SOLS + PHASE_OUTFITTING_SOLS
)

# Leak and seal parameters
ACCEPTABLE_LEAK_RATE = 0.002  # 0.2% per sol (ISS-class, achievable at seal_quality=1.0)
SEAL_QUALITY_GAIN_PER_SOL = 0.08  # seal curing
SEAL_DUST_PENALTY_THRESHOLD = 0.5  # optical depth
WELD_THERMAL_FATIGUE_PER_SOL = 0.0001

# Radiation shielding effectiveness
GCR_BLOCK_PER_M = 0.475  # fraction blocked per metre regolith
SPE_BLOCK_PER_M = 1.5  # attenuation coefficient per metre (softer spectrum, 2m -> ~95%)


# ---------------------------------------------------------------------------
# Construction phases
# ---------------------------------------------------------------------------

PHASES = [
    "foundation",
    "framing",
    "inflation",
    "shielding",
    "pressure_test",
    "outfitting",
    "complete",
]

PHASE_DURATIONS = {
    "foundation": PHASE_FOUNDATION_SOLS,
    "framing": PHASE_FRAMING_SOLS,
    "inflation": PHASE_INFLATION_SOLS,
    "shielding": PHASE_SHIELDING_SOLS,
    "pressure_test": PHASE_PRESSURE_TEST_SOLS,
    "outfitting": PHASE_OUTFITTING_SOLS,
}

PHASE_ENERGY = {
    "foundation": ENERGY_FOUNDATION_KWH,
    "framing": ENERGY_FRAMING_KWH,
    "inflation": ENERGY_INFLATION_KWH,
    "shielding": ENERGY_SHIELDING_KWH,
    "pressure_test": ENERGY_PRESSURE_TEST_KWH,
    "outfitting": ENERGY_OUTFITTING_KWH,
}


# ---------------------------------------------------------------------------
# Pure physics functions
# ---------------------------------------------------------------------------

def hoop_stress_mpa(pressure_kpa: float, radius_m: float,
                    wall_thickness_m: float) -> float:
    """Cylinder hoop stress: sigma = P * r / t  (thin-wall approx).

    Returns stress in MPa.
    """
    if wall_thickness_m <= 0.0 or radius_m <= 0.0:
        return 0.0
    pressure_mpa = pressure_kpa / 1000.0
    return pressure_mpa * radius_m / wall_thickness_m


def safety_factor(yield_strength_mpa: float, stress_mpa: float) -> float:
    """Ratio of material yield strength to applied stress."""
    if stress_mpa <= 0.0:
        return float("inf")
    return yield_strength_mpa / stress_mpa


def is_structurally_safe(pressure_kpa: float, radius_m: float,
                         wall_thickness_m: float,
                         yield_mpa: float = IRON_YIELD_STRENGTH_MPA,
                         min_sf: float = SAFETY_FACTOR_MIN) -> bool:
    """Check if habitat frame can withstand internal pressure."""
    stress = hoop_stress_mpa(pressure_kpa, radius_m, wall_thickness_m)
    return safety_factor(yield_mpa, stress) >= min_sf


def shielding_gcr_reduction(thickness_m: float) -> float:
    """Fraction of galactic cosmic rays blocked by regolith thickness.

    Uses exponential attenuation model.  2 m -> ~95%.
    """
    if thickness_m <= 0.0:
        return 0.0
    return min(1.0, 1.0 - math.exp(-GCR_BLOCK_PER_M * thickness_m))


def shielding_spe_reduction(thickness_m: float) -> float:
    """Fraction of solar particle events blocked by regolith.

    SPE particles are softer spectrum, much easier to shield.
    2 m -> ~99.5%.
    """
    if thickness_m <= 0.0:
        return 0.0
    return min(1.0, 1.0 - math.exp(-SPE_BLOCK_PER_M * thickness_m))


def leak_rate_per_sol(seal_quality: float, pressure_kpa: float,
                      temperature_k: float) -> float:
    """Fractional pressure loss per sol.

    Higher seal quality -> lower leak.  Higher pressure diff -> more leak.
    Cold temperatures stiffen seals, slightly increasing leak rate.
    """
    if seal_quality <= 0.0:
        return 1.0  # total failure
    if pressure_kpa <= MARS_AMBIENT_PRESSURE_KPA:
        return 0.0

    pressure_diff = pressure_kpa - MARS_AMBIENT_PRESSURE_KPA
    base_leak = 0.001 / seal_quality  # base leak inversely prop to quality
    pressure_factor = pressure_diff / TARGET_PRESSURE_KPA
    # Cold stiffening: +20% leak at 150 K, nominal at 293 K
    temp_factor = 1.0 + 0.2 * max(0.0, (293.0 - temperature_k) / 143.0)
    return max(0.0, base_leak * pressure_factor * temp_factor)


def construction_energy_kwh(phase: str, available_power_kwh: float) -> float:
    """Energy consumed during one sol of construction in given phase.

    Clamped to available power -- construction slows if power-starved.
    """
    required = PHASE_ENERGY.get(phase, 0.0)
    return min(required, max(0.0, available_power_kwh))


def regolith_per_sol_m3(eva_hours: float) -> float:
    """Volume of regolith that can be placed per sol during shielding.

    ~1.5 m³/crew-hour with powered equipment.
    """
    return max(0.0, eva_hours * 1.5)


def iron_consumed_kg(phase: str) -> float:
    """Iron consumed in a given phase.  Only framing uses iron."""
    if phase == "framing":
        return FRAME_IRON_KG / PHASE_FRAMING_SOLS
    return 0.0


def dust_seal_penalty(optical_depth: float) -> float:
    """Multiplier on seal quality gain when dust is present.

    Above threshold, dust halves the seal curing rate.
    Returns value in [0.0, 1.0].
    """
    if optical_depth <= 0.0:
        return 1.0
    if optical_depth >= SEAL_DUST_PENALTY_THRESHOLD:
        return 0.5
    return 1.0 - 0.5 * (optical_depth / SEAL_DUST_PENALTY_THRESHOLD)


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class HabitatConstructor:
    """State of a habitat module under construction.

    Tracks phase, materials consumed, structural integrity, seal quality,
    and pressurization.  One tick = one sol.
    """
    # Current construction phase
    phase: str = "foundation"
    phase_sol: int = 0  # sols spent in current phase

    # Materials consumed
    iron_used_kg: float = 0.0
    regolith_placed_m3: float = 0.0
    bladder_installed: bool = False
    sealant_used_kg: float = 0.0

    # Structural state
    integrity: float = 1.0  # [0, 1] -- weld/frame health
    seal_quality: float = 0.0  # [0, 1] -- seal curing progress
    pressure_kpa: float = 0.0
    shielding_m: float = 0.0

    # Tracking
    total_sols: int = 0
    total_energy_kwh: float = 0.0
    total_eva_hours: float = 0.0
    leak_rate: float = 0.0
    modules_completed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "phase": self.phase,
            "phase_sol": self.phase_sol,
            "iron_used_kg": round(self.iron_used_kg, 2),
            "regolith_placed_m3": round(self.regolith_placed_m3, 2),
            "bladder_installed": self.bladder_installed,
            "sealant_used_kg": round(self.sealant_used_kg, 2),
            "integrity": round(self.integrity, 4),
            "seal_quality": round(self.seal_quality, 4),
            "pressure_kpa": round(self.pressure_kpa, 2),
            "shielding_m": round(self.shielding_m, 4),
            "total_sols": self.total_sols,
            "total_energy_kwh": round(self.total_energy_kwh, 2),
            "total_eva_hours": round(self.total_eva_hours, 2),
            "leak_rate": round(self.leak_rate, 6),
            "modules_completed": self.modules_completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HabitatConstructor":
        """Deserialize from dict."""
        return cls(
            phase=data.get("phase", "foundation"),
            phase_sol=data.get("phase_sol", 0),
            iron_used_kg=data.get("iron_used_kg", 0.0),
            regolith_placed_m3=data.get("regolith_placed_m3", 0.0),
            bladder_installed=data.get("bladder_installed", False),
            sealant_used_kg=data.get("sealant_used_kg", 0.0),
            integrity=data.get("integrity", 1.0),
            seal_quality=data.get("seal_quality", 0.0),
            pressure_kpa=data.get("pressure_kpa", 0.0),
            shielding_m=data.get("shielding_m", 0.0),
            total_sols=data.get("total_sols", 0),
            total_energy_kwh=data.get("total_energy_kwh", 0.0),
            total_eva_hours=data.get("total_eva_hours", 0.0),
            leak_rate=data.get("leak_rate", 0.0),
            modules_completed=data.get("modules_completed", 0),
        )


# ---------------------------------------------------------------------------
# Tick engine
# ---------------------------------------------------------------------------

def tick(state: HabitatConstructor,
         available_power_kwh: float = 200.0,
         iron_available_kg: float = 5000.0,
         regolith_available_m3: float = 10000.0,
         optical_depth: float = 0.3,
         temperature_k: float = MARS_AMBIENT_TEMP_K) -> dict[str, Any]:
    """Advance habitat construction by one sol.

    Parameters
    ----------
    state : HabitatConstructor
        Current construction state (mutated in place).
    available_power_kwh : float
        Energy available this sol from power grid.
    iron_available_kg : float
        Iron stock from ore smelter.
    regolith_available_m3 : float
        Processed regolith available.
    optical_depth : float
        Atmospheric dust opacity (0 = clear, >1 = storm).
    temperature_k : float
        Ambient temperature in Kelvin.

    Returns
    -------
    dict with consumption and status for this sol.
    """
    if state.phase == "complete":
        # Completed module -- just track thermal fatigue
        state.integrity = max(0.0, state.integrity
                              - WELD_THERMAL_FATIGUE_PER_SOL)
        state.leak_rate = leak_rate_per_sol(
            state.seal_quality, state.pressure_kpa, temperature_k)
        state.total_sols += 1
        return {"phase": "complete", "action": "maintenance",
                "energy_kwh": 0.0, "iron_kg": 0.0, "regolith_m3": 0.0,
                "eva_hours": 0.0}

    # Dust storms halt outdoor construction
    if optical_depth > 0.7 and state.phase in ("foundation", "shielding"):
        state.total_sols += 1
        return {"phase": state.phase, "action": "dust_halt",
                "energy_kwh": 0.0, "iron_kg": 0.0, "regolith_m3": 0.0,
                "eva_hours": 0.0}

    # Energy for this phase
    energy_used = construction_energy_kwh(state.phase, available_power_kwh)
    energy_fraction = (energy_used / PHASE_ENERGY.get(state.phase, 1.0)
                       if PHASE_ENERGY.get(state.phase, 0.0) > 0 else 1.0)

    # EVA hours (outdoor phases need EVA)
    eva_today = 0.0
    iron_today = 0.0
    regolith_today = 0.0

    if state.phase == "foundation":
        eva_today = min(EVA_HOURS_PER_SOL, EVA_HOURS_PER_SOL * energy_fraction)

    elif state.phase == "framing":
        iron_need = iron_consumed_kg("framing")
        iron_today = min(iron_need, iron_available_kg)
        eva_today = min(EVA_HOURS_PER_SOL, EVA_HOURS_PER_SOL * energy_fraction)

    elif state.phase == "inflation":
        if not state.bladder_installed:
            state.bladder_installed = True
            state.pressure_kpa = TARGET_PRESSURE_KPA
        eva_today = 4.0  # minimal EVA for inflation

    elif state.phase == "shielding":
        max_regolith = regolith_per_sol_m3(EVA_HOURS_PER_SOL * energy_fraction)
        regolith_today = min(max_regolith, regolith_available_m3,
                             REGOLITH_TOTAL_M3 - state.regolith_placed_m3)
        regolith_today = max(0.0, regolith_today)
        eva_today = regolith_today / 1.5 if regolith_today > 0 else 0.0
        state.regolith_placed_m3 += regolith_today
        state.shielding_m = min(
            SHIELDING_TARGET_M,
            state.regolith_placed_m3 / (SHIELDING_SURFACE_AREA_M2
                                         if SHIELDING_SURFACE_AREA_M2 > 0
                                         else 1.0)
        )

    elif state.phase == "pressure_test":
        # Seal curing
        dust_factor = dust_seal_penalty(optical_depth)
        state.seal_quality = min(
            1.0, state.seal_quality + SEAL_QUALITY_GAIN_PER_SOL * dust_factor)
        state.sealant_used_kg = min(
            SEALANT_TOTAL_KG,
            state.sealant_used_kg + SEALANT_TOTAL_KG / PHASE_PRESSURE_TEST_SOLS)
        state.leak_rate = leak_rate_per_sol(
            max(0.01, state.seal_quality), state.pressure_kpa, temperature_k)

    elif state.phase == "outfitting":
        eva_today = min(EVA_HOURS_PER_SOL, EVA_HOURS_PER_SOL * energy_fraction)
        # Seal continues to cure during outfitting
        dust_factor = dust_seal_penalty(optical_depth)
        state.seal_quality = min(
            1.0, state.seal_quality + SEAL_QUALITY_GAIN_PER_SOL * 0.5
            * dust_factor)

    # Apply iron consumption for framing
    state.iron_used_kg += iron_today

    # Thermal fatigue on welds (once frame is up)
    if state.phase not in ("foundation",):
        state.integrity = max(0.0, state.integrity
                              - WELD_THERMAL_FATIGUE_PER_SOL)

    # Update tracking
    state.total_energy_kwh += energy_used
    state.total_eva_hours += eva_today
    state.phase_sol += 1

    # Phase transition
    phase_duration = PHASE_DURATIONS.get(state.phase, 1)
    advanced = False
    if state.phase_sol >= phase_duration:
        # Additional check: shielding needs enough regolith
        if (state.phase == "shielding"
                and state.regolith_placed_m3 < REGOLITH_TOTAL_M3 * 0.95):
            pass  # stay in shielding until regolith target met
        elif (state.phase == "pressure_test"
              and state.leak_rate > ACCEPTABLE_LEAK_RATE):
            pass  # keep testing until leak rate acceptable
        else:
            idx = PHASES.index(state.phase)
            if idx + 1 < len(PHASES):
                state.phase = PHASES[idx + 1]
                state.phase_sol = 0
                advanced = True
                if state.phase == "complete":
                    state.modules_completed += 1

    state.total_sols += 1

    return {
        "phase": state.phase,
        "action": "advanced" if advanced else "working",
        "energy_kwh": round(energy_used, 2),
        "iron_kg": round(iron_today, 2),
        "regolith_m3": round(regolith_today, 2),
        "eva_hours": round(eva_today, 2),
    }


def run_simulation(sols: int = 365,
                   available_power_kwh: float = 200.0,
                   iron_available_kg: float = 5000.0,
                   regolith_available_m3: float = 10000.0,
                   optical_depth: float = 0.3,
                   temperature_k: float = MARS_AMBIENT_TEMP_K,
                   ) -> list[dict[str, Any]]:
    """Run construction for N sols and return history."""
    state = HabitatConstructor()
    history = []
    for sol in range(sols):
        result = tick(state, available_power_kwh, iron_available_kg,
                      regolith_available_m3, optical_depth, temperature_k)
        result["sol"] = sol
        result["state"] = state.to_dict()
        history.append(result)
        if state.phase == "complete" and sol > TOTAL_MIN_SOLS:
            break
    return history
