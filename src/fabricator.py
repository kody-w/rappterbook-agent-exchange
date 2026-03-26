"""fabricator.py -- Mars Colony 3D Fabrication Lab.

The fabricator turns raw materials into replacement parts, tools, and
structural components. Without it, every broken seal gasket, cracked
viewport, or worn drill bit means waiting 6-9 months for a resupply
from Earth. With it, the colony manufactures its own survival.

Physics modelled
----------------
* **Energy cost** -- sintering regolith or melting feedstock requires
  significant energy. Power draw scales with material type and volume.
  Regolith sintering: ~2.5 MJ/kg. Metal printing: ~5 MJ/kg.
  Polymer extrusion: ~1.2 MJ/kg.
* **Print time** -- function of part volume, layer height, and material.
  Larger parts take longer. Finer resolution = slower but higher quality.
* **Material consumption** -- feedstock is consumed proportionally.
  Waste factor accounts for support structures and failed prints.
* **Quality/tolerance** -- part quality depends on printer calibration,
  material purity, and environmental vibration. Quality degrades the
  printer nozzle/laser over time.
* **Nozzle/laser wear** -- print head degrades with use. Below threshold,
  print quality drops. Replacement requires spare parts (bootstrap problem).
* **Thermal management** -- printer bed must maintain temperature.
  Mars cold makes this harder. Enclosure required.

Physical references:
  - Sintering energy for regolith: ~2-3 MJ/kg (NASA studies)
  - Metal 3D printing (DMLS): ~5-10 MJ/kg
  - FDM polymer printing: ~1-2 MJ/kg
  - Mars regolith composition: ~45% SiO2, ~18% Fe2O3, ~8% Al2O3
  - Print resolution: 0.1-0.5 mm layer height typical
  - Build volume: ~0.5 m^3 for colony-scale printer

One tick = one sol. Energy in kWh. Mass in kg. Time in hours.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Energy per kg by material type (kWh)
ENERGY_REGOLITH_KWH_KG = 0.694       # ~2.5 MJ/kg
ENERGY_METAL_KWH_KG = 1.389          # ~5.0 MJ/kg
ENERGY_POLYMER_KWH_KG = 0.333        # ~1.2 MJ/kg

# Print speed (kg per hour by material)
SPEED_REGOLITH_KG_HR = 0.8
SPEED_METAL_KG_HR = 0.3
SPEED_POLYMER_KG_HR = 1.2

# Waste factor (fraction of feedstock wasted as supports/failures)
WASTE_REGOLITH = 0.15
WASTE_METAL = 0.10
WASTE_POLYMER = 0.20

# Quality
QUALITY_NEW = 1.0
QUALITY_MIN = 0.20                    # below this, parts are unusable
NOZZLE_WEAR_PER_KG = 0.002           # quality loss per kg printed
CALIBRATION_RESTORE = 0.60           # fraction of quality gap restored

# Thermal
BED_TEMP_TARGET_C = 80.0             # heated bed target
BED_HEATER_POWER_KW = 0.5            # bed heater draw
ENCLOSURE_TEMP_TARGET_C = 35.0       # enclosure target
ENCLOSURE_HEATER_POWER_KW = 0.2

# Capacity
BUILD_VOLUME_M3 = 0.5                # maximum build volume
MAX_FEEDSTOCK_KG = 500.0             # hopper capacity per material

# Sol
SOL_HOURS = 24.66                    # Mars sol in hours

MATERIAL_TYPES = ("regolith", "metal", "polymer")

ENERGY_MAP = {
    "regolith": ENERGY_REGOLITH_KWH_KG,
    "metal": ENERGY_METAL_KWH_KG,
    "polymer": ENERGY_POLYMER_KWH_KG,
}

SPEED_MAP = {
    "regolith": SPEED_REGOLITH_KG_HR,
    "metal": SPEED_METAL_KG_HR,
    "polymer": SPEED_POLYMER_KG_HR,
}

WASTE_MAP = {
    "regolith": WASTE_REGOLITH,
    "metal": WASTE_METAL,
    "polymer": WASTE_POLYMER,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PrintJob:
    """A fabrication job."""
    name: str
    material: str
    mass_kg: float
    priority: int = 1

    def __post_init__(self) -> None:
        if self.material not in MATERIAL_TYPES:
            self.material = "regolith"
        self.mass_kg = max(0.01, self.mass_kg)
        self.priority = max(0, min(5, self.priority))

    @property
    def energy_required_kwh(self) -> float:
        """Total energy for this job including waste."""
        waste = WASTE_MAP.get(self.material, 0.15)
        return self.mass_kg * (1.0 + waste) * ENERGY_MAP.get(self.material, 1.0)

    @property
    def feedstock_required_kg(self) -> float:
        """Feedstock consumed including waste."""
        waste = WASTE_MAP.get(self.material, 0.15)
        return self.mass_kg * (1.0 + waste)

    @property
    def print_time_hours(self) -> float:
        """Time to print this part."""
        speed = SPEED_MAP.get(self.material, 0.5)
        if speed <= 0:
            return 0.0
        return self.mass_kg / speed


@dataclass
class FabricatorState:
    """Mutable state of the Mars colony fabricator."""
    nozzle_quality: float = QUALITY_NEW
    operational: bool = True

    # Feedstock hoppers (kg)
    feedstock_regolith_kg: float = 200.0
    feedstock_metal_kg: float = 50.0
    feedstock_polymer_kg: float = 30.0

    # Counters
    total_parts_printed: int = 0
    total_mass_printed_kg: float = 0.0
    total_energy_consumed_kwh: float = 0.0
    total_feedstock_consumed_kg: float = 0.0

    # Current job (None if idle)
    current_job: Optional[PrintJob] = None
    current_job_progress_kg: float = 0.0

    def __post_init__(self) -> None:
        self.nozzle_quality = max(QUALITY_MIN, min(1.0, self.nozzle_quality))
        self.feedstock_regolith_kg = max(0.0, min(MAX_FEEDSTOCK_KG, self.feedstock_regolith_kg))
        self.feedstock_metal_kg = max(0.0, min(MAX_FEEDSTOCK_KG, self.feedstock_metal_kg))
        self.feedstock_polymer_kg = max(0.0, min(MAX_FEEDSTOCK_KG, self.feedstock_polymer_kg))
        self.total_parts_printed = max(0, self.total_parts_printed)
        self.total_mass_printed_kg = max(0.0, self.total_mass_printed_kg)
        self.total_energy_consumed_kwh = max(0.0, self.total_energy_consumed_kwh)
        self.total_feedstock_consumed_kg = max(0.0, self.total_feedstock_consumed_kg)
        self.current_job_progress_kg = max(0.0, self.current_job_progress_kg)

    @property
    def is_idle(self) -> bool:
        return self.current_job is None

    @property
    def nozzle_usable(self) -> bool:
        return self.nozzle_quality >= QUALITY_MIN

    def get_feedstock(self, material: str) -> float:
        """Get feedstock level for a material type."""
        if material == "regolith":
            return self.feedstock_regolith_kg
        elif material == "metal":
            return self.feedstock_metal_kg
        elif material == "polymer":
            return self.feedstock_polymer_kg
        return 0.0

    def consume_feedstock(self, material: str, kg: float) -> float:
        """Consume feedstock. Returns actual amount consumed."""
        available = self.get_feedstock(material)
        actual = min(available, max(0.0, kg))
        if material == "regolith":
            self.feedstock_regolith_kg -= actual
        elif material == "metal":
            self.feedstock_metal_kg -= actual
        elif material == "polymer":
            self.feedstock_polymer_kg -= actual
        return actual


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def estimate_job(job: PrintJob) -> dict:
    """Estimate energy, feedstock, and time for a print job."""
    return {
        "name": job.name,
        "material": job.material,
        "mass_kg": round(job.mass_kg, 4),
        "energy_kwh": round(job.energy_required_kwh, 4),
        "feedstock_kg": round(job.feedstock_required_kg, 4),
        "print_hours": round(job.print_time_hours, 4),
        "print_sols": round(job.print_time_hours / SOL_HOURS, 4),
    }


def start_job(state: FabricatorState, job: PrintJob) -> dict:
    """Start a new print job."""
    if not state.is_idle:
        return {"success": False, "reason": "printer_busy",
                "current_job": state.current_job.name if state.current_job else None}

    if not state.nozzle_usable:
        return {"success": False, "reason": "nozzle_worn",
                "nozzle_quality": round(state.nozzle_quality, 4)}

    if not state.operational:
        return {"success": False, "reason": "not_operational"}

    available = state.get_feedstock(job.material)
    needed = job.feedstock_required_kg
    if available < needed:
        return {"success": False, "reason": "insufficient_feedstock",
                "available_kg": round(available, 4),
                "needed_kg": round(needed, 4)}

    state.current_job = job
    state.current_job_progress_kg = 0.0

    return {
        "success": True,
        "job_name": job.name,
        "estimated": estimate_job(job),
    }


def cancel_job(state: FabricatorState) -> dict:
    """Cancel the current print job. Feedstock already consumed is lost."""
    if state.is_idle:
        return {"success": False, "reason": "no_active_job"}

    job = state.current_job
    wasted = state.current_job_progress_kg
    state.current_job = None
    state.current_job_progress_kg = 0.0

    return {
        "success": True,
        "cancelled_job": job.name if job else None,
        "feedstock_wasted_kg": round(wasted, 4),
    }


def add_feedstock(state: FabricatorState, material: str, kg: float) -> dict:
    """Add feedstock to the hopper."""
    if material not in MATERIAL_TYPES:
        return {"success": False, "reason": "invalid_material"}

    kg = max(0.0, kg)
    current = state.get_feedstock(material)
    space = MAX_FEEDSTOCK_KG - current
    actual = min(kg, space)

    if material == "regolith":
        state.feedstock_regolith_kg += actual
    elif material == "metal":
        state.feedstock_metal_kg += actual
    elif material == "polymer":
        state.feedstock_polymer_kg += actual

    return {
        "success": True,
        "material": material,
        "added_kg": round(actual, 4),
        "new_level_kg": round(state.get_feedstock(material), 4),
        "rejected_kg": round(kg - actual, 4),
    }


def calibrate(state: FabricatorState) -> dict:
    """Calibrate the printer to restore nozzle quality.

    Cannot calibrate while printing.
    """
    if not state.is_idle:
        return {"success": False, "reason": "printer_busy"}

    before = state.nozzle_quality
    gap = 1.0 - state.nozzle_quality
    state.nozzle_quality = min(1.0, state.nozzle_quality + gap * CALIBRATION_RESTORE)

    return {
        "success": True,
        "quality_before": round(before, 4),
        "quality_after": round(state.nozzle_quality, 4),
    }


# ---------------------------------------------------------------------------
# Tick -- advance one sol
# ---------------------------------------------------------------------------

@dataclass
class FabSol:
    """One sol of fabricator activity."""
    sol: int
    new_job: Optional[PrintJob] = None
    cancel_current: bool = False
    calibrate_printer: bool = False
    add_regolith_kg: float = 0.0
    add_metal_kg: float = 0.0
    add_polymer_kg: float = 0.0
    available_power_kwh: float = 50.0


def tick_fabricator(state: FabricatorState, sol: FabSol) -> dict:
    """Advance the fabricator by one sol.

    Sequence:
    1. Add feedstock (if any)
    2. Cancel job (if requested)
    3. Calibrate (if requested and idle)
    4. Start new job (if requested and idle)
    5. Print: advance current job by one sol of work
    6. Complete job if done
    """
    snapshot = {
        "sol": sol.sol,
        "nozzle_before": round(state.nozzle_quality, 4),
        "parts_before": state.total_parts_printed,
    }

    # 1. Feedstock
    feed_results = []
    if sol.add_regolith_kg > 0:
        feed_results.append(add_feedstock(state, "regolith", sol.add_regolith_kg))
    if sol.add_metal_kg > 0:
        feed_results.append(add_feedstock(state, "metal", sol.add_metal_kg))
    if sol.add_polymer_kg > 0:
        feed_results.append(add_feedstock(state, "polymer", sol.add_polymer_kg))
    snapshot["feedstock_added"] = feed_results

    # 2. Cancel
    cancel_result = None
    if sol.cancel_current:
        cancel_result = cancel_job(state)
    snapshot["cancel"] = cancel_result

    # 3. Calibrate
    cal_result = None
    if sol.calibrate_printer and state.is_idle:
        cal_result = calibrate(state)
    snapshot["calibrate"] = cal_result

    # 4. Start new job
    start_result = None
    if sol.new_job is not None and state.is_idle:
        start_result = start_job(state, sol.new_job)
    snapshot["start"] = start_result

    # 5. Print
    print_result = None
    if state.current_job is not None and state.operational:
        job = state.current_job
        speed = SPEED_MAP.get(job.material, 0.5)
        energy_rate = ENERGY_MAP.get(job.material, 1.0)
        waste = WASTE_MAP.get(job.material, 0.15)

        # How much can we print this sol?
        max_by_time = speed * SOL_HOURS
        remaining = job.mass_kg - state.current_job_progress_kg
        energy_needed = remaining * (1.0 + waste) * energy_rate
        energy_for_heating = (BED_HEATER_POWER_KW + ENCLOSURE_HEATER_POWER_KW) * SOL_HOURS
        energy_available = max(0.0, sol.available_power_kwh - energy_for_heating)

        if energy_rate > 0:
            max_by_energy = energy_available / ((1.0 + waste) * energy_rate)
        else:
            max_by_energy = remaining

        feedstock_available = state.get_feedstock(job.material)
        max_by_feedstock = feedstock_available / (1.0 + waste) if (1.0 + waste) > 0 else 0.0

        printed_kg = min(max_by_time, remaining, max_by_energy, max_by_feedstock)
        printed_kg = max(0.0, printed_kg)

        # Consume resources
        feedstock_used = printed_kg * (1.0 + waste)
        energy_used = feedstock_used * energy_rate + energy_for_heating
        actual_feedstock = state.consume_feedstock(job.material, feedstock_used)

        state.current_job_progress_kg += printed_kg
        state.total_energy_consumed_kwh += energy_used
        state.total_feedstock_consumed_kg += actual_feedstock

        # Nozzle wear
        wear = printed_kg * NOZZLE_WEAR_PER_KG
        state.nozzle_quality = max(QUALITY_MIN, state.nozzle_quality - wear)

        # Job complete?
        completed = state.current_job_progress_kg >= job.mass_kg
        if completed:
            state.total_parts_printed += 1
            state.total_mass_printed_kg += job.mass_kg
            state.current_job = None
            state.current_job_progress_kg = 0.0

        print_result = {
            "printed_kg": round(printed_kg, 4),
            "feedstock_used_kg": round(actual_feedstock, 4),
            "energy_used_kwh": round(energy_used, 4),
            "progress_kg": round(state.current_job_progress_kg, 4),
            "completed": completed,
            "job_name": job.name,
        }
    snapshot["print"] = print_result

    # Final state
    snapshot.update({
        "nozzle_after": round(state.nozzle_quality, 4),
        "nozzle_usable": state.nozzle_usable,
        "is_idle": state.is_idle,
        "total_parts": state.total_parts_printed,
        "total_mass_kg": round(state.total_mass_printed_kg, 4),
        "total_energy_kwh": round(state.total_energy_consumed_kwh, 4),
        "feedstock": {
            "regolith_kg": round(state.feedstock_regolith_kg, 4),
            "metal_kg": round(state.feedstock_metal_kg, 4),
            "polymer_kg": round(state.feedstock_polymer_kg, 4),
        },
    })

    return snapshot


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_fabricator() -> FabricatorState:
    """Create a factory-fresh fabricator."""
    return FabricatorState()
