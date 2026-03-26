"""
rad_shield.py — Mars radiation shielding model.

Models radiation attenuation through habitat shielding materials
and tracks crew dose accumulation with biological repair.

Physical references:
  - GCR on Mars surface: 0.67 mSv/sol (Curiosity RAD instrument)
  - Regolith areal density: ~1.5 g/cm³, HVL for GCR ≈ 20 g/cm²
  - Water HVL for GCR: ~18 g/cm² (hydrogen-rich, excellent shield)
  - Polyethylene HVL: ~22 g/cm² (common spacecraft shield)
  - SEP events: 10–1000 mSv over hours (variable)
  - NASA career limit: 600 mSv (modern NSCR-2020 model)
  - Biological repair: ~1% dose equivalent cleared per sol (DNA repair)

One tick = one sol. Dose in mSv.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

GCR_SURFACE_MSV_SOL = 0.67       # Curiosity RAD baseline
SEP_MILD_MSV = 50.0              # mild solar particle event (total)
SEP_SEVERE_MSV = 500.0           # severe SPE (Carrington-class)
SEP_DURATION_SOLS = 2.0          # typical event duration

# Half-value layers (areal density in g/cm² to halve GCR dose)
HVL_REGOLITH_G_CM2 = 20.0       # Mars regolith
HVL_WATER_G_CM2 = 18.0          # liquid water
HVL_POLY_G_CM2 = 22.0           # polyethylene / HDPE

# Material bulk densities (g/cm³)
DENSITY_REGOLITH = 1.5           # Mars regolith
DENSITY_WATER = 1.0              # liquid water
DENSITY_POLY = 0.95              # polyethylene

# Biological repair
REPAIR_RATE_PER_SOL = 0.01       # fraction of cumulative dose repaired per sol
MAX_REPAIR_MSV_SOL = 0.5         # cap on daily repair (mSv)

# NASA career limit
NASA_CAREER_LIMIT_MSV = 600.0    # NSCR-2020 reference


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShieldLayer:
    """One layer of radiation shielding material.

    Attributes:
        material: one of 'regolith', 'water', 'polyethylene'
        thickness_cm: physical thickness in centimetres
    """
    material: str
    thickness_cm: float

    def __post_init__(self) -> None:
        if self.thickness_cm < 0:
            raise ValueError("thickness_cm must be non-negative")
        valid = ("regolith", "water", "polyethylene")
        if self.material not in valid:
            raise ValueError(f"material must be one of {valid}")


@dataclass
class HabitatShield:
    """Complete shielding stack for a habitat module.

    Layers are ordered outside-in. Radiation passes through each
    layer sequentially, attenuating at each step.
    """
    layers: list[ShieldLayer] = field(default_factory=list)

    def add_layer(self, material: str, thickness_cm: float) -> None:
        """Append a shielding layer."""
        self.layers.append(ShieldLayer(material, thickness_cm))


@dataclass
class CrewDose:
    """Cumulative radiation dose tracker for one crew member.

    Attributes:
        cumulative_msv: total effective dose (after biological repair)
        peak_daily_msv: highest single-sol dose received
        sols_exposed: number of sols tracked
    """
    cumulative_msv: float = 0.0
    peak_daily_msv: float = 0.0
    sols_exposed: int = 0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _hvl_for_material(material: str) -> float:
    """Return the half-value layer (g/cm²) for a shielding material."""
    table = {
        "regolith": HVL_REGOLITH_G_CM2,
        "water": HVL_WATER_G_CM2,
        "polyethylene": HVL_POLY_G_CM2,
    }
    return table[material]


def _density_for_material(material: str) -> float:
    """Return bulk density (g/cm³) for a shielding material."""
    table = {
        "regolith": DENSITY_REGOLITH,
        "water": DENSITY_WATER,
        "polyethylene": DENSITY_POLY,
    }
    return table[material]


def areal_density(material: str, thickness_cm: float) -> float:
    """Convert thickness (cm) to areal density (g/cm²).

    Areal density = thickness × bulk density.
    """
    return thickness_cm * _density_for_material(material)


def attenuation_factor(shield: HabitatShield) -> float:
    """Compute total GCR attenuation factor (0–1) through a shielding stack.

    Uses exponential attenuation: I/I₀ = (1/2)^(x/HVL) per layer.
    Layers compound multiplicatively.

    Returns:
        Factor in (0, 1] where 1.0 = no shielding, approaching 0 = perfect.
    """
    factor = 1.0
    for layer in shield.layers:
        ad = areal_density(layer.material, layer.thickness_cm)
        hvl = _hvl_for_material(layer.material)
        factor *= math.pow(0.5, ad / hvl)
    return factor


def shielded_dose(
    ambient_msv: float,
    shield: HabitatShield,
    fraction_indoors: float = 0.85,
) -> float:
    """Effective dose (mSv) after shielding, accounting for time indoors.

    Crew spend fraction_indoors of their sol inside the shielded habitat
    and the remainder outside (EVA, maintenance) with no shielding.

    Returns:
        Effective dose in mSv for one sol.
    """
    if not 0.0 <= fraction_indoors <= 1.0:
        raise ValueError("fraction_indoors must be in [0, 1]")
    att = attenuation_factor(shield)
    indoor_dose = ambient_msv * att * fraction_indoors
    outdoor_dose = ambient_msv * (1.0 - fraction_indoors)
    return indoor_dose + outdoor_dose


def biological_repair(dose: CrewDose) -> float:
    """Apply one sol of biological DNA repair. Returns dose repaired (mSv).

    The body repairs a fraction of cumulative dose each sol, capped
    at MAX_REPAIR_MSV_SOL to prevent unrealistic rapid clearance.
    """
    repair = min(
        dose.cumulative_msv * REPAIR_RATE_PER_SOL,
        MAX_REPAIR_MSV_SOL,
    )
    repair = min(repair, dose.cumulative_msv)  # can't repair below zero
    dose.cumulative_msv -= repair
    return repair


def career_fraction(dose: CrewDose) -> float:
    """Fraction of NASA career radiation limit consumed (0–1+)."""
    return dose.cumulative_msv / NASA_CAREER_LIMIT_MSV


def tick_radiation(
    dose: CrewDose,
    ambient_msv: float,
    shield: HabitatShield,
    fraction_indoors: float = 0.85,
    sep_event: bool = False,
    sep_severity: float = 0.0,
) -> float:
    """Advance radiation tracking by one sol.

    Args:
        dose: crew member dose tracker (mutated in place)
        ambient_msv: GCR dose without shielding (mSv/sol)
        shield: habitat shielding configuration
        fraction_indoors: time fraction spent inside habitat
        sep_event: True if a solar particle event is active
        sep_severity: 0.0 (mild) to 1.0 (Carrington-class)

    Returns:
        Net dose added this sol (mSv), after repair.
    """
    daily = shielded_dose(ambient_msv, shield, fraction_indoors)

    if sep_event:
        sev = max(0.0, min(1.0, sep_severity))
        sep_total = SEP_MILD_MSV + sev * (SEP_SEVERE_MSV - SEP_MILD_MSV)
        sep_daily = sep_total / SEP_DURATION_SOLS
        # During SEP, crew shelters fully indoors
        sep_shielded = sep_daily * attenuation_factor(shield)
        daily += sep_shielded

    dose.cumulative_msv += daily
    dose.peak_daily_msv = max(dose.peak_daily_msv, daily)
    dose.sols_exposed += 1

    repaired = biological_repair(dose)
    return daily - repaired
