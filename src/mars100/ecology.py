"""
Mars-100 ecology organ — biosphere simulation (engine v10.0).

Models the evolving Martian biosphere across 5 interconnected subsystems:
atmosphere, soil, water cycle, flora, and fauna.  All values normalized
[0, 1].  Terraforming and farming actions drive biosphere growth; zero
effort means net decay.

Design decisions:
  - One-year lag: LAST year's biosphere drives THIS year's resource bonuses.
  - Indoor (greenhouse) and outdoor (wild) flora tracked separately.
  - Fauna gated on outdoor flora + atmosphere thresholds.
  - Zero colonist effort → net biosphere decay (no free lunch on Mars).
  - Nature exposure (biosphere health) feeds into psychology as stress relief.
  - ecology_rng = Random(seed + 11213) — isolated RNG stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Atmosphere
PRESSURE_TERRAFORM_GAIN: float = 0.003
PRESSURE_NATURAL_LOSS: float = 0.001
O2_TERRAFORM_GAIN: float = 0.002
O2_FLORA_CONTRIBUTION: float = 0.001
CO2_TERRAFORM_REDUCTION: float = 0.002
CO2_NATURAL_RISE: float = 0.0005
TEMP_PRESSURE_COUPLING: float = 0.1

# Soil
PERCHLORATE_INITIAL: float = 0.8
PERCHLORATE_FARM_REDUCTION: float = 0.008
PERCHLORATE_NATURAL_REDUCTION: float = 0.0
ORGANIC_FARM_GAIN: float = 0.005
ORGANIC_FLORA_GAIN: float = 0.002
ORGANIC_DECAY: float = 0.001
NITROGEN_FARM_GAIN: float = 0.003
NITROGEN_DECAY: float = 0.0005

# Water
AQUIFER_INITIAL: float = 0.3
AQUIFER_TERRAFORM_GAIN: float = 0.002
AQUIFER_NATURAL_LOSS: float = 0.0005
SURFACE_WATER_PRESSURE_THRESHOLD: float = 0.15
SURFACE_WATER_GAIN_RATE: float = 0.005
SURFACE_WATER_LOSS_RATE: float = 0.002
ICE_CAP_TERRAFORM_MELT: float = 0.001
ICE_CAP_INITIAL: float = 0.6

# Flora
CROP_FARM_GAIN: float = 0.01
CROP_DECAY: float = 0.003
WILD_PLANT_PERCHLORATE_THRESHOLD: float = 0.3
WILD_PLANT_GROWTH_RATE: float = 0.005
WILD_PLANT_DECAY_RATE: float = 0.003
WILD_PLANT_PRESSURE_THRESHOLD: float = 0.1
BIODIVERSITY_GROWTH_RATE: float = 0.002
BIODIVERSITY_DECAY_RATE: float = 0.001

# Fauna
INSECT_FLORA_THRESHOLD: float = 0.15
INSECT_GROWTH_RATE: float = 0.004
INSECT_DECAY_RATE: float = 0.005
MICROBE_GROWTH_RATE: float = 0.006
MICROBE_DECAY_RATE: float = 0.002
MICROBE_SOIL_THRESHOLD: float = 0.1

# Biosphere composite weights
BIOSPHERE_WEIGHTS: dict[str, float] = {
    "atmosphere": 0.25,
    "soil": 0.20,
    "water": 0.20,
    "flora": 0.20,
    "fauna": 0.15,
}

# Resource bonus caps (per year, from ecology)
MAX_FOOD_BONUS: float = 0.04
MAX_WATER_BONUS: float = 0.03
MAX_AIR_BONUS: float = 0.02
MAX_MEDICINE_BONUS: float = 0.02

# Psychology integration
MAX_STRESS_REDUCTION: float = 0.05

# Tipping points: below these thresholds, subsystems actively decline faster
ATMOSPHERE_TIPPING_POINT: float = 0.05
SOIL_TIPPING_POINT: float = 0.1
WATER_TIPPING_POINT: float = 0.05
FLORA_TIPPING_POINT: float = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Atmosphere:
    """Atmospheric state, all normalized [0, 1]."""
    pressure: float = 0.02
    o2_fraction: float = 0.01
    co2_fraction: float = 0.95
    temperature: float = 0.1

    def health(self) -> float:
        """Composite atmosphere health score [0, 1]."""
        return (self.pressure * 0.4
                + self.o2_fraction * 0.3
                + (1.0 - self.co2_fraction) * 0.2
                + self.temperature * 0.1)

    def to_dict(self) -> dict[str, float]:
        return {
            "pressure": round(self.pressure, 6),
            "o2_fraction": round(self.o2_fraction, 6),
            "co2_fraction": round(self.co2_fraction, 6),
            "temperature": round(self.temperature, 6),
            "health": round(self.health(), 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Atmosphere:
        return cls(
            pressure=d.get("pressure", 0.02),
            o2_fraction=d.get("o2_fraction", 0.01),
            co2_fraction=d.get("co2_fraction", 0.95),
            temperature=d.get("temperature", 0.1),
        )


@dataclass
class Soil:
    """Soil state, all normalized [0, 1]."""
    perchlorate: float = PERCHLORATE_INITIAL
    organic_content: float = 0.01
    nitrogen: float = 0.02

    def health(self) -> float:
        """Composite soil health score [0, 1]."""
        perchlorate_score = 1.0 - self.perchlorate
        return (perchlorate_score * 0.4
                + self.organic_content * 0.35
                + self.nitrogen * 0.25)

    def to_dict(self) -> dict[str, float]:
        return {
            "perchlorate": round(self.perchlorate, 6),
            "organic_content": round(self.organic_content, 6),
            "nitrogen": round(self.nitrogen, 6),
            "health": round(self.health(), 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Soil:
        return cls(
            perchlorate=d.get("perchlorate", PERCHLORATE_INITIAL),
            organic_content=d.get("organic_content", 0.01),
            nitrogen=d.get("nitrogen", 0.02),
        )


@dataclass
class WaterCycle:
    """Water cycle state, all normalized [0, 1]."""
    aquifer: float = AQUIFER_INITIAL
    surface_water: float = 0.0
    ice_cap: float = ICE_CAP_INITIAL

    def health(self) -> float:
        """Composite water health score [0, 1]."""
        return (self.aquifer * 0.4
                + self.surface_water * 0.35
                + self.ice_cap * 0.25)

    def to_dict(self) -> dict[str, float]:
        return {
            "aquifer": round(self.aquifer, 6),
            "surface_water": round(self.surface_water, 6),
            "ice_cap": round(self.ice_cap, 6),
            "health": round(self.health(), 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WaterCycle:
        return cls(
            aquifer=d.get("aquifer", AQUIFER_INITIAL),
            surface_water=d.get("surface_water", 0.0),
            ice_cap=d.get("ice_cap", ICE_CAP_INITIAL),
        )


@dataclass
class Flora:
    """Flora state — separate indoor (crop) and outdoor (wild) tracking."""
    crop_yield: float = 0.1
    wild_plants: float = 0.0
    biodiversity: float = 0.0

    def health(self) -> float:
        """Composite flora health score [0, 1]."""
        return (self.crop_yield * 0.3
                + self.wild_plants * 0.4
                + self.biodiversity * 0.3)

    def outdoor_biomass(self) -> float:
        """Outdoor flora only (gates fauna)."""
        return (self.wild_plants * 0.6 + self.biodiversity * 0.4)

    def to_dict(self) -> dict[str, float]:
        return {
            "crop_yield": round(self.crop_yield, 6),
            "wild_plants": round(self.wild_plants, 6),
            "biodiversity": round(self.biodiversity, 6),
            "health": round(self.health(), 6),
            "outdoor_biomass": round(self.outdoor_biomass(), 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Flora:
        return cls(
            crop_yield=d.get("crop_yield", 0.1),
            wild_plants=d.get("wild_plants", 0.0),
            biodiversity=d.get("biodiversity", 0.0),
        )


@dataclass
class Fauna:
    """Fauna state — insects and microbes."""
    insects: float = 0.0
    microbes: float = 0.02

    def health(self) -> float:
        """Composite fauna health score [0, 1]."""
        return self.insects * 0.5 + self.microbes * 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "insects": round(self.insects, 6),
            "microbes": round(self.microbes, 6),
            "health": round(self.health(), 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Fauna:
        return cls(
            insects=d.get("insects", 0.0),
            microbes=d.get("microbes", 0.02),
        )


@dataclass
class Biosphere:
    """Complete Martian biosphere state."""
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    soil: Soil = field(default_factory=Soil)
    water: WaterCycle = field(default_factory=WaterCycle)
    flora: Flora = field(default_factory=Flora)
    fauna: Fauna = field(default_factory=Fauna)

    def index(self) -> float:
        """Weighted composite biosphere health index [0, 1]."""
        scores = {
            "atmosphere": self.atmosphere.health(),
            "soil": self.soil.health(),
            "water": self.water.health(),
            "flora": self.flora.health(),
            "fauna": self.fauna.health(),
        }
        return sum(scores[k] * BIOSPHERE_WEIGHTS[k] for k in BIOSPHERE_WEIGHTS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atmosphere": self.atmosphere.to_dict(),
            "soil": self.soil.to_dict(),
            "water": self.water.to_dict(),
            "flora": self.flora.to_dict(),
            "fauna": self.fauna.to_dict(),
            "biosphere_index": round(self.index(), 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Biosphere:
        return cls(
            atmosphere=Atmosphere.from_dict(d.get("atmosphere", {})),
            soil=Soil.from_dict(d.get("soil", {})),
            water=WaterCycle.from_dict(d.get("water", {})),
            flora=Flora.from_dict(d.get("flora", {})),
            fauna=Fauna.from_dict(d.get("fauna", {})),
        )


@dataclass
class EcologyTickResult:
    """Result of one year's ecology tick."""
    biosphere_before: dict[str, Any] = field(default_factory=dict)
    biosphere_after: dict[str, Any] = field(default_factory=dict)
    resource_bonuses: dict[str, float] = field(default_factory=dict)
    stress_reduction: float = 0.0
    tipping_points_hit: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "biosphere_before": self.biosphere_before,
            "biosphere_after": self.biosphere_after,
            "resource_bonuses": self.resource_bonuses,
            "stress_reduction": round(self.stress_reduction, 6),
            "tipping_points_hit": self.tipping_points_hit,
        }


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_resource_bonuses(biosphere: Biosphere) -> dict[str, float]:
    """Compute resource bonuses from PREVIOUS year's biosphere state.

    Returns additive deltas for colony resources.
    """
    idx = biosphere.index()
    flora_h = biosphere.flora.health()
    water_h = biosphere.water.health()
    atmo_h = biosphere.atmosphere.health()

    return {
        "food": min(MAX_FOOD_BONUS, flora_h * 0.06),
        "water": min(MAX_WATER_BONUS, water_h * 0.05),
        "air": min(MAX_AIR_BONUS, atmo_h * 0.04),
        "medicine": min(MAX_MEDICINE_BONUS, idx * 0.03),
        "power": 0.0,
    }


def compute_nature_stress_reduction(biosphere: Biosphere) -> float:
    """Compute stress reduction from nature exposure.

    Returns a non-negative value to SUBTRACT from colonist stress.
    """
    idx = biosphere.index()
    outdoor = biosphere.flora.outdoor_biomass()
    raw = idx * 0.03 + outdoor * 0.04
    return min(MAX_STRESS_REDUCTION, max(0.0, raw))


def tick_atmosphere(
    atmo: Atmosphere,
    terraform_count: int,
    flora_outdoor: float,
    event_severity: float,
) -> list[str]:
    """Advance atmosphere one year. Returns list of tipping points hit."""
    tipping: list[str] = []

    # Terraforming effort
    effort = terraform_count * PRESSURE_TERRAFORM_GAIN
    atmo.pressure = _clamp(atmo.pressure + effort - PRESSURE_NATURAL_LOSS)

    # O₂: from terraforming + flora photosynthesis
    o2_gain = terraform_count * O2_TERRAFORM_GAIN + flora_outdoor * O2_FLORA_CONTRIBUTION
    atmo.o2_fraction = _clamp(atmo.o2_fraction + o2_gain)

    # CO₂: reduced by terraforming, rises naturally
    co2_change = -terraform_count * CO2_TERRAFORM_REDUCTION + CO2_NATURAL_RISE
    atmo.co2_fraction = _clamp(atmo.co2_fraction + co2_change)

    # Temperature: coupled to pressure (greenhouse effect)
    target_temp = atmo.pressure * TEMP_PRESSURE_COUPLING + atmo.co2_fraction * 0.05
    atmo.temperature = _clamp(
        atmo.temperature + (target_temp - atmo.temperature) * 0.1)

    # Event damage to atmosphere
    if event_severity > 0.6:
        atmo.pressure = _clamp(atmo.pressure - event_severity * 0.005)

    # Tipping point
    if atmo.health() < ATMOSPHERE_TIPPING_POINT:
        atmo.pressure = _clamp(atmo.pressure - 0.002)
        tipping.append("atmosphere_collapse")

    return tipping


def tick_soil(
    soil: Soil,
    farm_count: int,
    flora_wild: float,
) -> list[str]:
    """Advance soil one year. Returns list of tipping points hit."""
    tipping: list[str] = []

    # Farming remediates perchlorate
    remediation = farm_count * PERCHLORATE_FARM_REDUCTION
    soil.perchlorate = _clamp(soil.perchlorate - remediation)

    # Organic content: farming + wild plant decomposition
    organic_gain = (farm_count * ORGANIC_FARM_GAIN
                    + flora_wild * ORGANIC_FLORA_GAIN
                    - ORGANIC_DECAY)
    soil.organic_content = _clamp(soil.organic_content + organic_gain)

    # Nitrogen fixation
    nitrogen_gain = farm_count * NITROGEN_FARM_GAIN - NITROGEN_DECAY
    soil.nitrogen = _clamp(soil.nitrogen + nitrogen_gain)

    # Tipping point
    if soil.health() < SOIL_TIPPING_POINT:
        soil.organic_content = _clamp(soil.organic_content - 0.002)
        tipping.append("soil_degradation")

    return tipping


def tick_water(
    water: WaterCycle,
    terraform_count: int,
    pressure: float,
) -> list[str]:
    """Advance water cycle one year. Returns list of tipping points hit."""
    tipping: list[str] = []

    # Aquifer: grows with terraforming
    aquifer_change = (terraform_count * AQUIFER_TERRAFORM_GAIN
                      - AQUIFER_NATURAL_LOSS)
    water.aquifer = _clamp(water.aquifer + aquifer_change)

    # Surface water: only appears above pressure threshold
    if pressure >= SURFACE_WATER_PRESSURE_THRESHOLD:
        surface_gain = (water.aquifer * SURFACE_WATER_GAIN_RATE
                        - SURFACE_WATER_LOSS_RATE)
        water.surface_water = _clamp(water.surface_water + surface_gain)
    else:
        water.surface_water = _clamp(water.surface_water - SURFACE_WATER_LOSS_RATE)

    # Ice cap: melted by terraforming (slow)
    if terraform_count > 0:
        melt = terraform_count * ICE_CAP_TERRAFORM_MELT
        water.ice_cap = _clamp(water.ice_cap - melt)
        water.aquifer = _clamp(water.aquifer + melt * 0.5)

    # Tipping point
    if water.health() < WATER_TIPPING_POINT:
        water.surface_water = _clamp(water.surface_water - 0.002)
        tipping.append("water_crisis")

    return tipping


def tick_flora(
    flora: Flora,
    farm_count: int,
    perchlorate: float,
    pressure: float,
    surface_water: float,
) -> list[str]:
    """Advance flora one year. Returns list of tipping points hit."""
    tipping: list[str] = []

    # Indoor crops: driven by farming effort
    crop_change = farm_count * CROP_FARM_GAIN - CROP_DECAY
    flora.crop_yield = _clamp(flora.crop_yield + crop_change)

    # Outdoor wild plants: gated on perchlorate AND pressure thresholds
    can_grow_outdoor = (perchlorate < WILD_PLANT_PERCHLORATE_THRESHOLD
                        and pressure >= WILD_PLANT_PRESSURE_THRESHOLD)
    if can_grow_outdoor:
        wild_gain = (WILD_PLANT_GROWTH_RATE
                     * (1.0 - perchlorate)
                     * min(1.0, surface_water * 5.0))
        flora.wild_plants = _clamp(flora.wild_plants + wild_gain)
    else:
        flora.wild_plants = _clamp(flora.wild_plants - WILD_PLANT_DECAY_RATE)

    # Biodiversity: needs both crop and wild base
    if flora.wild_plants > 0.05 and flora.crop_yield > 0.05:
        bio_gain = BIODIVERSITY_GROWTH_RATE * (flora.wild_plants + flora.crop_yield)
        flora.biodiversity = _clamp(flora.biodiversity + bio_gain)
    else:
        flora.biodiversity = _clamp(flora.biodiversity - BIODIVERSITY_DECAY_RATE)

    # Tipping point
    if flora.health() < FLORA_TIPPING_POINT and flora.health() > 0:
        flora.wild_plants = _clamp(flora.wild_plants - 0.002)
        tipping.append("flora_collapse")

    return tipping


def tick_fauna(
    fauna: Fauna,
    outdoor_biomass: float,
    atmosphere_health: float,
    soil_organic: float,
) -> list[str]:
    """Advance fauna one year. Returns list of tipping points hit."""
    tipping: list[str] = []

    # Insects: gated on outdoor flora + atmosphere
    if (outdoor_biomass >= INSECT_FLORA_THRESHOLD
            and atmosphere_health > ATMOSPHERE_TIPPING_POINT):
        insect_gain = INSECT_GROWTH_RATE * outdoor_biomass
        fauna.insects = _clamp(fauna.insects + insect_gain)
    else:
        fauna.insects = _clamp(fauna.insects - INSECT_DECAY_RATE)

    # Microbes: gated on soil organic content
    if soil_organic >= MICROBE_SOIL_THRESHOLD:
        microbe_gain = MICROBE_GROWTH_RATE * soil_organic
        fauna.microbes = _clamp(fauna.microbes + microbe_gain)
    else:
        fauna.microbes = _clamp(fauna.microbes - MICROBE_DECAY_RATE)

    return tipping


def tick_ecology(
    biosphere: Biosphere,
    terraform_count: int,
    farm_count: int,
    event_severity: float,
) -> EcologyTickResult:
    """Run one year of ecological evolution. Mutates biosphere in place.

    Args:
        biosphere: Current biosphere state (mutated).
        terraform_count: Number of colonists terraforming this year.
        farm_count: Number of colonists farming this year.
        event_severity: Max event severity this year (for damage).

    Returns:
        EcologyTickResult with before/after snapshots and bonuses.
    """
    before = biosphere.to_dict()

    # Tick each subsystem in dependency order
    tipping: list[str] = []

    tipping.extend(tick_atmosphere(
        biosphere.atmosphere, terraform_count,
        biosphere.flora.outdoor_biomass(), event_severity))

    tipping.extend(tick_soil(
        biosphere.soil, farm_count, biosphere.flora.wild_plants))

    tipping.extend(tick_water(
        biosphere.water, terraform_count, biosphere.atmosphere.pressure))

    tipping.extend(tick_flora(
        biosphere.flora, farm_count,
        biosphere.soil.perchlorate, biosphere.atmosphere.pressure,
        biosphere.water.surface_water))

    tipping.extend(tick_fauna(
        biosphere.fauna, biosphere.flora.outdoor_biomass(),
        biosphere.atmosphere.health(), biosphere.soil.organic_content))

    after = biosphere.to_dict()

    # Resource bonuses computed from BEFORE state (one-year lag)
    bonuses = compute_resource_bonuses(Biosphere.from_dict(before))
    stress_reduction = compute_nature_stress_reduction(Biosphere.from_dict(before))

    return EcologyTickResult(
        biosphere_before=before,
        biosphere_after=after,
        resource_bonuses=bonuses,
        stress_reduction=stress_reduction,
        tipping_points_hit=tipping,
    )
