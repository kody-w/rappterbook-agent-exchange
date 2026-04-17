"""
Ecology organ for the Mars-100 colony simulation (engine v10.0).

Models a living Mars biosphere with five interconnected subsystems:
Atmosphere, SoilState, WaterCycle, Flora, Fauna → Biosphere.

One-year lag: LAST year's biosphere drives THIS year's resource bonuses.
RNG offset: ecology_rng = seed + 11213
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

AMBIENT_PRESSURE_KPA = 0.6
PRESSURE_PER_TERRAFORM = 0.35
PRESSURE_DECAY = 0.02
O2_PRODUCTION_PER_FLORA = 0.008
CO2_ABSORPTION_PER_FLORA = 0.006
INITIAL_PERCHLORATE = 0.75
PERCHLORATE_REMEDIATION_PER_FARM = 0.012
ORGANIC_BUILDUP_PER_FARM = 0.005
ORGANIC_BUILDUP_FROM_FAUNA = 0.003
MOISTURE_FROM_WATER_CYCLE = 0.01
AQUIFER_DRAIN_RATE = 0.003
SURFACE_WATER_PRESSURE_THRESHOLD = 8.0
SURFACE_WATER_FILL_RATE = 0.015
ICE_SUBLIMATION_RATE = 0.002
WILD_PLANT_PERCHLORATE_MAX = 0.35
WILD_PLANT_PRESSURE_MIN = 4.0
CROP_BONUS_PER_FARM = 0.02
WILD_GROWTH_RATE = 0.01
FLORA_DECAY_RATE = 0.015
INSECT_FLORA_MIN = 0.15
MICROBE_ORGANIC_MIN = 0.05
FAUNA_GROWTH_RATE = 0.008
FAUNA_DECAY_RATE = 0.012
WEIGHT_ATMOSPHERE = 0.25
WEIGHT_SOIL = 0.20
WEIGHT_WATER = 0.20
WEIGHT_FLORA = 0.20
WEIGHT_FAUNA = 0.15
MAX_FOOD_BONUS = 0.03
MAX_WATER_BONUS = 0.02
MAX_AIR_BONUS = 0.025
MAX_MEDICINE_BONUS = 0.01
MAX_STRESS_REDUCTION = 0.05
MAX_PURPOSE_BOOST = 0.03


@dataclass
class Atmosphere:
    """Mars atmospheric state."""
    pressure_kpa: float = AMBIENT_PRESSURE_KPA
    o2_fraction: float = 0.001
    co2_fraction: float = 0.95

    def health(self) -> float:
        """Normalised health score [0, 1]."""
        p = min(1.0, self.pressure_kpa / 30.0)
        o = min(1.0, self.o2_fraction / 0.21)
        c = max(0.0, (self.co2_fraction - 0.5) * 0.5)
        return max(0.0, min(1.0, p * 0.5 + o * 0.3 - c * 0.2))

    def clamp(self) -> None:
        self.pressure_kpa = max(AMBIENT_PRESSURE_KPA, self.pressure_kpa)
        self.o2_fraction = max(0.0, min(0.25, self.o2_fraction))
        self.co2_fraction = max(0.01, min(1.0, self.co2_fraction))

    def to_dict(self) -> dict[str, float]:
        return {"pressure_kpa": round(self.pressure_kpa, 4),
                "o2_fraction": round(self.o2_fraction, 4),
                "co2_fraction": round(self.co2_fraction, 4),
                "health": round(self.health(), 4)}

    @classmethod
    def from_dict(cls, d: dict) -> Atmosphere:
        return cls(pressure_kpa=d.get("pressure_kpa", AMBIENT_PRESSURE_KPA),
                   o2_fraction=d.get("o2_fraction", 0.001),
                   co2_fraction=d.get("co2_fraction", 0.95))


@dataclass
class SoilState:
    """Mars soil composition."""
    perchlorate: float = INITIAL_PERCHLORATE
    organic_content: float = 0.01
    moisture: float = 0.05

    def health(self) -> float:
        p = 1.0 - self.perchlorate
        o = min(1.0, self.organic_content / 0.3)
        m = min(1.0, self.moisture / 0.3)
        return max(0.0, min(1.0, p * 0.4 + o * 0.35 + m * 0.25))

    def clamp(self) -> None:
        self.perchlorate = max(0.0, min(1.0, self.perchlorate))
        self.organic_content = max(0.0, min(1.0, self.organic_content))
        self.moisture = max(0.0, min(1.0, self.moisture))

    def to_dict(self) -> dict[str, float]:
        return {"perchlorate": round(self.perchlorate, 4),
                "organic_content": round(self.organic_content, 4),
                "moisture": round(self.moisture, 4),
                "health": round(self.health(), 4)}

    @classmethod
    def from_dict(cls, d: dict) -> SoilState:
        return cls(perchlorate=d.get("perchlorate", INITIAL_PERCHLORATE),
                   organic_content=d.get("organic_content", 0.01),
                   moisture=d.get("moisture", 0.05))


@dataclass
class WaterCycle:
    """Mars hydrological state."""
    aquifer: float = 0.6
    surface_water: float = 0.0
    ice_reserves: float = 0.8

    def health(self) -> float:
        return max(0.0, min(1.0,
            self.aquifer * 0.3 + self.surface_water * 0.4 + self.ice_reserves * 0.3))

    def clamp(self) -> None:
        self.aquifer = max(0.0, min(1.0, self.aquifer))
        self.surface_water = max(0.0, min(1.0, self.surface_water))
        self.ice_reserves = max(0.0, min(1.0, self.ice_reserves))

    def to_dict(self) -> dict[str, float]:
        return {"aquifer": round(self.aquifer, 4),
                "surface_water": round(self.surface_water, 4),
                "ice_reserves": round(self.ice_reserves, 4),
                "health": round(self.health(), 4)}

    @classmethod
    def from_dict(cls, d: dict) -> WaterCycle:
        return cls(aquifer=d.get("aquifer", 0.6),
                   surface_water=d.get("surface_water", 0.0),
                   ice_reserves=d.get("ice_reserves", 0.8))


@dataclass
class Flora:
    """Mars plant life."""
    crops: float = 0.1
    wild_plants: float = 0.0
    biomass: float = 0.05

    def health(self) -> float:
        return max(0.0, min(1.0,
            self.crops * 0.4 + self.wild_plants * 0.3 + self.biomass * 0.3))

    def clamp(self) -> None:
        self.crops = max(0.0, min(1.0, self.crops))
        self.wild_plants = max(0.0, min(1.0, self.wild_plants))
        self.biomass = max(0.0, min(1.0, self.biomass))

    def to_dict(self) -> dict[str, float]:
        return {"crops": round(self.crops, 4), "wild_plants": round(self.wild_plants, 4),
                "biomass": round(self.biomass, 4), "health": round(self.health(), 4)}

    @classmethod
    def from_dict(cls, d: dict) -> Flora:
        return cls(crops=d.get("crops", 0.1), wild_plants=d.get("wild_plants", 0.0),
                   biomass=d.get("biomass", 0.05))


@dataclass
class Fauna:
    """Mars animal / microbial life."""
    insects: float = 0.0
    microbes: float = 0.02

    def health(self) -> float:
        return max(0.0, min(1.0, self.insects * 0.5 + self.microbes * 0.5))

    def clamp(self) -> None:
        self.insects = max(0.0, min(1.0, self.insects))
        self.microbes = max(0.0, min(1.0, self.microbes))

    def to_dict(self) -> dict[str, float]:
        return {"insects": round(self.insects, 4), "microbes": round(self.microbes, 4),
                "health": round(self.health(), 4)}

    @classmethod
    def from_dict(cls, d: dict) -> Fauna:
        return cls(insects=d.get("insects", 0.0), microbes=d.get("microbes", 0.02))


@dataclass
class Biosphere:
    """Top-level ecology container for the Mars colony."""
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    soil: SoilState = field(default_factory=SoilState)
    water: WaterCycle = field(default_factory=WaterCycle)
    flora: Flora = field(default_factory=Flora)
    fauna: Fauna = field(default_factory=Fauna)

    def biosphere_index(self) -> float:
        """Weighted composite health score [0, 1]."""
        return max(0.0, min(1.0,
            self.atmosphere.health() * WEIGHT_ATMOSPHERE
            + self.soil.health() * WEIGHT_SOIL
            + self.water.health() * WEIGHT_WATER
            + self.flora.health() * WEIGHT_FLORA
            + self.fauna.health() * WEIGHT_FAUNA))

    def health(self) -> float:
        """Alias for biosphere_index — used by engine integration."""
        return self.biosphere_index()

    def clamp(self) -> None:
        self.atmosphere.clamp()
        self.soil.clamp()
        self.water.clamp()
        self.flora.clamp()
        self.fauna.clamp()

    def to_dict(self) -> dict[str, Any]:
        return {"atmosphere": self.atmosphere.to_dict(), "soil": self.soil.to_dict(),
                "water": self.water.to_dict(), "flora": self.flora.to_dict(),
                "fauna": self.fauna.to_dict(),
                "biosphere_index": round(self.biosphere_index(), 4)}

    @classmethod
    def from_dict(cls, d: dict) -> Biosphere:
        return cls(atmosphere=Atmosphere.from_dict(d.get("atmosphere", {})),
                   soil=SoilState.from_dict(d.get("soil", {})),
                   water=WaterCycle.from_dict(d.get("water", {})),
                   flora=Flora.from_dict(d.get("flora", {})),
                   fauna=Fauna.from_dict(d.get("fauna", {})))


@dataclass
class EcologyTickResult:
    """Output of one ecology tick."""
    biosphere_before: dict[str, Any] = field(default_factory=dict)
    biosphere_after: dict[str, Any] = field(default_factory=dict)
    resource_bonus: dict[str, float] = field(default_factory=dict)
    psych_pressure: dict[str, float] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"biosphere_before": self.biosphere_before,
                "biosphere_after": self.biosphere_after,
                "resource_bonus": self.resource_bonus,
                "psych_pressure": self.psych_pressure,
                "events": self.events}


# ---------------------------------------------------------------------------
# Tick: advance biosphere by one Martian year
# ---------------------------------------------------------------------------

def tick_ecology(
    bio: Biosphere,
    year: int,
    *,
    terraformers: int = 0,
    farmers: int = 0,
    researchers: int = 0,
    saboteurs: int = 0,
    infra_completed: list[str] | None = None,
    event_damage: float = 0.0,
    rng: random.Random,
) -> EcologyTickResult:
    """Advance the biosphere by one Martian year.  Mutates *bio* in place."""
    result = EcologyTickResult(biosphere_before=bio.to_dict())
    _infra = infra_completed or []

    # --- Atmosphere ---
    terraform_pressure = terraformers * PRESSURE_PER_TERRAFORM
    bio.atmosphere.pressure_kpa += terraform_pressure - PRESSURE_DECAY
    bio.atmosphere.pressure_kpa = max(AMBIENT_PRESSURE_KPA, bio.atmosphere.pressure_kpa)

    flora_o2 = bio.flora.biomass * O2_PRODUCTION_PER_FLORA
    flora_co2 = bio.flora.biomass * CO2_ABSORPTION_PER_FLORA
    bio.atmosphere.o2_fraction += flora_o2
    bio.atmosphere.co2_fraction -= flora_co2

    if event_damage > 0.3:
        damage = event_damage * 0.15 * rng.uniform(0.5, 1.0)
        if "shelter_reinforcement" in _infra:
            damage *= 0.6
        bio.atmosphere.pressure_kpa = max(AMBIENT_PRESSURE_KPA,
            bio.atmosphere.pressure_kpa - damage)
        result.events.append(f"event damage reduced pressure by {damage:.2f} kPa")

    if saboteurs > 0:
        sab_damage = saboteurs * 0.05 * rng.uniform(0.3, 1.0)
        bio.flora.crops = max(0.0, bio.flora.crops - sab_damage)
        result.events.append(f"sabotage damaged crops by {sab_damage:.3f}")

    # --- Soil ---
    bio.soil.perchlorate -= farmers * PERCHLORATE_REMEDIATION_PER_FARM
    bio.soil.organic_content += (farmers * ORGANIC_BUILDUP_PER_FARM
                                  + bio.fauna.microbes * ORGANIC_BUILDUP_FROM_FAUNA)
    if bio.water.surface_water > 0:
        bio.soil.moisture += MOISTURE_FROM_WATER_CYCLE
    else:
        bio.soil.moisture -= 0.005

    # --- Water cycle ---
    bio.water.aquifer -= AQUIFER_DRAIN_RATE
    if bio.atmosphere.pressure_kpa >= SURFACE_WATER_PRESSURE_THRESHOLD:
        fill = SURFACE_WATER_FILL_RATE * (bio.water.ice_reserves * 0.5 + 0.5)
        bio.water.surface_water += fill
        if bio.water.surface_water > 0.01:
            result.events.append("surface water forming")
    else:
        bio.water.surface_water -= 0.01
    bio.water.ice_reserves -= ICE_SUBLIMATION_RATE
    if "water_recycler" in _infra:
        bio.water.aquifer += 0.002
    if researchers > 0 and rng.random() < 0.1 * researchers:
        discovery = rng.uniform(0.02, 0.06)
        bio.water.aquifer += discovery
        result.events.append(f"aquifer discovery: +{discovery:.3f}")

    # --- Flora ---
    crop_growth = farmers * CROP_BONUS_PER_FARM
    if "greenhouse_dome" in _infra:
        crop_growth *= 1.3
    bio.flora.crops += crop_growth
    can_wild = (bio.soil.perchlorate < WILD_PLANT_PERCHLORATE_MAX
                and bio.atmosphere.pressure_kpa >= WILD_PLANT_PRESSURE_MIN)
    if can_wild:
        mf = min(1.0, bio.soil.moisture / 0.15)
        of = min(1.0, bio.soil.organic_content / 0.1)
        bio.flora.wild_plants += WILD_GROWTH_RATE * mf * of
        if bio.flora.wild_plants > 0.01:
            result.events.append("wild plants emerging")
    else:
        bio.flora.wild_plants -= FLORA_DECAY_RATE
    bio.flora.biomass = bio.flora.crops * 0.6 + bio.flora.wild_plants * 0.4

    # --- Fauna ---
    if bio.flora.biomass >= INSECT_FLORA_MIN:
        bio.fauna.insects += FAUNA_GROWTH_RATE * bio.flora.biomass
    else:
        bio.fauna.insects -= FAUNA_DECAY_RATE
    if bio.soil.organic_content >= MICROBE_ORGANIC_MIN:
        bio.fauna.microbes += FAUNA_GROWTH_RATE * bio.soil.organic_content
    else:
        bio.fauna.microbes -= FAUNA_DECAY_RATE * 0.5

    # Random ecological events
    if rng.random() < 0.05:
        _ecological_event(bio, rng, result)

    bio.clamp()
    result.biosphere_after = bio.to_dict()
    return result


def _ecological_event(bio: Biosphere, rng: random.Random,
                      result: EcologyTickResult) -> None:
    """Apply a rare ecological event."""
    etype = rng.choice(["microbe_bloom", "crop_blight", "ice_melt", "regolith_shift"])
    if etype == "microbe_bloom":
        b = rng.uniform(0.02, 0.06)
        bio.fauna.microbes += b
        bio.soil.organic_content += b * 0.5
        result.events.append(f"microbe bloom: +{b:.3f}")
    elif etype == "crop_blight":
        loss = rng.uniform(0.02, 0.08)
        bio.flora.crops -= loss
        result.events.append(f"crop blight: -{loss:.3f}")
    elif etype == "ice_melt":
        m = rng.uniform(0.02, 0.05)
        bio.water.ice_reserves -= m
        bio.water.aquifer += m * 0.7
        result.events.append(f"ice melt: aquifer +{m * 0.7:.3f}")
    elif etype == "regolith_shift":
        s = rng.uniform(0.01, 0.04)
        bio.soil.perchlorate += s
        result.events.append(f"regolith shift: perchlorate +{s:.3f}")


# ---------------------------------------------------------------------------
# Resource / modifier / upkeep / psych functions
# ---------------------------------------------------------------------------

def compute_ecology_resource_bonus(bio: Biosphere) -> dict[str, float]:
    """Compute additive resource bonuses from the biosphere (one-year lag)."""
    food = min(MAX_FOOD_BONUS, bio.flora.crops * 0.02 + bio.flora.wild_plants * 0.01)
    water = min(MAX_WATER_BONUS, bio.water.surface_water * 0.015 + bio.water.aquifer * 0.005)
    air = min(MAX_AIR_BONUS, bio.atmosphere.o2_fraction * 0.1
              * min(1.0, bio.atmosphere.pressure_kpa / 15.0))
    medicine = min(MAX_MEDICINE_BONUS, bio.flora.biomass * 0.008 + bio.fauna.microbes * 0.005)
    return {"food": round(food, 6), "water": round(water, 6),
            "air": round(air, 6), "medicine": round(medicine, 6)}


def compute_ecology_psych_pressure(bio: Biosphere) -> dict[str, float]:
    """Compute psychology effects: greening -> less stress, more purpose."""
    idx = bio.biosphere_index()
    return {"stress": round(-MAX_STRESS_REDUCTION * idx, 6),
            "purpose": round(MAX_PURPOSE_BOOST * idx, 6)}


def compute_ecology_modifiers(bio: Biosphere) -> dict[str, float]:
    """Multiplier-style resource modifiers (compatible with infra pattern).

    A fully green biosphere reduces food spoilage by 25%, air maintenance
    by 20%, water maintenance by 15%.  Scales linearly with biosphere index.
    """
    idx = bio.biosphere_index()
    return {"food_spoilage_mult": round(max(0.75, 1.0 - 0.25 * idx), 4),
            "air_maintenance_mult": round(max(0.80, 1.0 - 0.20 * idx), 4),
            "water_maintenance_mult": round(max(0.85, 1.0 - 0.15 * idx), 4)}


def compute_ecology_upkeep(bio: Biosphere) -> dict[str, float]:
    """Operating costs for maintaining the biosphere (greenhouses etc.)."""
    return {"power": round(bio.flora.crops * 0.003 + bio.fauna.insects * 0.001, 6),
            "water": round(bio.flora.biomass * 0.002, 6)}
