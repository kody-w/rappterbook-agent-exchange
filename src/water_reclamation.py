"""water_reclamation.py — Mars Colony Water Recovery System.

Models closed-loop water reclamation for a Mars habitat, based on the
ISS Water Recovery System (WRS) architecture scaled for colony operations.

Subsystems modelled
-------------------
* **Condensate collector** — extracts water vapor from habitat atmosphere.
  Humidity from crew respiration/perspiration condenses on cold plates.
  Rate depends on crew count, ambient temperature, and collector area.
* **Urine processor** — vapor-compression distillation of urine to water.
  ISS UPA recovers ~85% of water from urine.  Mars version targets 93%.
  Energy-intensive: ~0.4 kWh per liter processed.
* **Brine processor** — extracts remaining water from concentrated brine.
  The last 7-15% of water locked in brine.  High energy, high value.
  Based on NASA Brine Processor Assembly (BPA) design.
* **Greywater recycler** — treats shower, laundry, and food-prep water.
  Filtration + catalytic oxidation.  90-95% recovery.
* **Quality monitor** — tracks contaminant levels (TOC, conductivity,
  iodine/silver biocide).  Water below quality threshold is re-processed.

Physical references (NASA ECLSS / Mars DRA 5.0):
  - Crew water production: ~2.5 L/person/sol (urine ~1.5L + sweat/resp ~1.0L)
  - Greywater production: ~3.0 L/person/sol (hygiene + food prep)
  - ISS WRS power: ~1.5 kW continuous for 6 crew
  - ISS water recovery: 93% overall (urine 85% + condensate 100%)
  - Target Mars recovery: ≥96% (brine processor closes the gap)
  - Potable water standard: TOC < 3.0 mg/L, conductivity < 2.0 µS/cm

One tick = one sol.  Volume in liters, energy in kWh, mass in kg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Crew wastewater production (L/person/sol)
URINE_L_PER_PERSON_SOL = 1.5
SWEAT_RESP_L_PER_PERSON_SOL = 1.0
GREYWATER_L_PER_PERSON_SOL = 3.0

# Subsystem recovery rates (fraction)
CONDENSATE_RECOVERY = 0.98           # near-perfect on cold plates
URINE_DISTILL_RECOVERY = 0.85       # ISS baseline VCD
URINE_DISTILL_MARS_TARGET = 0.93    # improved Mars version
BRINE_RECOVERY = 0.85               # of remaining brine water content
GREYWATER_RECOVERY = 0.92           # filtration + catalytic oxidation

# Energy costs (kWh per liter processed)
CONDENSATE_ENERGY_KWH_L = 0.05      # just fan + cold plate
URINE_DISTILL_ENERGY_KWH_L = 0.40   # vapor compression distillation
BRINE_ENERGY_KWH_L = 0.80           # high-energy last-drop extraction
GREYWATER_ENERGY_KWH_L = 0.15       # filtration + UV + catalytic oxidation

# Water quality thresholds
TOC_LIMIT_MG_L = 3.0                # total organic carbon
CONDUCTIVITY_LIMIT_US_CM = 2.0      # microsiemens/cm
BIOCIDE_MIN_MG_L = 0.2              # minimum silver-ion biocide

# Degradation
FILTER_LIFE_LITERS = 50_000.0       # total throughput before replacement
MEMBRANE_DEGRADATION_PER_SOL = 0.0003  # fractional efficiency loss
MIN_EFFICIENCY_FACTOR = 0.60        # minimum before system fails
MAINTENANCE_RESTORE = 0.90          # how much maintenance recovers

# Temperature effects on condensation
CONDENSATION_OPTIMAL_C = 18.0       # habitat target temperature
CONDENSATION_COEFF = 0.02           # efficiency drop per °C above optimal


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CondensateCollector:
    """Atmospheric condensate extraction system."""

    area_m2: float = 10.0            # cold-plate surface area
    efficiency: float = CONDENSATE_RECOVERY
    total_collected_l: float = 0.0
    health: float = 1.0              # 0-1, degrades over time

    def collect(self, crew: int, hab_temp_c: float = 22.0) -> tuple[float, float]:
        """Collect condensate from crew respiration/perspiration.

        Returns (liters_collected, energy_used_kwh).
        """
        raw_vapor = crew * SWEAT_RESP_L_PER_PERSON_SOL

        # Temperature penalty: warmer habitat = less condensation
        temp_penalty = max(0.0, (hab_temp_c - CONDENSATION_OPTIMAL_C) * CONDENSATION_COEFF)
        effective_eff = max(0.3, self.efficiency * self.health * (1.0 - temp_penalty))

        collected = raw_vapor * effective_eff
        energy = collected * CONDENSATE_ENERGY_KWH_L

        self.total_collected_l += collected
        self._degrade(collected)
        return collected, energy

    def _degrade(self, liters: float) -> None:
        """Cold plates accumulate biofilm, reducing efficiency."""
        degradation = liters / FILTER_LIFE_LITERS
        self.health = max(MIN_EFFICIENCY_FACTOR, self.health - degradation)


@dataclass
class UrineProcessor:
    """Vapor-compression distillation urine processor."""

    recovery_rate: float = URINE_DISTILL_RECOVERY
    health: float = 1.0
    total_processed_l: float = 0.0
    brine_accumulated_l: float = 0.0

    def process(self, crew: int) -> tuple[float, float, float]:
        """Process crew urine for one sol.

        Returns (clean_water_l, brine_l, energy_used_kwh).
        """
        raw_urine = crew * URINE_L_PER_PERSON_SOL
        effective_rate = self.recovery_rate * self.health

        # Mars target: improved distillation at higher efficiency
        target = min(URINE_DISTILL_MARS_TARGET, effective_rate + 0.08)
        effective_rate = min(target, effective_rate)

        clean = raw_urine * effective_rate
        brine = raw_urine - clean
        energy = raw_urine * URINE_DISTILL_ENERGY_KWH_L

        self.total_processed_l += raw_urine
        self.brine_accumulated_l += brine
        self._degrade(raw_urine)
        return clean, brine, energy

    def _degrade(self, liters: float) -> None:
        """Mineral deposits reduce distillation efficiency."""
        degradation = liters / FILTER_LIFE_LITERS * 1.5  # harsher than condensate
        self.health = max(MIN_EFFICIENCY_FACTOR, self.health - degradation)


@dataclass
class BrineProcessor:
    """Extracts remaining water from concentrated urine brine."""

    recovery_rate: float = BRINE_RECOVERY
    health: float = 1.0
    total_recovered_l: float = 0.0

    def process(self, brine_l: float) -> tuple[float, float, float]:
        """Extract water from brine.

        Returns (clean_water_l, solid_waste_kg, energy_used_kwh).
        """
        effective_rate = self.recovery_rate * self.health
        clean = brine_l * effective_rate
        # Remaining brine becomes solid mineral waste (~1.2 kg/L brine density)
        solid_waste = (brine_l - clean) * 1.2
        energy = brine_l * BRINE_ENERGY_KWH_L

        self.total_recovered_l += clean
        self._degrade(brine_l)
        return clean, solid_waste, energy

    def _degrade(self, liters: float) -> None:
        """Concentrated minerals accelerate membrane fouling."""
        degradation = liters / FILTER_LIFE_LITERS * 2.0
        self.health = max(MIN_EFFICIENCY_FACTOR, self.health - degradation)


@dataclass
class GreywaterRecycler:
    """Treats hygiene and food-preparation wastewater."""

    recovery_rate: float = GREYWATER_RECOVERY
    health: float = 1.0
    total_processed_l: float = 0.0

    def process(self, crew: int) -> tuple[float, float, float]:
        """Process greywater for one sol.

        Returns (clean_water_l, waste_l, energy_used_kwh).
        """
        raw_grey = crew * GREYWATER_L_PER_PERSON_SOL
        effective_rate = self.recovery_rate * self.health

        clean = raw_grey * effective_rate
        waste = raw_grey - clean
        energy = raw_grey * GREYWATER_ENERGY_KWH_L

        self.total_processed_l += raw_grey
        self._degrade(raw_grey)
        return clean, waste, energy

    def _degrade(self, liters: float) -> None:
        """Organic buildup clogs filters."""
        degradation = liters / FILTER_LIFE_LITERS * 1.2
        self.health = max(MIN_EFFICIENCY_FACTOR, self.health - degradation)


@dataclass
class QualityMonitor:
    """Water quality monitoring and reject/reprocess decision."""

    toc_mg_l: float = 0.5           # current total organic carbon
    conductivity_us_cm: float = 0.3  # current conductivity
    biocide_mg_l: float = 0.5       # silver-ion concentration
    rejects_l: float = 0.0          # cumulative rejected water

    def check(self, water_l: float, system_health: float) -> tuple[float, float]:
        """Check water quality. Degrade = more contaminated output.

        Returns (potable_l, rejected_l).
        Lower system health → higher contamination → more rejects.
        """
        # Contamination increases as system health decreases
        contamination_factor = 1.0 - system_health
        self.toc_mg_l = 0.5 + contamination_factor * 8.0
        self.conductivity_us_cm = 0.3 + contamination_factor * 5.0
        self.biocide_mg_l = max(0.05, 0.5 - contamination_factor * 0.4)

        passes_quality = (
            self.toc_mg_l <= TOC_LIMIT_MG_L
            and self.conductivity_us_cm <= CONDUCTIVITY_LIMIT_US_CM
            and self.biocide_mg_l >= BIOCIDE_MIN_MG_L
        )

        if passes_quality:
            return water_l, 0.0

        # Partial pass: some water is potable, rest needs reprocessing
        # Quality degrades linearly — reject fraction = contamination
        reject_fraction = min(0.5, contamination_factor)
        rejected = water_l * reject_fraction
        self.rejects_l += rejected
        return water_l - rejected, rejected


@dataclass
class WaterReclamationSystem:
    """Complete colony water recovery system.

    Integrates all four processing subsystems + quality monitoring
    into a single per-sol tick.
    """

    condensate: CondensateCollector = field(default_factory=CondensateCollector)
    urine_proc: UrineProcessor = field(default_factory=UrineProcessor)
    brine_proc: BrineProcessor = field(default_factory=BrineProcessor)
    greywater: GreywaterRecycler = field(default_factory=GreywaterRecycler)
    quality: QualityMonitor = field(default_factory=QualityMonitor)

    # Tracking
    sol: int = 0
    total_recovered_l: float = 0.0
    total_energy_kwh: float = 0.0
    total_waste_kg: float = 0.0
    history: list[dict] = field(default_factory=list)

    def tick(
        self,
        crew: int,
        available_power_kwh: float = float("inf"),
        hab_temp_c: float = 22.0,
    ) -> dict:
        """Advance one sol of water reclamation.

        Args:
            crew: number of crew in habitat
            available_power_kwh: power budget for water systems
            hab_temp_c: habitat temperature (affects condensation)

        Returns dict with sol results including recovery rate and water balance.
        """
        self.sol += 1
        remaining_power = available_power_kwh

        # Total wastewater produced this sol
        total_wastewater = crew * (
            URINE_L_PER_PERSON_SOL
            + SWEAT_RESP_L_PER_PERSON_SOL
            + GREYWATER_L_PER_PERSON_SOL
        )

        # 1. Condensate collection (lowest energy, do first)
        cond_water, cond_energy = self.condensate.collect(crew, hab_temp_c)
        cond_energy = min(cond_energy, remaining_power)
        if cond_energy < crew * SWEAT_RESP_L_PER_PERSON_SOL * CONDENSATE_ENERGY_KWH_L * 0.5:
            cond_water *= cond_energy / max(0.01, crew * SWEAT_RESP_L_PER_PERSON_SOL * CONDENSATE_ENERGY_KWH_L)
        remaining_power -= cond_energy

        # 2. Greywater recycling (second priority — high volume, low energy)
        grey_water, grey_waste, grey_energy = self.greywater.process(crew)
        if grey_energy > remaining_power:
            ratio = remaining_power / max(0.01, grey_energy)
            grey_water *= ratio
            grey_waste *= ratio
            grey_energy = remaining_power
        remaining_power -= grey_energy

        # 3. Urine processing (third — medium volume, medium energy)
        urine_water, brine_l, urine_energy = self.urine_proc.process(crew)
        if urine_energy > remaining_power:
            ratio = remaining_power / max(0.01, urine_energy)
            urine_water *= ratio
            brine_l = crew * URINE_L_PER_PERSON_SOL - urine_water
            urine_energy = remaining_power
        remaining_power -= urine_energy

        # 4. Brine processing (last — low volume, high energy, high value)
        brine_water, solid_waste, brine_energy = self.brine_proc.process(brine_l)
        if brine_energy > remaining_power:
            ratio = remaining_power / max(0.01, brine_energy)
            brine_water *= ratio
            solid_waste *= ratio
            brine_energy = remaining_power
        remaining_power -= brine_energy

        # Total raw recovery
        raw_recovered = cond_water + grey_water + urine_water + brine_water
        total_energy = cond_energy + grey_energy + urine_energy + brine_energy

        # 5. Quality check — reject contaminated water for reprocessing
        avg_health = (
            self.condensate.health
            + self.urine_proc.health
            + self.brine_proc.health
            + self.greywater.health
        ) / 4.0
        potable, rejected = self.quality.check(raw_recovered, avg_health)

        # Recovery rate
        recovery_rate = potable / max(0.01, total_wastewater)

        # Update totals
        self.total_recovered_l += potable
        self.total_energy_kwh += total_energy
        self.total_waste_kg += solid_waste + grey_waste

        result = {
            "sol": self.sol,
            "crew": crew,
            "wastewater_l": round(total_wastewater, 3),
            "condensate_l": round(cond_water, 3),
            "greywater_recovered_l": round(grey_water, 3),
            "urine_recovered_l": round(urine_water, 3),
            "brine_recovered_l": round(brine_water, 3),
            "potable_l": round(potable, 3),
            "rejected_l": round(rejected, 3),
            "recovery_rate": round(recovery_rate, 4),
            "energy_kwh": round(total_energy, 3),
            "solid_waste_kg": round(solid_waste, 3),
            "system_health": round(avg_health, 4),
            "subsystem_health": {
                "condensate": round(self.condensate.health, 4),
                "urine_processor": round(self.urine_proc.health, 4),
                "brine_processor": round(self.brine_proc.health, 4),
                "greywater": round(self.greywater.health, 4),
            },
            "quality": {
                "toc_mg_l": round(self.quality.toc_mg_l, 3),
                "conductivity_us_cm": round(self.quality.conductivity_us_cm, 3),
                "biocide_mg_l": round(self.quality.biocide_mg_l, 3),
            },
        }
        self.history.append(result)
        return result

    def perform_maintenance(self) -> dict:
        """Perform maintenance on all subsystems.

        Returns health improvements per subsystem.
        """
        improvements = {}
        for name, subsystem in [
            ("condensate", self.condensate),
            ("urine_processor", self.urine_proc),
            ("brine_processor", self.brine_proc),
            ("greywater", self.greywater),
        ]:
            old_health = subsystem.health
            subsystem.health = min(1.0, old_health + (1.0 - old_health) * MAINTENANCE_RESTORE)
            improvements[name] = round(subsystem.health - old_health, 4)
        return improvements

    def get_overall_recovery_rate(self) -> float:
        """Lifetime average recovery rate."""
        if not self.history:
            return 0.0
        return sum(h["recovery_rate"] for h in self.history) / len(self.history)

    def get_water_balance(self, crew: int) -> dict:
        """Current water balance: production vs consumption needs.

        Args:
            crew: current crew count

        Returns dict with daily water budget analysis.
        """
        daily_need = crew * 3.0  # 3.0 L/person/sol drinking + hygiene
        daily_wastewater = crew * (
            URINE_L_PER_PERSON_SOL
            + SWEAT_RESP_L_PER_PERSON_SOL
            + GREYWATER_L_PER_PERSON_SOL
        )

        # Estimate recovery from current system health
        avg_health = (
            self.condensate.health
            + self.urine_proc.health
            + self.brine_proc.health
            + self.greywater.health
        ) / 4.0
        estimated_recovery = daily_wastewater * avg_health * 0.90
        deficit = max(0.0, daily_need - estimated_recovery)

        return {
            "daily_need_l": round(daily_need, 2),
            "daily_wastewater_l": round(daily_wastewater, 2),
            "estimated_recovery_l": round(estimated_recovery, 2),
            "deficit_l": round(deficit, 2),
            "self_sufficient": deficit < 0.1,
        }
