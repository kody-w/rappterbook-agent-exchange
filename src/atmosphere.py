"""
atmosphere.py — Mars habitat atmosphere processor.

Models closed-loop life support for a pressurised Mars habitat:
  - O2 generation via water electrolysis (Sabatier + MOXIE-style ISRU)
  - CO2 scrubbing from crew respiration
  - Pressure regulation (leak rate, repressurisation)
  - Trace contaminant removal

Physical references:
  - ISS ECLSS: 0.84 kg O2/person/day, 1.04 kg CO2/person/day exhaled
  - Water electrolysis: 2 H2O → 2 H2 + O2 (9 kg water → 8 kg O2)
  - Sabatier: CO2 + 4H2 → CH4 + 2H2O (recovers 50% of water)
  - MOXIE (Mars 2020): CO2 → CO + ½O2, 6 g O2/hr at 300 W
  - Habitat target: 21 kPa O2, 79 kPa N2, total ~101 kPa
  - Mars ambient: 0.636 kPa (95% CO2)
  - Leak rate: ISS loses ~0.23 kg air/day through micro-leaks

One tick = one sol. Masses in kg, pressures in kPa, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Human metabolism
O2_KG_PER_PERSON_SOL = 0.84      # ISS ECLSS value
CO2_KG_PER_PERSON_SOL = 1.04     # exhaled CO2
H2O_METABOLIC_KG_SOL = 0.35      # metabolic water produced per person

# Electrolysis: 2H2O → 2H2 + O2
WATER_TO_O2_RATIO = 8.0 / 9.0    # 8 kg O2 from 9 kg water
ELECTROLYSIS_KWH_PER_KG_O2 = 5.0 # energy cost per kg O2 produced

# Sabatier reactor: CO2 + 4H2 → CH4 + 2H2O
# Recovers water from CO2 + H2 byproduct of electrolysis
SABATIER_CO2_TO_H2O_RATIO = 0.818  # 36 kg H2O from 44 kg CO2
SABATIER_KWH_PER_KG_CO2 = 0.5      # low energy — exothermic, just pumping

# MOXIE-style ISRU: atmospheric CO2 → O2
MOXIE_KG_O2_PER_KWH = 0.02      # 6g/hr at 300W ≈ 0.02 kg/kWh
MOXIE_EFFICIENCY_BASE = 0.85     # conversion efficiency

# Atmosphere targets
TARGET_O2_KPA = 21.0             # partial pressure of O2
TARGET_TOTAL_KPA = 101.3         # Earth-standard total pressure
TARGET_CO2_KPA = 0.04            # CO2 partial pressure (Earth normal)
CO2_DANGER_KPA = 1.0             # headaches, impaired cognition
CO2_LETHAL_KPA = 5.0             # loss of consciousness risk

# Pressure dynamics
HABITAT_LEAK_RATE_KPA_SOL = 0.023   # ~0.023% of total per sol (ISS-like)
REPRESSURISATION_KWH_PER_KPA = 2.0  # energy to pump/heat buffer gas
N2_BUFFER_AVAILABLE = True           # assume N2 extraction from regolith

# Scrubber
SCRUBBER_EFFICIENCY_BASE = 0.95     # CO2 removal per pass
SCRUBBER_KWH_PER_KG_CO2 = 1.2      # zeolite/amine swing beds
SCRUBBER_DEGRADATION_PER_SOL = 0.0001  # filter wear

# Trace contaminants
TRACE_ACCUMULATION_PER_PERSON_SOL = 0.001  # arbitrary scale 0-1
TRACE_REMOVAL_PER_SOL = 0.02       # activated charcoal scrubbing


@dataclass
class AtmosphereState:
    """Snapshot of habitat atmosphere at a given sol."""
    o2_kpa: float = TARGET_O2_KPA
    co2_kpa: float = TARGET_CO2_KPA
    n2_kpa: float = TARGET_TOTAL_KPA - TARGET_O2_KPA - TARGET_CO2_KPA
    total_kpa: float = TARGET_TOTAL_KPA
    trace_contaminants: float = 0.0
    scrubber_health: float = 1.0
    sabatier_water_recovered_kg: float = 0.0
    moxie_o2_produced_kg: float = 0.0
    electrolysis_o2_produced_kg: float = 0.0
    co2_scrubbed_kg: float = 0.0
    energy_used_kwh: float = 0.0
    leak_loss_kpa: float = 0.0
    sol: int = 0
    alert: str | None = None


@dataclass
class AtmosphereProcessor:
    """Closed-loop atmosphere management for a Mars habitat.

    Tracks O2/CO2/N2 partial pressures and manages life support
    equipment: electrolyser, CO2 scrubber, Sabatier reactor, MOXIE unit.

    Parameters:
        habitat_volume_m3: pressurised volume of the habitat
        electrolyser_capacity_kg_sol: max O2 output per sol
        moxie_installed: whether MOXIE ISRU unit is available
        scrubber_health: 0.0-1.0 efficiency of CO2 scrubber
    """
    habitat_volume_m3: float = 500.0
    electrolyser_capacity_kg_sol: float = 20.0
    moxie_installed: bool = False
    scrubber_health: float = 1.0

    # Internal atmosphere state
    o2_kpa: float = TARGET_O2_KPA
    co2_kpa: float = TARGET_CO2_KPA
    n2_kpa: float = field(default_factory=lambda: TARGET_TOTAL_KPA - TARGET_O2_KPA - TARGET_CO2_KPA)

    # Cumulative trackers
    trace_contaminants: float = 0.0
    total_water_consumed_kg: float = 0.0
    total_energy_used_kwh: float = 0.0
    sol: int = 0

    def total_pressure(self) -> float:
        """Current total habitat pressure in kPa."""
        return self.o2_kpa + self.co2_kpa + self.n2_kpa

    def o2_fraction(self) -> float:
        """O2 mole fraction (approximation via partial pressure)."""
        total = self.total_pressure()
        if total <= 0:
            return 0.0
        return self.o2_kpa / total

    def co2_status(self) -> str:
        """Human-readable CO2 danger level."""
        if self.co2_kpa >= CO2_LETHAL_KPA:
            return "lethal"
        if self.co2_kpa >= CO2_DANGER_KPA:
            return "dangerous"
        if self.co2_kpa >= 0.2:
            return "elevated"
        return "nominal"

    def pressure_status(self) -> str:
        """Human-readable pressure status."""
        total = self.total_pressure()
        if total < 50.0:
            return "critical"
        if total < 80.0:
            return "low"
        if total > 120.0:
            return "over_pressurised"
        return "nominal"

    def _apply_leak(self) -> float:
        """Simulate micro-leak pressure loss. Returns kPa lost."""
        total = self.total_pressure()
        if total <= 0:
            return 0.0
        leak = HABITAT_LEAK_RATE_KPA_SOL
        # Proportional leak from each component
        fraction_o2 = self.o2_kpa / total
        fraction_co2 = self.co2_kpa / total
        fraction_n2 = self.n2_kpa / total

        self.o2_kpa = max(0.0, self.o2_kpa - leak * fraction_o2)
        self.co2_kpa = max(0.0, self.co2_kpa - leak * fraction_co2)
        self.n2_kpa = max(0.0, self.n2_kpa - leak * fraction_n2)
        return leak

    def _crew_respiration(self, population: int) -> tuple[float, float]:
        """Model crew O2 consumption and CO2 production.

        Returns (o2_consumed_kg, co2_produced_kg).
        """
        o2_consumed = O2_KG_PER_PERSON_SOL * population
        co2_produced = CO2_KG_PER_PERSON_SOL * population
        return o2_consumed, co2_produced

    def _kg_to_kpa(self, mass_kg: float, molar_mass: float) -> float:
        """Convert gas mass to partial pressure contribution.

        Uses ideal gas law: PV = nRT
          n = mass_kg / molar_mass * 1000  (moles)
          R = 8.314 J/(mol·K)
          T = 293 K (20°C habitat)
          P = nRT / V  (Pa) → convert to kPa
        """
        if self.habitat_volume_m3 <= 0:
            return 0.0
        moles = (mass_kg * 1000.0) / molar_mass
        pressure_pa = moles * 8.314 * 293.0 / self.habitat_volume_m3
        return pressure_pa / 1000.0

    def _scrub_co2(self, co2_produced_kg: float, power_available_kwh: float) -> tuple[float, float]:
        """Scrub CO2 from habitat atmosphere.

        Returns (co2_removed_kg, energy_used_kwh).
        """
        eff = SCRUBBER_EFFICIENCY_BASE * self.scrubber_health
        eff = max(0.0, min(1.0, eff))

        max_removal = co2_produced_kg * eff
        energy_needed = max_removal * SCRUBBER_KWH_PER_KG_CO2
        if energy_needed > power_available_kwh and energy_needed > 0:
            ratio = power_available_kwh / energy_needed
            max_removal *= ratio
            energy_needed = power_available_kwh

        return max_removal, energy_needed

    def _electrolyse(self, o2_needed_kg: float, water_available_kg: float,
                     power_available_kwh: float) -> tuple[float, float, float]:
        """Electrolyse water to produce O2.

        Returns (o2_produced_kg, water_consumed_kg, energy_used_kwh).
        """
        # Capacity limit
        o2_target = min(o2_needed_kg, self.electrolyser_capacity_kg_sol)

        # Water limit: need water/O2 ratio of 9/8
        water_needed = o2_target / WATER_TO_O2_RATIO
        if water_needed > water_available_kg:
            water_needed = water_available_kg
            o2_target = water_needed * WATER_TO_O2_RATIO

        # Power limit
        energy_needed = o2_target * ELECTROLYSIS_KWH_PER_KG_O2
        if energy_needed > power_available_kwh and energy_needed > 0:
            ratio = power_available_kwh / energy_needed
            o2_target *= ratio
            water_needed *= ratio
            energy_needed = power_available_kwh

        return o2_target, water_needed, energy_needed

    def _run_sabatier(self, co2_removed_kg: float, power_available_kwh: float) -> tuple[float, float]:
        """Run Sabatier reactor on scrubbed CO2 to recover water.

        Returns (water_recovered_kg, energy_used_kwh).
        """
        energy_needed = co2_removed_kg * SABATIER_KWH_PER_KG_CO2
        if energy_needed > power_available_kwh and energy_needed > 0:
            ratio = power_available_kwh / energy_needed
            co2_processed = co2_removed_kg * ratio
            energy_needed = power_available_kwh
        else:
            co2_processed = co2_removed_kg

        water_recovered = co2_processed * SABATIER_CO2_TO_H2O_RATIO
        return water_recovered, energy_needed

    def _run_moxie(self, power_available_kwh: float) -> tuple[float, float]:
        """MOXIE ISRU: extract O2 from Mars atmospheric CO2.

        Returns (o2_produced_kg, energy_used_kwh).
        """
        if not self.moxie_installed or power_available_kwh <= 0:
            return 0.0, 0.0

        o2_produced = power_available_kwh * MOXIE_KG_O2_PER_KWH * MOXIE_EFFICIENCY_BASE
        return o2_produced, power_available_kwh

    def _repressurize(self, power_available_kwh: float) -> tuple[float, float]:
        """Add buffer gas (N2) if total pressure is low.

        Returns (n2_added_kpa, energy_used_kwh).
        """
        total = self.total_pressure()
        deficit = TARGET_TOTAL_KPA - total

        if deficit <= 0 or not N2_BUFFER_AVAILABLE:
            return 0.0, 0.0

        # Only repressurize up to what power allows
        energy_needed = deficit * REPRESSURISATION_KWH_PER_KPA
        if energy_needed > power_available_kwh and energy_needed > 0:
            deficit = power_available_kwh / REPRESSURISATION_KWH_PER_KPA
            energy_needed = power_available_kwh

        self.n2_kpa += deficit
        return deficit, energy_needed

    def _update_traces(self, population: int) -> None:
        """Accumulate and scrub trace contaminants."""
        self.trace_contaminants += TRACE_ACCUMULATION_PER_PERSON_SOL * population
        self.trace_contaminants -= TRACE_REMOVAL_PER_SOL
        self.trace_contaminants = max(0.0, min(1.0, self.trace_contaminants))

    def tick(self, population: int, water_available_kg: float,
             power_available_kwh: float) -> AtmosphereState:
        """Advance one sol of atmosphere processing.

        Args:
            population: number of crew breathing the habitat air
            water_available_kg: water budget for electrolysis
            power_available_kwh: power budget for life support

        Returns:
            AtmosphereState snapshot after processing.
        """
        self.sol += 1
        energy_budget = max(0.0, power_available_kwh)
        energy_used = 0.0

        # --- 1. Leak ---
        leak_loss = self._apply_leak()

        # --- 2. Crew respiration ---
        o2_consumed_kg, co2_produced_kg = self._crew_respiration(population)

        # Remove O2, add CO2 (as partial pressure deltas)
        o2_delta_kpa = self._kg_to_kpa(o2_consumed_kg, 32.0)  # O2 molar mass = 32
        co2_delta_kpa = self._kg_to_kpa(co2_produced_kg, 44.0)  # CO2 molar mass = 44

        self.o2_kpa = max(0.0, self.o2_kpa - o2_delta_kpa)
        self.co2_kpa = max(0.0, self.co2_kpa + co2_delta_kpa)

        # --- 3. CO2 scrubbing ---
        co2_removed_kg, scrub_energy = self._scrub_co2(
            co2_produced_kg, energy_budget - energy_used
        )
        energy_used += scrub_energy

        co2_removal_kpa = self._kg_to_kpa(co2_removed_kg, 44.0)
        self.co2_kpa = max(0.0, self.co2_kpa - co2_removal_kpa)

        # --- 4. Sabatier water recovery ---
        water_recovered, sabatier_energy = self._run_sabatier(
            co2_removed_kg, energy_budget - energy_used
        )
        energy_used += sabatier_energy

        # --- 5. O2 generation via electrolysis ---
        o2_from_electrolysis, water_consumed, elec_energy = self._electrolyse(
            o2_consumed_kg, water_available_kg, energy_budget - energy_used
        )
        energy_used += elec_energy
        self.total_water_consumed_kg += water_consumed

        o2_gen_kpa = self._kg_to_kpa(o2_from_electrolysis, 32.0)
        self.o2_kpa += o2_gen_kpa

        # --- 6. MOXIE supplemental O2 ---
        moxie_o2, moxie_energy = self._run_moxie(
            max(0.0, (energy_budget - energy_used) * 0.3)  # allocate 30% of remaining
        )
        energy_used += moxie_energy

        moxie_o2_kpa = self._kg_to_kpa(moxie_o2, 32.0)
        self.o2_kpa += moxie_o2_kpa

        # --- 7. Repressurize with N2 buffer ---
        n2_added, repress_energy = self._repressurize(energy_budget - energy_used)
        energy_used += repress_energy

        # --- 8. Trace contaminants ---
        self._update_traces(population)

        # --- 9. Scrubber degradation ---
        self.scrubber_health = max(0.0, self.scrubber_health - SCRUBBER_DEGRADATION_PER_SOL)

        # --- 10. Total energy ---
        self.total_energy_used_kwh += energy_used

        # --- Generate alert ---
        alert = None
        if self.co2_kpa >= CO2_LETHAL_KPA:
            alert = "CO2_LETHAL"
        elif self.co2_kpa >= CO2_DANGER_KPA:
            alert = "CO2_DANGER"
        elif self.o2_kpa < 16.0:
            alert = "O2_LOW"
        elif self.total_pressure() < 50.0:
            alert = "PRESSURE_CRITICAL"

        return AtmosphereState(
            o2_kpa=round(self.o2_kpa, 4),
            co2_kpa=round(self.co2_kpa, 4),
            n2_kpa=round(self.n2_kpa, 4),
            total_kpa=round(self.total_pressure(), 4),
            trace_contaminants=round(self.trace_contaminants, 6),
            scrubber_health=round(self.scrubber_health, 6),
            sabatier_water_recovered_kg=round(water_recovered, 4),
            moxie_o2_produced_kg=round(moxie_o2, 4),
            electrolysis_o2_produced_kg=round(o2_from_electrolysis, 4),
            co2_scrubbed_kg=round(co2_removed_kg, 4),
            energy_used_kwh=round(energy_used, 4),
            leak_loss_kpa=round(leak_loss, 4),
            sol=self.sol,
            alert=alert,
        )

    def snapshot(self) -> dict:
        """JSON-serialisable state for persistence."""
        return {
            "sol": self.sol,
            "o2_kpa": round(self.o2_kpa, 4),
            "co2_kpa": round(self.co2_kpa, 4),
            "n2_kpa": round(self.n2_kpa, 4),
            "total_kpa": round(self.total_pressure(), 4),
            "scrubber_health": round(self.scrubber_health, 6),
            "trace_contaminants": round(self.trace_contaminants, 6),
            "co2_status": self.co2_status(),
            "pressure_status": self.pressure_status(),
            "o2_fraction": round(self.o2_fraction(), 4),
            "total_water_consumed_kg": round(self.total_water_consumed_kg, 2),
            "total_energy_used_kwh": round(self.total_energy_used_kwh, 2),
        }
