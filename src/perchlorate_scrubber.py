"""perchlorate_scrubber.py — Mars Regolith Perchlorate Remediation System.

Models three complementary perchlorate (ClO₄⁻) removal pathways for
making Mars regolith safe for agriculture, habitat air, and water supply.

Pathways modelled
-----------------
* **Ion-exchange column** — Regolith slurry passes through a selective
  anion-exchange resin (Type-I strong-base, quaternary ammonium).
  ClO₄⁻ is captured; clean effluent contains < 15 µg/L perchlorate.
  Resin capacity: ~1.4 eq/L → ~140 g ClO₄⁻ per liter of resin before
  regeneration with NaCl brine.

* **Catalytic reduction reactor** — Captured perchlorate is destroyed
  by catalytic hydrogenation:  ClO₄⁻ + 4H₂ → Cl⁻ + 4H₂O.
  Re/Pd bimetallic catalyst on activated carbon.  Needs H₂ feed
  (from water electrolysis) and 80-120 °C.  Converts toxic perchlorate
  into harmless chloride salt + water.

* **Bioremediation tank** — Perchlorate-reducing bacteria (PRB) such as
  *Dechloromonas* and *Azospira* use ClO₄⁻ as a terminal electron
  acceptor under anaerobic conditions.  Slow but self-renewing.
  Rate: ~50 mg ClO₄⁻ / L / day at 30 °C, halves for every 10 °C drop.

Physical references:
  - Phoenix lander (2008): 0.4–0.6 wt% perchlorate in soil
  - Curiosity SAM: 0.5–1.0 wt% in Gale Crater soil
  - Human toxicity: thyroid disruption at > 15 µg/L in drinking water
  - EPA MCL for perchlorate: 56 µg/L (proposed); CA: 6 µg/L
  - Ion-exchange capacity: Type-I resin ~1.4 eq/L (Purolite A-520E)
  - Catalytic reduction: Re-Pd/C at 95 °C, τ½ ≈ 30 min
  - PRB doubling time: ~6 h at 30 °C (Dechloromonas aromatica RCB)

One tick = one sol.  Mass in kg, volume in liters, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Mars regolith perchlorate content (mass fraction)
PERCHLORATE_FRACTION_LOW = 0.004   # 0.4 wt% (Phoenix low end)
PERCHLORATE_FRACTION_HIGH = 0.010  # 1.0 wt% (Curiosity high end)
PERCHLORATE_FRACTION_MEAN = 0.007  # 0.7 wt% typical

# Safety thresholds
SAFE_SOIL_PERCHLORATE_PPM = 500.0     # ppm in treated soil (for crops; conservative)
SAFE_WATER_PERCHLORATE_UG_L = 15.0    # µg/L in water (drinking standard)

# Ion-exchange column
RESIN_CAPACITY_G_CLO4_PER_L = 140.0   # grams ClO₄⁻ per liter resin
RESIN_FLOW_RATE_L_PER_H = 10.0        # slurry throughput per hour
COLUMN_ENERGY_KWH_PER_M3 = 0.3        # pump energy per m³ slurry
REGEN_NACL_KG_PER_L_RESIN = 0.3       # NaCl needed for regeneration
REGEN_WATER_L_PER_L_RESIN = 5.0       # water needed for regen rinse
COLUMN_EFFICIENCY_NEW = 0.97           # fresh resin removal efficiency
RESIN_DEGRADATION_PER_CYCLE = 0.002    # efficiency loss per regen cycle

# Catalytic reduction reactor
CATALYST_RATE_G_PER_H = 50.0          # g ClO₄⁻ destroyed per hour per kg catalyst
H2_STOICH_KG_PER_KG_CLO4 = 0.081      # 4 mol H₂ per mol ClO₄⁻ (4×2.016/99.45)
REACTOR_TEMP_C = 95.0                  # operating temperature
REACTOR_ENERGY_KWH_PER_KG_CLO4 = 8.0  # heating + pumping per kg destroyed
CATALYST_LIFE_KG_CLO4 = 500.0         # kg perchlorate destroyed before replacement
CATALYST_MIN_EFFICIENCY = 0.50
CATALYST_DEGRADATION_PER_KG = 0.001    # efficiency loss per kg processed

# Bioremediation tank
BIO_RATE_MG_L_DAY_30C = 50.0          # mg ClO₄⁻ reduced per liter per day at 30 °C
BIO_Q10 = 2.0                         # rate halves per 10 °C drop
BIO_OPTIMAL_TEMP_C = 30.0
BIO_MIN_TEMP_C = 5.0                  # bacteria dormant below this
BIO_TANK_ENERGY_KWH_PER_M3 = 0.1      # aeration + mixing per m³ per sol
BIO_CULTURE_DOUBLING_H = 6.0          # population doubling time at optimal
BIO_MAX_POPULATION = 1.0              # normalized 0-1 carrying capacity
BIO_DEATH_RATE_PER_SOL = 0.02         # natural die-off fraction per sol

# Regolith processing
SLURRY_WATER_RATIO = 2.0              # liters water per kg regolith
REGOLITH_DENSITY_KG_M3 = 1500.0

# General
MAINTENANCE_RESTORE_FRACTION = 0.80    # how much maintenance recovers


# ---------------------------------------------------------------------------
# Ion-exchange column
# ---------------------------------------------------------------------------

@dataclass
class IonExchangeColumn:
    """Selective anion-exchange resin column for ClO₄⁻ capture."""

    resin_volume_l: float = 50.0
    efficiency: float = COLUMN_EFFICIENCY_NEW
    total_clo4_captured_g: float = 0.0
    cycles: int = 0
    loaded_g: float = 0.0              # current ClO₄⁻ load on resin

    @property
    def capacity_g(self) -> float:
        """Total ClO₄⁻ the resin can hold before regeneration."""
        return self.resin_volume_l * RESIN_CAPACITY_G_CLO4_PER_L

    @property
    def load_fraction(self) -> float:
        """How full the resin is (0 = fresh, 1 = exhausted)."""
        cap = self.capacity_g
        if cap <= 0:
            return 1.0
        return min(1.0, self.loaded_g / cap)

    @property
    def needs_regeneration(self) -> bool:
        """True when resin is > 85% loaded."""
        return self.load_fraction > 0.85

    def treat_regolith(self, regolith_kg: float, perchlorate_fraction: float
                       ) -> dict:
        """Pass regolith slurry through column.

        Args:
            regolith_kg: mass of regolith to treat
            perchlorate_fraction: ClO₄⁻ mass fraction in regolith

        Returns:
            dict with treated_kg, clo4_removed_g, clo4_remaining_ppm,
            water_used_l, energy_kwh, safe_for_crops
        """
        regolith_kg = max(0.0, regolith_kg)
        perchlorate_fraction = max(0.0, min(1.0, perchlorate_fraction))

        incoming_clo4_g = regolith_kg * perchlorate_fraction * 1000.0
        available_capacity = max(0.0, self.capacity_g - self.loaded_g)

        # Effective removal limited by resin load and efficiency
        load_penalty = max(0.3, 1.0 - self.load_fraction * 0.5)
        effective_eff = self.efficiency * load_penalty

        removable = min(incoming_clo4_g * effective_eff, available_capacity)
        remaining_g = incoming_clo4_g - removable

        # Convert remaining to ppm in treated regolith
        remaining_ppm = (remaining_g / max(0.001, regolith_kg)) * 1000.0

        # Slurry water
        water_l = regolith_kg * SLURRY_WATER_RATIO
        slurry_m3 = water_l / 1000.0
        energy = slurry_m3 * COLUMN_ENERGY_KWH_PER_M3

        self.loaded_g += removable
        self.total_clo4_captured_g += removable

        return {
            "treated_kg": round(regolith_kg, 3),
            "clo4_removed_g": round(removable, 3),
            "clo4_remaining_ppm": round(remaining_ppm, 3),
            "water_used_l": round(water_l, 3),
            "energy_kwh": round(energy, 4),
            "safe_for_crops": remaining_ppm <= SAFE_SOIL_PERCHLORATE_PPM,
            "resin_load_fraction": round(self.load_fraction, 4),
        }

    def regenerate(self) -> dict:
        """Regenerate resin with NaCl brine flush.

        Returns:
            dict with nacl_kg, water_l, clo4_released_g, efficiency_after
        """
        released = self.loaded_g
        self.loaded_g = 0.0
        self.cycles += 1
        self.efficiency = max(
            0.50,
            self.efficiency - RESIN_DEGRADATION_PER_CYCLE
        )

        nacl = self.resin_volume_l * REGEN_NACL_KG_PER_L_RESIN
        water = self.resin_volume_l * REGEN_WATER_L_PER_L_RESIN

        return {
            "nacl_kg": round(nacl, 3),
            "water_l": round(water, 3),
            "clo4_released_g": round(released, 3),
            "efficiency_after": round(self.efficiency, 4),
            "total_cycles": self.cycles,
        }


# ---------------------------------------------------------------------------
# Catalytic reduction reactor
# ---------------------------------------------------------------------------

@dataclass
class CatalyticReactor:
    """Re/Pd bimetallic catalyst reactor: ClO₄⁻ + 4H₂ → Cl⁻ + 4H₂O."""

    catalyst_kg: float = 5.0
    efficiency: float = 1.0
    total_destroyed_g: float = 0.0
    catalyst_age_kg: float = 0.0       # cumulative kg ClO₄⁻ processed

    @property
    def remaining_life_fraction(self) -> float:
        """Fraction of catalyst life remaining."""
        return max(0.0, 1.0 - self.catalyst_age_kg / CATALYST_LIFE_KG_CLO4)

    def destroy_perchlorate(self, clo4_g: float, h2_available_g: float,
                            power_available_kwh: float) -> dict:
        """Catalytically reduce perchlorate to chloride + water.

        Args:
            clo4_g: grams of ClO₄⁻ in brine/solution
            h2_available_g: grams H₂ available
            power_available_kwh: power budget

        Returns:
            dict with destroyed_g, h2_consumed_g, energy_kwh,
            chloride_produced_g, water_produced_g
        """
        clo4_g = max(0.0, clo4_g)
        h2_available_g = max(0.0, h2_available_g)
        power_available_kwh = max(0.0, power_available_kwh)

        # Rate limit: catalyst capacity per sol (24.6 h)
        max_rate = (self.catalyst_kg * CATALYST_RATE_G_PER_H * 24.6
                    * self.efficiency)

        # H₂ limit: stoichiometric requirement (kg/kg ratio = g/g ratio)
        h2_needed_per_g = H2_STOICH_KG_PER_KG_CLO4  # g H₂ per g ClO₄⁻
        h2_limited = h2_available_g / max(0.001, h2_needed_per_g)

        # Power limit
        energy_per_g = REACTOR_ENERGY_KWH_PER_KG_CLO4 / 1000.0
        power_limited = power_available_kwh / max(0.001, energy_per_g)

        destroyed = min(clo4_g, max_rate, h2_limited, power_limited)

        h2_consumed = destroyed * h2_needed_per_g
        energy = destroyed * energy_per_g

        # Products: ClO₄⁻ → Cl⁻ + 4H₂O
        # Molar masses: ClO₄⁻=99.45, Cl⁻=35.45, H₂O=18.015
        chloride_g = destroyed * (35.45 / 99.45)
        water_g = destroyed * (4 * 18.015 / 99.45)

        # Catalyst aging
        self.catalyst_age_kg += destroyed / 1000.0
        self.efficiency = max(
            CATALYST_MIN_EFFICIENCY,
            self.efficiency - (destroyed / 1000.0) * CATALYST_DEGRADATION_PER_KG
        )
        self.total_destroyed_g += destroyed

        return {
            "destroyed_g": round(destroyed, 3),
            "h2_consumed_g": round(h2_consumed, 3),
            "energy_kwh": round(energy, 4),
            "chloride_produced_g": round(chloride_g, 3),
            "water_produced_g": round(water_g, 3),
            "catalyst_life_remaining": round(self.remaining_life_fraction, 4),
        }


# ---------------------------------------------------------------------------
# Bioremediation tank
# ---------------------------------------------------------------------------

@dataclass
class BioremediationTank:
    """Anaerobic perchlorate-reducing bacteria (PRB) culture tank."""

    volume_l: float = 500.0
    population: float = 0.5            # normalized 0-1
    temperature_c: float = 25.0
    total_reduced_g: float = 0.0

    @property
    def reduction_rate_mg_l_sol(self) -> float:
        """Current perchlorate reduction rate in mg/L/sol.

        Q10 model: rate doubles per 10 °C increase from reference.
        """
        if self.temperature_c < BIO_MIN_TEMP_C:
            return 0.0
        temp_factor = BIO_Q10 ** ((self.temperature_c - BIO_OPTIMAL_TEMP_C) / 10.0)
        return BIO_RATE_MG_L_DAY_30C * temp_factor * self.population

    def tick(self, clo4_concentration_mg_l: float,
             temperature_c: float | None = None) -> dict:
        """Advance one sol of bioremediation.

        Args:
            clo4_concentration_mg_l: perchlorate in tank solution
            temperature_c: tank temperature (if None, use stored)

        Returns:
            dict with reduced_mg, remaining_mg_l, population,
            energy_kwh
        """
        if temperature_c is not None:
            self.temperature_c = temperature_c

        rate = self.reduction_rate_mg_l_sol
        total_clo4_mg = clo4_concentration_mg_l * self.volume_l

        reduced_mg = min(rate * self.volume_l, total_clo4_mg)
        remaining_mg = total_clo4_mg - reduced_mg
        remaining_mg_l = remaining_mg / max(0.001, self.volume_l)

        self.total_reduced_g += reduced_mg / 1000.0

        # Population dynamics: growth when food available, die-off otherwise
        if clo4_concentration_mg_l > 1.0 and self.temperature_c >= BIO_MIN_TEMP_C:
            growth = min(
                0.1,  # max 10% growth per sol
                (self.population * 0.15)
                * (self.temperature_c / BIO_OPTIMAL_TEMP_C)
            )
        else:
            growth = 0.0
        death = self.population * BIO_DEATH_RATE_PER_SOL
        self.population = max(0.01, min(BIO_MAX_POPULATION,
                                        self.population + growth - death))

        energy = (self.volume_l / 1000.0) * BIO_TANK_ENERGY_KWH_PER_M3

        return {
            "reduced_mg": round(reduced_mg, 3),
            "remaining_mg_l": round(remaining_mg_l, 3),
            "population": round(self.population, 4),
            "energy_kwh": round(energy, 4),
            "temperature_c": self.temperature_c,
        }


# ---------------------------------------------------------------------------
# Integrated scrubber system
# ---------------------------------------------------------------------------

@dataclass
class PerchlorateScrubber:
    """Complete three-stage perchlorate remediation system.

    Stage 1: Ion-exchange capture (fast, bulk removal)
    Stage 2: Catalytic destruction of captured perchlorate
    Stage 3: Bioremediation polishing of residual traces

    The system processes regolith to make it safe for greenhouse use
    and produces clean chloride salt + water as byproducts.
    """

    column: IonExchangeColumn = field(default_factory=IonExchangeColumn)
    reactor: CatalyticReactor = field(default_factory=CatalyticReactor)
    bio_tank: BioremediationTank = field(default_factory=BioremediationTank)

    sol: int = 0
    total_regolith_treated_kg: float = 0.0
    total_perchlorate_removed_g: float = 0.0
    total_energy_kwh: float = 0.0
    history: list[dict] = field(default_factory=list)

    def tick(
        self,
        regolith_kg: float,
        perchlorate_fraction: float = PERCHLORATE_FRACTION_MEAN,
        h2_available_g: float = 500.0,
        power_available_kwh: float = 50.0,
        bio_temp_c: float = 25.0,
    ) -> dict:
        """Process one sol of regolith through the three-stage scrubber.

        Args:
            regolith_kg: raw regolith to process
            perchlorate_fraction: ClO₄⁻ mass fraction in regolith
            h2_available_g: hydrogen gas available for catalysis
            power_available_kwh: total power budget
            bio_temp_c: bioremediation tank temperature

        Returns:
            dict with per-stage results and overall metrics
        """
        self.sol += 1
        regolith_kg = max(0.0, regolith_kg)
        perchlorate_fraction = max(0.0, min(1.0, perchlorate_fraction))

        incoming_clo4_g = regolith_kg * perchlorate_fraction * 1000.0

        # Stage 1: Ion-exchange — bulk capture
        if self.column.needs_regeneration:
            regen = self.column.regenerate()
            released_g = regen["clo4_released_g"]
        else:
            regen = None
            released_g = 0.0

        ix_result = self.column.treat_regolith(regolith_kg, perchlorate_fraction)
        stage1_energy = ix_result["energy_kwh"]
        remaining_power = power_available_kwh - stage1_energy

        # Stage 2: Catalytic destruction of captured perchlorate
        # Feed = ClO₄⁻ from current capture + any released during regen
        catalyst_feed_g = ix_result["clo4_removed_g"] + released_g
        cat_result = self.reactor.destroy_perchlorate(
            catalyst_feed_g, h2_available_g, max(0.0, remaining_power)
        )
        stage2_energy = cat_result["energy_kwh"]
        remaining_power -= stage2_energy

        # Stage 3: Bioremediation polishing of residual in treated soil
        # Convert remaining soil ppm to solution concentration for bio tank
        residual_mg = ix_result["clo4_remaining_ppm"] * regolith_kg
        bio_concentration = residual_mg / max(0.001, self.bio_tank.volume_l)
        bio_result = self.bio_tank.tick(bio_concentration, bio_temp_c)
        stage3_energy = bio_result["energy_kwh"]

        # Total perchlorate removed across all stages
        total_removed_g = (ix_result["clo4_removed_g"]
                           + bio_result["reduced_mg"] / 1000.0)
        total_energy = stage1_energy + stage2_energy + stage3_energy

        # Final soil quality
        bio_removed_from_soil_mg = bio_result["reduced_mg"]
        final_remaining_ppm = max(
            0.0,
            ix_result["clo4_remaining_ppm"]
            - (bio_removed_from_soil_mg / max(0.001, regolith_kg))
        )

        # Overall removal rate
        removal_rate = (total_removed_g / max(0.001, incoming_clo4_g)
                        if incoming_clo4_g > 0 else 0.0)

        self.total_regolith_treated_kg += regolith_kg
        self.total_perchlorate_removed_g += total_removed_g
        self.total_energy_kwh += total_energy

        result = {
            "sol": self.sol,
            "regolith_kg": round(regolith_kg, 3),
            "incoming_clo4_g": round(incoming_clo4_g, 3),
            "stage1_ion_exchange": ix_result,
            "stage1_regeneration": regen,
            "stage2_catalytic": cat_result,
            "stage3_bioremediation": bio_result,
            "final_soil_ppm": round(final_remaining_ppm, 3),
            "safe_for_crops": final_remaining_ppm <= SAFE_SOIL_PERCHLORATE_PPM,
            "total_removed_g": round(total_removed_g, 3),
            "removal_rate": round(min(1.0, removal_rate), 4),
            "total_energy_kwh": round(total_energy, 4),
            "overall": {
                "total_regolith_treated_kg": round(
                    self.total_regolith_treated_kg, 1),
                "total_perchlorate_removed_g": round(
                    self.total_perchlorate_removed_g, 1),
                "lifetime_energy_kwh": round(self.total_energy_kwh, 2),
            },
        }
        self.history.append(result)
        return result

    def perform_maintenance(self) -> dict:
        """Maintain all subsystems: regen column, top up catalyst."""
        regen = None
        if self.column.load_fraction > 0.5:
            regen = self.column.regenerate()

        old_eff = self.reactor.efficiency
        self.reactor.efficiency = min(
            1.0,
            old_eff + (1.0 - old_eff) * MAINTENANCE_RESTORE_FRACTION
        )

        old_pop = self.bio_tank.population
        self.bio_tank.population = min(
            BIO_MAX_POPULATION,
            old_pop + (BIO_MAX_POPULATION - old_pop) * 0.3
        )

        return {
            "column_regenerated": regen is not None,
            "column_regen": regen,
            "catalyst_efficiency_before": round(old_eff, 4),
            "catalyst_efficiency_after": round(self.reactor.efficiency, 4),
            "bio_population_before": round(old_pop, 4),
            "bio_population_after": round(self.bio_tank.population, 4),
        }

    def get_status(self) -> dict:
        """Return current system status summary."""
        return {
            "sol": self.sol,
            "column_load": round(self.column.load_fraction, 4),
            "column_efficiency": round(self.column.efficiency, 4),
            "column_needs_regen": self.column.needs_regeneration,
            "catalyst_life": round(self.reactor.remaining_life_fraction, 4),
            "catalyst_efficiency": round(self.reactor.efficiency, 4),
            "bio_population": round(self.bio_tank.population, 4),
            "bio_rate_mg_l_sol": round(
                self.bio_tank.reduction_rate_mg_l_sol, 3),
            "total_treated_kg": round(self.total_regolith_treated_kg, 1),
            "total_removed_g": round(self.total_perchlorate_removed_g, 1),
        }
