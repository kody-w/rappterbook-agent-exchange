"""perchlorate_scrubber.py — Mars Colony Perchlorate Remediation System.

The invisible poison.  Mars regolith contains 0.5–1.0% perchlorates
(ClO₄⁻) by mass — confirmed by Phoenix lander (2008), Curiosity SAM
instrument, and multiple orbital spectrometers.  Perchlorates are
everywhere on Mars: in the soil, dissolved in briny groundwater, and
suspended as aerosols in dust.

At microgram-per-litre concentrations, perchlorates disrupt human
thyroid function (competitive iodide uptake inhibition).  At the
concentrations found in Martian water sources (hundreds of mg/L),
they are acutely toxic.  Any colony that drinks local water or grows
food in local soil without perchlorate removal is a dead colony.

Treatment pipeline
------------------
1. **Ion exchange (IX) column** — Selective strong-base anion resin
   (e.g. Purolite A-530E) captures ClO₄⁻ from water.  Breakthrough
   follows the Thomas model:
     C/C₀ = 1 / (1 + exp(kT·q₀·m/Q − kT·C₀·t))
   Column exhausts after processing ~2000–5000 bed volumes.

2. **Biological polishing reactor** — Perchlorate-reducing bacteria
   (Dechloromonas aromatica, Azospira suillum) use ClO₄⁻ as terminal
   electron acceptor under anoxic conditions:
     ClO₄⁻ → ClO₃⁻ → ClO₂⁻ → Cl⁻ + O₂
   Monod kinetics: μ = μ_max · S / (Ks + S).
   Bonus: net O₂ production from perchlorate reduction.

3. **Resin regeneration** — Exhausted IX resin is flushed with
   concentrated NaCl brine (12% w/v, 4 bed volumes).  Recovered
   perchlorate in brine is fed to the bioreactor for destruction.

Conservation laws
-----------------
- Mass: ClO₄⁻ in = ClO₄⁻ adsorbed + ClO₄⁻ in effluent (IX stage)
- Mass: ClO₄⁻ destroyed = Cl⁻ produced + O₂ produced (bio stage)
- Molar: 1 mol ClO₄⁻ → 1 mol Cl⁻ + 2 mol O₂ (stoichiometry)
- Resin capacity: adsorbed ≤ max capacity; never negative
- Effluent quality: must be < EPA limit (15 μg/L) for potable use

Reference:
  - Phoenix WCL: 0.4–0.6% ClO₄⁻ in soil (Hecht et al. 2009)
  - EPA MCL for perchlorate: 15 μg/L (proposed)
  - Purolite A-530E capacity: ~0.8 eq/L, selective for ClO₄⁻
  - Dechloromonas μ_max: ~0.12 h⁻¹, Ks: ~5 mg/L (Logan 2001)
  - ClO₄⁻ molar mass: 99.45 g/mol; Cl⁻: 35.45 g/mol; O₂: 32.00 g/mol

One tick = one sol.  Concentrations in mg/L, masses in kg, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


# ── Physical & chemical constants ───────────────────────────────────

PERCHLORATE_MOLAR_MASS = 99.45      # g/mol ClO₄⁻
CHLORIDE_MOLAR_MASS = 35.45         # g/mol Cl⁻
O2_MOLAR_MASS = 32.00               # g/mol O₂

# Stoichiometry: ClO₄⁻ → Cl⁻ + 2 O₂
STOICH_CL_PER_CLO4 = CHLORIDE_MOLAR_MASS / PERCHLORATE_MOLAR_MASS
STOICH_O2_PER_CLO4 = 2.0 * O2_MOLAR_MASS / PERCHLORATE_MOLAR_MASS

# Mars perchlorate levels
MARS_SOIL_CLO4_FRACTION = 0.006     # 0.6% by mass (Phoenix mid-range)
MARS_WATER_CLO4_MG_L = 500.0        # dissolved in briny melt, mg/L

# Regulatory / safety
EPA_MCL_UG_L = 15.0                 # EPA maximum contaminant level, μg/L
POTABLE_LIMIT_MG_L = EPA_MCL_UG_L / 1000.0  # 0.015 mg/L
IRRIGATION_LIMIT_MG_L = 1.0         # safe for greenhouse crops

# ── Ion exchange parameters ─────────────────────────────────────────

RESIN_CAPACITY_EQ_L = 0.8           # equivalents per litre of resin
RESIN_BED_VOLUME_L = 50.0           # litres of resin per column
CLO4_EQUIVALENT_WEIGHT = PERCHLORATE_MOLAR_MASS  # monovalent
RESIN_MAX_CLO4_G = (RESIN_CAPACITY_EQ_L * RESIN_BED_VOLUME_L
                    * CLO4_EQUIVALENT_WEIGHT)  # ~3978 g = ~4.0 kg

# Thomas model kinetics
THOMAS_KT = 0.002                   # Thomas rate constant, L/(mg·h)
THOMAS_Q0 = RESIN_MAX_CLO4_G * 1000.0 / RESIN_BED_VOLUME_L  # mg/L capacity

# Flow rates
IX_FLOW_RATE_L_H = 20.0             # litres per hour through column
IX_HOURS_PER_SOL = 20.0             # operational hours per sol

# Regeneration
REGEN_BRINE_BED_VOLUMES = 4         # bed volumes of brine for regen
REGEN_NACL_FRACTION = 0.12          # 12% NaCl by weight
REGEN_EFFICIENCY = 0.90             # fraction of capacity restored
REGEN_ENERGY_KWH = 2.5              # energy for pumping + heating brine

# ── Bioreactor parameters ──────────────────────────────────────────

BIO_MU_MAX = 0.12                   # max specific growth rate, h⁻¹
BIO_KS_MG_L = 5.0                   # half-saturation constant, mg/L
BIO_YIELD = 0.35                    # biomass yield (g biomass / g ClO₄⁻)
BIO_VOLUME_L = 100.0                # reactor volume
BIO_INITIAL_BIOMASS_G = 50.0        # starting biomass (g)
BIO_MAX_BIOMASS_G = 500.0           # carrying capacity
BIO_DECAY_RATE = 0.005              # endogenous decay rate, h⁻¹
BIO_HOURS_PER_SOL = 24.65           # Mars sol in hours
BIO_TEMP_OPTIMAL_C = 30.0           # optimal temperature
BIO_TEMP_MIN_C = 5.0                # minimum for activity
BIO_TEMP_MAX_C = 45.0               # maximum before die-off

# Energy
IX_PUMP_KWH_PER_SOL = 1.5           # IX column pumping energy
BIO_HEATING_KWH_PER_SOL = 3.0       # bioreactor temperature control
BIO_MIXING_KWH_PER_SOL = 0.5        # stirring / aeration


# ── Data structures ─────────────────────────────────────────────────

@dataclass
class ScrubberState:
    """Full state of the perchlorate remediation system."""

    # Ion exchange column
    resin_clo4_g: float = 0.0           # ClO₄⁻ adsorbed on resin (g)
    resin_capacity_g: float = RESIN_MAX_CLO4_G
    bed_volumes_processed: int = 0
    regeneration_count: int = 0

    # Bioreactor
    biomass_g: float = BIO_INITIAL_BIOMASS_G
    bio_clo4_destroyed_g: float = 0.0

    # Water quality
    influent_clo4_mg_l: float = MARS_WATER_CLO4_MG_L
    effluent_clo4_mg_l: float = 0.0

    # Cumulative totals
    total_water_treated_l: float = 0.0
    total_clo4_removed_g: float = 0.0
    total_chloride_produced_g: float = 0.0
    total_o2_produced_g: float = 0.0
    total_energy_kwh: float = 0.0

    # Operational
    sols_running: int = 0
    alert: str = "nominal"

    def __post_init__(self) -> None:
        """Clamp state to physical bounds."""
        self.resin_clo4_g = max(0.0, min(self.resin_capacity_g,
                                         self.resin_clo4_g))
        self.biomass_g = max(0.0, min(BIO_MAX_BIOMASS_G, self.biomass_g))
        self.influent_clo4_mg_l = max(0.0, self.influent_clo4_mg_l)
        self.total_water_treated_l = max(0.0, self.total_water_treated_l)
        self.total_clo4_removed_g = max(0.0, self.total_clo4_removed_g)
        self.total_energy_kwh = max(0.0, self.total_energy_kwh)


@dataclass
class ScrubberTickResult:
    """Result of one sol of perchlorate remediation."""
    water_treated_l: float = 0.0
    ix_clo4_captured_g: float = 0.0
    bio_clo4_destroyed_g: float = 0.0
    total_clo4_removed_g: float = 0.0
    chloride_produced_g: float = 0.0
    o2_produced_g: float = 0.0
    effluent_clo4_mg_l: float = 0.0
    resin_saturation: float = 0.0
    biomass_g: float = 0.0
    energy_kwh: float = 0.0
    potable: bool = False
    irrigable: bool = False
    regenerated: bool = False
    alert: str = "nominal"


# ── Pure chemistry / physics functions ──────────────────────────────

def thomas_breakthrough(bed_volumes: int, influent_mg_l: float,
                        resin_clo4_g: float,
                        capacity_g: float) -> float:
    """Fraction of influent ClO₄⁻ passing through IX column (C/C₀).

    Uses the Thomas model adapted for bed-volume operation.
    Returns value in [0, 1].  0 = full capture, 1 = full breakthrough.
    """
    if capacity_g <= 0 or influent_mg_l <= 0:
        return 1.0

    saturation = resin_clo4_g / capacity_g
    # Sigmoid breakthrough: steep rise near saturation
    exponent = 20.0 * (saturation - 0.85)
    exponent = max(-50.0, min(50.0, exponent))  # prevent overflow
    ratio = 1.0 / (1.0 + math.exp(-exponent))
    return max(0.0, min(1.0, ratio))


def ix_removal_sol(flow_l_h: float, hours: float,
                   influent_mg_l: float,
                   breakthrough_frac: float) -> float:
    """Mass of ClO₄⁻ captured by IX column in one sol (grams).

    Parameters
    ----------
    flow_l_h : float
        Water flow rate through column (L/h).
    hours : float
        Operational hours this sol.
    influent_mg_l : float
        Incoming ClO₄⁻ concentration (mg/L).
    breakthrough_frac : float
        Fraction of ClO₄⁻ that passes through (0–1).
    """
    flow_l_h = max(0.0, flow_l_h)
    hours = max(0.0, hours)
    influent_mg_l = max(0.0, influent_mg_l)
    breakthrough_frac = max(0.0, min(1.0, breakthrough_frac))

    total_volume_l = flow_l_h * hours
    total_clo4_mg = total_volume_l * influent_mg_l
    captured_mg = total_clo4_mg * (1.0 - breakthrough_frac)
    return captured_mg / 1000.0  # grams


def ix_effluent_mg_l(influent_mg_l: float,
                     breakthrough_frac: float) -> float:
    """ClO₄⁻ concentration in IX column effluent (mg/L)."""
    return max(0.0, influent_mg_l * max(0.0, min(1.0, breakthrough_frac)))


def monod_rate(substrate_mg_l: float, mu_max: float = BIO_MU_MAX,
               ks: float = BIO_KS_MG_L) -> float:
    """Monod specific growth rate (h⁻¹).

    μ = μ_max · S / (Ks + S)
    """
    substrate_mg_l = max(0.0, substrate_mg_l)
    ks = max(0.001, ks)
    return mu_max * substrate_mg_l / (ks + substrate_mg_l)


def bio_temperature_factor(temp_c: float) -> float:
    """Temperature correction factor for bioreactor activity [0, 1].

    Gaussian-like curve centered on optimal temperature.
    """
    if temp_c < BIO_TEMP_MIN_C or temp_c > BIO_TEMP_MAX_C:
        return 0.0
    diff = temp_c - BIO_TEMP_OPTIMAL_C
    sigma = 10.0
    return math.exp(-(diff ** 2) / (2.0 * sigma ** 2))


def bio_destruction_sol(biomass_g: float, substrate_mg_l: float,
                        reactor_volume_l: float, hours: float,
                        temp_c: float = BIO_TEMP_OPTIMAL_C) -> Tuple[float, float]:
    """ClO₄⁻ destroyed by bioreactor in one sol.

    Returns
    -------
    tuple[float, float]
        (clo4_destroyed_g, new_biomass_g)
    """
    biomass_g = max(0.0, biomass_g)
    substrate_mg_l = max(0.0, substrate_mg_l)
    reactor_volume_l = max(0.01, reactor_volume_l)
    hours = max(0.0, hours)

    temp_factor = bio_temperature_factor(temp_c)
    if temp_factor < 0.01 or biomass_g < 0.1:
        return (0.0, max(0.0, biomass_g * (1.0 - BIO_DECAY_RATE * hours)))

    mu = monod_rate(substrate_mg_l) * temp_factor
    substrate_total_mg = substrate_mg_l * reactor_volume_l

    # Biomass-limited uptake rate: grams ClO₄⁻ consumed per hour
    uptake_rate_g_h = (mu / max(0.001, BIO_YIELD)) * biomass_g / 1000.0
    max_destruction_g = uptake_rate_g_h * hours
    available_g = substrate_total_mg / 1000.0
    destroyed_g = min(max_destruction_g, available_g)

    # Biomass growth (Monod) minus endogenous decay
    growth = biomass_g * mu * hours
    decay = biomass_g * BIO_DECAY_RATE * hours
    new_biomass = biomass_g + growth - decay
    new_biomass = max(0.0, min(BIO_MAX_BIOMASS_G, new_biomass))

    return (max(0.0, destroyed_g), new_biomass)


def stoichiometry_products(clo4_destroyed_g: float) -> Tuple[float, float]:
    """Chloride and O₂ produced from perchlorate destruction (grams).

    ClO₄⁻ → Cl⁻ + 2 O₂
    """
    clo4_destroyed_g = max(0.0, clo4_destroyed_g)
    chloride_g = clo4_destroyed_g * STOICH_CL_PER_CLO4
    o2_g = clo4_destroyed_g * STOICH_O2_PER_CLO4
    return (chloride_g, o2_g)


def resin_needs_regen(resin_clo4_g: float,
                      capacity_g: float) -> bool:
    """Check if IX resin needs regeneration (>85% saturated)."""
    if capacity_g <= 0:
        return True
    return (resin_clo4_g / capacity_g) > 0.85


def regenerate_resin(resin_clo4_g: float,
                     capacity_g: float) -> Tuple[float, float, float]:
    """Regenerate IX resin with brine wash.

    Returns
    -------
    tuple[float, float, float]
        (new_resin_clo4_g, clo4_recovered_g, capacity_after_g)
    """
    recovered_g = resin_clo4_g * REGEN_EFFICIENCY
    remaining_g = resin_clo4_g - recovered_g
    # Capacity degrades slightly with each regen (~2% loss)
    new_capacity = capacity_g * 0.98
    return (max(0.0, remaining_g), max(0.0, recovered_g), max(0.0, new_capacity))


def is_potable(clo4_mg_l: float) -> bool:
    """Check if water meets potable (drinking) standard."""
    return clo4_mg_l <= POTABLE_LIMIT_MG_L


def is_irrigable(clo4_mg_l: float) -> bool:
    """Check if water is safe for greenhouse irrigation."""
    return clo4_mg_l <= IRRIGATION_LIMIT_MG_L


def energy_per_sol(regenerating: bool = False) -> float:
    """Total energy consumption for one sol of operation (kWh)."""
    base = IX_PUMP_KWH_PER_SOL + BIO_HEATING_KWH_PER_SOL + BIO_MIXING_KWH_PER_SOL
    if regenerating:
        base += REGEN_ENERGY_KWH
    return base


def assess_alert(effluent_mg_l: float, resin_saturation: float,
                 biomass_g: float) -> str:
    """Determine system alert level."""
    if effluent_mg_l > 10.0 or biomass_g < 5.0:
        return "critical"
    if resin_saturation > 0.85 or effluent_mg_l > 1.0:
        return "warning"
    return "nominal"


# ── Tick function ───────────────────────────────────────────────────

def tick_scrubber(state: ScrubberState,
                  power_available_kwh: float = 50.0,
                  bio_temp_c: float = BIO_TEMP_OPTIMAL_C) -> Tuple[ScrubberState, ScrubberTickResult]:
    """Advance perchlorate scrubber by one sol.

    Parameters
    ----------
    state : ScrubberState
        Current system state.
    power_available_kwh : float
        Energy budget for this sol.
    bio_temp_c : float
        Bioreactor temperature (°C).

    Returns
    -------
    tuple[ScrubberState, ScrubberTickResult]
        Updated state and tick result.
    """
    result = ScrubberTickResult()
    energy_needed = energy_per_sol(False)

    # Power-limited: scale flow if insufficient energy
    power_scale = min(1.0, power_available_kwh / max(0.01, energy_needed))
    effective_hours = IX_HOURS_PER_SOL * power_scale

    # ── Stage 1: Ion exchange ───────────────────────────────────────
    bt_frac = thomas_breakthrough(
        state.bed_volumes_processed, state.influent_clo4_mg_l,
        state.resin_clo4_g, state.resin_capacity_g,
    )
    captured_g = ix_removal_sol(
        IX_FLOW_RATE_L_H, effective_hours,
        state.influent_clo4_mg_l, bt_frac,
    )
    # Don't exceed remaining resin capacity
    space = max(0.0, state.resin_capacity_g - state.resin_clo4_g)
    captured_g = min(captured_g, space)

    state.resin_clo4_g += captured_g
    ix_effluent = ix_effluent_mg_l(state.influent_clo4_mg_l, bt_frac)

    water_volume = IX_FLOW_RATE_L_H * effective_hours
    state.bed_volumes_processed += max(1, int(water_volume / RESIN_BED_VOLUME_L))

    result.ix_clo4_captured_g = captured_g
    result.water_treated_l = water_volume

    # ── Stage 2: Bioreactor polishing ───────────────────────────────
    destroyed_g, new_biomass = bio_destruction_sol(
        state.biomass_g, ix_effluent, BIO_VOLUME_L,
        BIO_HOURS_PER_SOL * power_scale, bio_temp_c,
    )
    state.biomass_g = new_biomass
    state.bio_clo4_destroyed_g += destroyed_g

    cl_g, o2_g = stoichiometry_products(destroyed_g)
    state.total_chloride_produced_g += cl_g
    state.total_o2_produced_g += o2_g

    result.bio_clo4_destroyed_g = destroyed_g
    result.chloride_produced_g = cl_g
    result.o2_produced_g = o2_g
    result.biomass_g = new_biomass

    # Final effluent: IX effluent minus what bioreactor destroyed
    if water_volume > 0:
        bio_removed_mg = destroyed_g * 1000.0
        effluent_total_mg = ix_effluent * water_volume - bio_removed_mg
        final_effluent = max(0.0, effluent_total_mg / water_volume)
    else:
        final_effluent = ix_effluent

    state.effluent_clo4_mg_l = final_effluent
    result.effluent_clo4_mg_l = final_effluent

    # ── Resin regeneration (if needed) ──────────────────────────────
    regen = False
    if resin_needs_regen(state.resin_clo4_g, state.resin_capacity_g):
        if power_available_kwh >= energy_per_sol(True):
            new_load, recovered, new_cap = regenerate_resin(
                state.resin_clo4_g, state.resin_capacity_g,
            )
            state.resin_clo4_g = new_load
            state.resin_capacity_g = new_cap
            state.regeneration_count += 1
            regen = True
            energy_needed = energy_per_sol(True)
    result.regenerated = regen

    # ── Totals ──────────────────────────────────────────────────────
    total_removed = captured_g + destroyed_g
    state.total_clo4_removed_g += total_removed
    state.total_water_treated_l += water_volume
    state.total_energy_kwh += energy_needed * power_scale

    result.total_clo4_removed_g = total_removed
    result.energy_kwh = energy_needed * power_scale
    result.resin_saturation = (state.resin_clo4_g / state.resin_capacity_g
                               if state.resin_capacity_g > 0 else 1.0)
    result.potable = is_potable(final_effluent)
    result.irrigable = is_irrigable(final_effluent)

    alert = assess_alert(final_effluent, result.resin_saturation,
                         state.biomass_g)
    result.alert = alert
    state.alert = alert
    state.sols_running += 1

    return state, result


# ── Factory ─────────────────────────────────────────────────────────

def create_scrubber(influent_mg_l: float = MARS_WATER_CLO4_MG_L,
                    biomass_g: float = BIO_INITIAL_BIOMASS_G) -> ScrubberState:
    """Create a perchlorate scrubber with given influent concentration."""
    return ScrubberState(
        influent_clo4_mg_l=max(0.0, influent_mg_l),
        biomass_g=max(0.0, min(BIO_MAX_BIOMASS_G, biomass_g)),
    )
