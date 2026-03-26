"""oxygen_farm.py — Mars Colony Algae Photobioreactor

Closed-loop biological oxygen production using microalgae cultures.
Supplements the electrochemical O₂ generator with a living, self-
replicating system that also produces edible biomass and scrubs CO₂.

Physics
-------
* **Photosynthesis stoichiometry**:
    6 CO₂ + 6 H₂O  →  C₆H₁₂O₆ + 6 O₂
  By mass: 264 g CO₂ + 108 g H₂O → 180 g glucose + 192 g O₂.
  Molar ratio: 1 mol CO₂ consumed produces (32/44) = 0.727 kg O₂/kg CO₂.
* **Chlorella vulgaris** — workhorse species.  Doubling time 8–24 h
  under optimal light (200–400 µmol/m²/s PAR), temperature 25–30 °C,
  pH 6.5–7.5.  Biomass productivity: 0.5–2.0 g/L/day in tubular PBR.
* **Light-limited growth** — follows the Monod model:
    µ = µ_max · I / (K_I + I)
  where µ is specific growth rate, I is irradiance, K_I is the half-
  saturation constant (~100 µmol/m²/s for Chlorella).
* **Temperature dependence** — Arrhenius-like.  Growth peaks at 28 °C,
  drops sharply above 35 °C (protein denaturation) and below 15 °C.
* **pH buffering** — CO₂ dissolution acidifies the medium.  Automated
  NaOH dosing keeps pH in the 6.5–7.5 sweet spot.  Outside this range,
  growth drops to near zero.
* **Harvesting** — centrifugal separation when culture density exceeds
  ~5 g/L.  Harvested biomass is 50–60 % protein, edible as supplement.
* **O₂ yield** — 1 g dry algal biomass produces ~1.8 g O₂ during growth
  (accounting for respiration losses of ~15 %).

Reference systems:
  - ESA MELiSSA (Micro-Ecological Life Support System Alternative):
    Spirulina arthroplastis compartment produces 0.84 kg O₂/m³/day.
  - NASA BLSS (Bioregenerative Life Support System): algae reactors
    as tertiary O₂ backup.
  - ISS photobioreactor experiment (PBR@ACLS, 2019): Chlorella in 6 L
    reactor, validated CO₂ → O₂ conversion in microgravity.

One tick = one sol.  Volume in liters, mass in grams, energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── Physical constants ──────────────────────────────────────────────

# Photosynthesis mass ratios (from stoichiometry)
O2_PER_CO2_KG = 192.0 / 264.0       # 0.727 kg O₂ per kg CO₂ consumed
CO2_PER_BIOMASS_G = 1.83             # g CO₂ consumed per g dry biomass grown
O2_PER_BIOMASS_G = 1.8               # g O₂ net produced per g dry biomass
WATER_PER_BIOMASS_G = 1.2            # g H₂O consumed per g dry biomass

# Chlorella vulgaris growth kinetics
MU_MAX_PER_SOL = 0.7                 # maximum specific growth rate (1/sol)
K_LIGHT = 100.0                      # half-saturation irradiance (µmol/m²/s)
OPTIMAL_TEMP_C = 28.0                # optimal growth temperature (°C)
TEMP_RANGE_C = 10.0                  # half-width of temperature bell curve
MIN_GROWTH_TEMP_C = 5.0              # below this, zero growth
MAX_GROWTH_TEMP_C = 42.0             # above this, culture death begins
OPTIMAL_PH = 7.0                     # optimal pH for Chlorella
PH_RANGE = 1.0                       # half-width of pH bell curve

# Culture parameters
MAX_DENSITY_G_L = 8.0                # maximum sustainable culture density
HARVEST_THRESHOLD_G_L = 5.0          # trigger centrifugal harvest
HARVEST_FRACTION = 0.6               # fraction removed during harvest
RESPIRATION_LOSS = 0.15              # 15% of gross O₂ lost to algal respiration
BIOMASS_PROTEIN_FRACTION = 0.55      # 55% protein (edible)

# Reactor defaults
DEFAULT_VOLUME_L = 500.0             # total culture volume (liters)
DEFAULT_IRRADIANCE = 300.0           # µmol/m²/s PAR from LEDs
DEFAULT_TEMP_C = 25.0                # habitat-regulated temperature
DEFAULT_PH = 7.0                     # buffered pH
DEFAULT_INOCULUM_G_L = 1.0          # starting culture density

# Energy
LED_POWER_KWH_PER_L_SOL = 0.012     # LED lighting per liter per sol
PUMP_POWER_KWH_PER_L_SOL = 0.003    # circulation pumps per liter per sol
HARVEST_ENERGY_KWH_PER_G = 0.001    # centrifuge energy per gram harvested
CO2_INJECTION_KWH_PER_G = 0.0005    # CO₂ sparging energy per gram
HOURS_PER_SOL = 24.66                # Mars sol in hours


# ── Pure physics functions ──────────────────────────────────────────

def monod_growth_rate(irradiance: float, mu_max: float = MU_MAX_PER_SOL,
                      k_light: float = K_LIGHT) -> float:
    """Monod light-limited specific growth rate (1/sol).

    µ = µ_max · I / (K_I + I)

    Returns 0 for non-positive irradiance.
    """
    if irradiance <= 0.0 or mu_max <= 0.0:
        return 0.0
    return mu_max * irradiance / (k_light + irradiance)


def temperature_factor(temp_c: float) -> float:
    """Gaussian temperature response, peak at OPTIMAL_TEMP_C.

    Returns value in [0, 1].  Zero outside [MIN_GROWTH_TEMP_C, MAX_GROWTH_TEMP_C].
    """
    if temp_c < MIN_GROWTH_TEMP_C or temp_c > MAX_GROWTH_TEMP_C:
        return 0.0
    delta = temp_c - OPTIMAL_TEMP_C
    return math.exp(-(delta ** 2) / (2.0 * TEMP_RANGE_C ** 2))


def ph_factor(ph: float) -> float:
    """Gaussian pH response, peak at OPTIMAL_PH.

    Returns value in [0, 1].  Clamped to [0, 14].
    """
    ph = max(0.0, min(14.0, ph))
    delta = ph - OPTIMAL_PH
    return math.exp(-(delta ** 2) / (2.0 * PH_RANGE ** 2))


def density_inhibition(density_g_l: float,
                       max_density: float = MAX_DENSITY_G_L) -> float:
    """Logistic self-shading / nutrient competition factor.

    Returns 1.0 at low density, approaches 0.0 at max_density.
    Factor = 1 - (density / max_density)², clamped to [0, 1].
    """
    if max_density <= 0.0 or density_g_l <= 0.0:
        return 1.0 if max_density > 0.0 else 0.0
    ratio = min(density_g_l / max_density, 1.0)
    return max(0.0, 1.0 - ratio * ratio)


def biomass_growth_g(density_g_l: float, volume_l: float,
                     growth_rate: float, temp_f: float,
                     ph_f: float, inhibition: float) -> float:
    """Net biomass produced in one sol (grams).

    ΔM = density · volume · µ · f_temp · f_pH · f_inhibition
    """
    if volume_l <= 0.0 or growth_rate <= 0.0:
        return 0.0
    raw = density_g_l * volume_l * growth_rate * temp_f * ph_f * inhibition
    return max(0.0, raw)


def o2_produced_g(biomass_grown_g: float) -> float:
    """Net O₂ produced from biomass growth (grams).

    Gross O₂ minus respiration losses.
    """
    gross = biomass_grown_g * O2_PER_BIOMASS_G
    return max(0.0, gross * (1.0 - RESPIRATION_LOSS))


def co2_consumed_g(biomass_grown_g: float) -> float:
    """CO₂ consumed during photosynthesis (grams)."""
    return max(0.0, biomass_grown_g * CO2_PER_BIOMASS_G)


def water_consumed_g(biomass_grown_g: float) -> float:
    """Water consumed during photosynthesis (grams ≈ mL)."""
    return max(0.0, biomass_grown_g * WATER_PER_BIOMASS_G)


def should_harvest(density_g_l: float,
                   threshold: float = HARVEST_THRESHOLD_G_L) -> bool:
    """True when culture density exceeds harvest threshold."""
    return density_g_l >= threshold


def harvest_biomass(density_g_l: float, volume_l: float,
                    fraction: float = HARVEST_FRACTION) -> tuple[float, float]:
    """Harvest a fraction of the culture.

    Returns (new_density_g_l, harvested_mass_g).
    """
    fraction = max(0.0, min(1.0, fraction))
    harvested_g = density_g_l * volume_l * fraction
    new_density = density_g_l * (1.0 - fraction)
    return new_density, max(0.0, harvested_g)


def energy_per_sol_kwh(volume_l: float, biomass_grown_g: float,
                       harvested_g: float) -> float:
    """Total energy consumed in one sol (kWh).

    LED lighting + pumps + CO₂ injection + harvesting.
    """
    led = volume_l * LED_POWER_KWH_PER_L_SOL
    pump = volume_l * PUMP_POWER_KWH_PER_L_SOL
    co2_energy = co2_consumed_g(biomass_grown_g) * CO2_INJECTION_KWH_PER_G
    harvest_energy = harvested_g * HARVEST_ENERGY_KWH_PER_G
    return max(0.0, led + pump + co2_energy + harvest_energy)


def culture_health(temp_f: float, ph_f: float,
                   density_g_l: float) -> str:
    """Qualitative culture health assessment.

    Returns one of: 'thriving', 'stressed', 'critical', 'dead'.
    """
    if density_g_l <= 0.01:
        return "dead"
    combined = temp_f * ph_f
    if combined >= 0.7:
        return "thriving"
    elif combined >= 0.3:
        return "stressed"
    else:
        return "critical"


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class ReactorState:
    """State of one algae photobioreactor.

    All fields are clamped to physical bounds in __post_init__.
    """
    volume_l: float = DEFAULT_VOLUME_L
    density_g_l: float = DEFAULT_INOCULUM_G_L
    irradiance: float = DEFAULT_IRRADIANCE
    temperature_c: float = DEFAULT_TEMP_C
    ph: float = DEFAULT_PH
    sol: int = 0
    total_o2_produced_g: float = 0.0
    total_co2_consumed_g: float = 0.0
    total_water_consumed_g: float = 0.0
    total_biomass_harvested_g: float = 0.0
    total_energy_kwh: float = 0.0
    harvests_performed: int = 0
    culture_status: str = "thriving"

    def __post_init__(self) -> None:
        self.volume_l = max(0.0, self.volume_l)
        self.density_g_l = max(0.0, min(MAX_DENSITY_G_L, self.density_g_l))
        self.irradiance = max(0.0, self.irradiance)
        self.temperature_c = max(-40.0, min(60.0, self.temperature_c))
        self.ph = max(0.0, min(14.0, self.ph))
        self.sol = max(0, self.sol)
        self.total_o2_produced_g = max(0.0, self.total_o2_produced_g)
        self.total_co2_consumed_g = max(0.0, self.total_co2_consumed_g)
        self.total_water_consumed_g = max(0.0, self.total_water_consumed_g)
        self.total_biomass_harvested_g = max(0.0, self.total_biomass_harvested_g)
        self.total_energy_kwh = max(0.0, self.total_energy_kwh)
        self.harvests_performed = max(0, self.harvests_performed)


@dataclass
class SolRecord:
    """Per-sol telemetry record."""
    sol: int = 0
    biomass_grown_g: float = 0.0
    o2_produced_g: float = 0.0
    co2_consumed_g: float = 0.0
    water_consumed_g: float = 0.0
    energy_kwh: float = 0.0
    density_g_l: float = 0.0
    harvested_g: float = 0.0
    growth_rate: float = 0.0
    temp_factor: float = 0.0
    ph_factor: float = 0.0
    culture_status: str = "thriving"


# ── Tick function ───────────────────────────────────────────────────

def tick_oxygen_farm(state: ReactorState,
                     available_power_kwh: float = 100.0) -> tuple[ReactorState, SolRecord]:
    """Advance the photobioreactor by one sol.

    Steps:
      1. Compute growth factors (light, temperature, pH, density).
      2. Calculate biomass growth for this sol.
      3. Compute O₂, CO₂, H₂O exchange.
      4. Check harvest trigger and execute if needed.
      5. Compute energy consumption (cap by available power).
      6. Update cumulative state.

    Returns (new_state, sol_record).
    """
    # Step 1: growth factors
    mu = monod_growth_rate(state.irradiance)
    t_f = temperature_factor(state.temperature_c)
    p_f = ph_factor(state.ph)
    inhib = density_inhibition(state.density_g_l)

    # Step 2: biomass growth
    grown_g = biomass_growth_g(state.density_g_l, state.volume_l,
                               mu, t_f, p_f, inhib)

    # Step 3: gas and water exchange
    o2_g = o2_produced_g(grown_g)
    co2_g = co2_consumed_g(grown_g)
    h2o_g = water_consumed_g(grown_g)

    # Update density from growth
    new_density = state.density_g_l
    if state.volume_l > 0:
        new_density += grown_g / state.volume_l

    # Step 4: harvest check
    harvested_g = 0.0
    harvests = 0
    if should_harvest(new_density):
        new_density, harvested_g = harvest_biomass(new_density, state.volume_l)
        harvests = 1

    # Step 5: energy (cap by available power)
    energy = energy_per_sol_kwh(state.volume_l, grown_g, harvested_g)
    if available_power_kwh <= 0:
        grown_g = 0.0
        o2_g = 0.0
        co2_g = 0.0
        h2o_g = 0.0
        harvested_g = 0.0
        harvests = 0
        new_density = state.density_g_l
        energy = 0.0
    elif energy > available_power_kwh:
        # Scale growth proportionally to available power
        scale = available_power_kwh / energy
        grown_g *= scale
        o2_g = o2_produced_g(grown_g)
        co2_g = co2_consumed_g(grown_g)
        h2o_g = water_consumed_g(grown_g)
        if state.volume_l > 0:
            new_density = state.density_g_l + grown_g / state.volume_l
        if should_harvest(new_density):
            new_density, harvested_g = harvest_biomass(new_density,
                                                       state.volume_l)
            harvests = 1
        else:
            harvested_g = 0.0
            harvests = 0
        energy = available_power_kwh

    # Clamp density
    new_density = max(0.0, min(MAX_DENSITY_G_L, new_density))

    # Step 6: culture health
    status = culture_health(t_f, p_f, new_density)

    new_state = ReactorState(
        volume_l=state.volume_l,
        density_g_l=new_density,
        irradiance=state.irradiance,
        temperature_c=state.temperature_c,
        ph=state.ph,
        sol=state.sol + 1,
        total_o2_produced_g=state.total_o2_produced_g + o2_g,
        total_co2_consumed_g=state.total_co2_consumed_g + co2_g,
        total_water_consumed_g=state.total_water_consumed_g + h2o_g,
        total_biomass_harvested_g=state.total_biomass_harvested_g + harvested_g,
        total_energy_kwh=state.total_energy_kwh + energy,
        harvests_performed=state.harvests_performed + harvests,
        culture_status=status,
    )

    record = SolRecord(
        sol=state.sol + 1,
        biomass_grown_g=grown_g,
        o2_produced_g=o2_g,
        co2_consumed_g=co2_g,
        water_consumed_g=h2o_g,
        energy_kwh=energy,
        density_g_l=new_density,
        harvested_g=harvested_g,
        growth_rate=mu * t_f * p_f * inhib,
        temp_factor=t_f,
        ph_factor=p_f,
        culture_status=status,
    )

    return new_state, record


# ── Factory + runner ────────────────────────────────────────────────

def create_oxygen_farm(volume_l: float = DEFAULT_VOLUME_L,
                       inoculum_g_l: float = DEFAULT_INOCULUM_G_L,
                       irradiance: float = DEFAULT_IRRADIANCE,
                       temperature_c: float = DEFAULT_TEMP_C,
                       ph: float = DEFAULT_PH) -> ReactorState:
    """Create a new photobioreactor with given parameters."""
    return ReactorState(
        volume_l=volume_l,
        density_g_l=inoculum_g_l,
        irradiance=irradiance,
        temperature_c=temperature_c,
        ph=ph,
    )


def run_oxygen_farm(state: ReactorState, sols: int,
                    power_per_sol: float = 100.0) -> tuple[ReactorState, list[SolRecord]]:
    """Run the reactor for N sols.  Returns (final_state, records)."""
    records: list[SolRecord] = []
    for _ in range(max(0, sols)):
        state, rec = tick_oxygen_farm(state, power_per_sol)
        records.append(rec)
    return state, records
