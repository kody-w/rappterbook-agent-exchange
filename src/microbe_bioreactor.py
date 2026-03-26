"""microbe_bioreactor.py -- Mars Cyanobacteria Photobioreactor.

The colony scrubs CO₂ chemically (zeolite + Sabatier) and cracks water
for O₂ (electrolysis).  Both cost kilowatts.  Cyanobacteria do both
for FREE — powered by light and fed by waste CO₂.  They are the oldest
engineers on Earth: 2.4 billion years ago they oxygenated an entire
planet.  Now they'll do it again on Mars.

This module models a closed photobioreactor (PBR) growing *Anabaena* sp.
PCC 7120 — a filamentous cyanobacterium chosen for Mars because it:
  - Fixes CO₂ via photosynthesis (6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂)
  - Fixes atmospheric N₂ (heterocyst cells), reducing fertiliser imports
  - Tolerates UV and desiccation better than most microbes
  - Produces biomass usable as greenhouse soil amendment
  - Grows in simple mineral medium (water + trace salts)

Real references: ESA MELiSSA loop, NASA CUBES (Center for Utilization
of Biological Engineering in Space), DLR PBR@LSR on ISS (2019).

Physics modelled
----------------
* **Monod growth kinetics** — μ = μ_max · S / (Ks + S), where S is
  dissolved CO₂ concentration.  Classic chemostat microbiology.

* **Light-limited growth** — μ_light = μ_max · I / (KI + I).
  Beer-Lambert light attenuation through culture: I(z) = I₀ · exp(-ε·X·z).
  Dense cultures self-shade.

* **Temperature dependence** — Arrhenius-type: rate peaks at 30°C,
  drops sharply above 40°C (thermal kill) and below 10°C (dormancy).

* **Photosynthesis stoichiometry** — 1 mol CO₂ consumed produces
  1 mol O₂.  Mass ratio: 32/44 = 0.727 kg O₂ per kg CO₂.

* **Biomass yield** — Y_x/co2 ≈ 0.5 g dry biomass per g CO₂ consumed.

* **pH dynamics** — CO₂ dissolution lowers pH; photosynthesis raises it.
  Optimal range 7.0-8.5.  Outside range, growth rate penalised.

* **Dilution & harvest** — Continuous-mode chemostat.  Dilution rate
  D = F/V.  If D > μ, culture washes out.  Harvest = D · X · V.

* **Contamination** — Risk increases with culture age.  Contamination
  event crashes productivity to 10% until reactor is cleaned.

* **LED lighting** — 50 W red/blue LEDs per m² of reactor face.
  PAR delivery: ~400 μmol/m²/s at full power.

Conservation laws
-----------------
- CO₂_consumed ≥ 0, O₂_produced ≥ 0
- O₂/CO₂ mass ratio = 32/44 (exact stoichiometry)
- biomass ≥ 0, biomass ≤ physical maximum for vessel volume
- pH in [0, 14]
- temperature ≥ 0 K
- light intensity ≥ 0
- growth rate ≥ 0
- dilution rate ≥ 0
- contamination probability in [0, 1]

One tick = one sol.  Mass in kg, volume in litres, temperature in °C.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Photosynthesis: 6CO₂ + 6H₂O  →  C₆H₁₂O₆ + 6O₂
CO2_MOLAR_MASS = 44.01e-3     # kg/mol
O2_MOLAR_MASS = 32.00e-3      # kg/mol
O2_PER_CO2_MASS = O2_MOLAR_MASS / CO2_MOLAR_MASS  # ~0.727

# Monod kinetics — Anabaena sp. PCC 7120
MU_MAX_PER_HOUR = 0.05         # max specific growth rate (1/h)
KS_CO2_MG_L = 50.0            # half-saturation for dissolved CO₂ (mg/L)
KI_LIGHT_UMOL = 150.0         # half-saturation for light (μmol/m²/s)

# Biomass yield
YIELD_BIOMASS_PER_CO2 = 0.50  # g dry biomass per g CO₂ consumed

# Light
BEER_LAMBERT_EPSILON = 0.02   # extinction coeff (L/(g·cm))
LED_POWER_W_PER_M2 = 50.0     # LED power per m² of reactor face
LED_PAR_UMOL = 400.0          # PAR at full LED power
LED_EFFICIENCY = 0.45          # electrical → PAR conversion

# Temperature (°C)
TEMP_OPT_C = 30.0
TEMP_MIN_C = 5.0
TEMP_MAX_C = 45.0
TEMP_KILL_C = 50.0

# pH
PH_OPT_LOW = 7.0
PH_OPT_HIGH = 8.5
PH_MIN = 5.0
PH_MAX = 11.0

# Reactor defaults
DEFAULT_VOLUME_L = 500.0       # 500 L tubular PBR
DEFAULT_DEPTH_CM = 10.0        # light path length
DEFAULT_FACE_AREA_M2 = 2.0    # illuminated face
DEFAULT_INITIAL_BIOMASS_G_L = 0.5
DEFAULT_CO2_FEED_G_L = 2.0    # dissolved CO₂ in feed
DEFAULT_DILUTION_RATE_PER_H = 0.01  # gentle continuous harvest
DEFAULT_TEMP_C = 25.0
DEFAULT_PH = 7.5

# Contamination
CONTAMINATION_BASE_PROB_PER_SOL = 0.002  # 0.2% per sol
CONTAMINATION_AGE_FACTOR = 0.0001        # increases with age
CONTAMINATION_PRODUCTIVITY_FACTOR = 0.10  # 90% loss when contaminated
CLEANUP_SOLS = 3                          # sols to clean and restart

# Time
HOURS_PER_SOL = 24.66
LIGHT_HOURS_PER_SOL = 16.0  # LED photoperiod (longer than Earth for growth)

# Physical limits
MAX_BIOMASS_G_L = 15.0  # dense culture ceiling


# ---------------------------------------------------------------------------
# Pure physics functions
# ---------------------------------------------------------------------------

def monod(substrate_mg_l: float, mu_max: float = MU_MAX_PER_HOUR,
          ks: float = KS_CO2_MG_L) -> float:
    """Monod specific growth rate (1/h) as function of substrate."""
    substrate_mg_l = max(0.0, substrate_mg_l)
    if substrate_mg_l + ks <= 0.0:
        return 0.0
    return mu_max * substrate_mg_l / (ks + substrate_mg_l)


def light_factor(par_umol: float, ki: float = KI_LIGHT_UMOL) -> float:
    """Light limitation factor [0, 1] — Monod-type for light."""
    par_umol = max(0.0, par_umol)
    if par_umol + ki <= 0.0:
        return 0.0
    return par_umol / (ki + par_umol)


def beer_lambert(incident_par: float, biomass_g_l: float,
                 depth_cm: float, epsilon: float = BEER_LAMBERT_EPSILON) -> float:
    """Average PAR through culture (μmol/m²/s) via Beer-Lambert."""
    incident_par = max(0.0, incident_par)
    biomass_g_l = max(0.0, biomass_g_l)
    depth_cm = max(0.01, depth_cm)
    optical_depth = epsilon * biomass_g_l * depth_cm
    if optical_depth > 20.0:
        return 0.0
    if optical_depth < 1e-6:
        return incident_par
    return incident_par * (1.0 - math.exp(-optical_depth)) / optical_depth


def temperature_factor(temp_c: float) -> float:
    """Temperature growth modifier [0, 1].

    Cardinal temperature model: ramp up from TEMP_MIN to TEMP_OPT,
    ramp down from TEMP_OPT to TEMP_MAX, zero outside bounds.
    """
    if temp_c <= TEMP_MIN_C or temp_c >= TEMP_KILL_C:
        return 0.0
    if temp_c <= TEMP_OPT_C:
        return (temp_c - TEMP_MIN_C) / (TEMP_OPT_C - TEMP_MIN_C)
    # Above optimum — decline toward kill temperature
    return max(0.0, (TEMP_KILL_C - temp_c) / (TEMP_KILL_C - TEMP_OPT_C))


def ph_factor(ph: float) -> float:
    """pH growth modifier [0, 1].  Optimal in [PH_OPT_LOW, PH_OPT_HIGH]."""
    if ph < PH_MIN or ph > PH_MAX:
        return 0.0
    if PH_OPT_LOW <= ph <= PH_OPT_HIGH:
        return 1.0
    if ph < PH_OPT_LOW:
        return (ph - PH_MIN) / (PH_OPT_LOW - PH_MIN)
    return (PH_MAX - ph) / (PH_MAX - PH_OPT_HIGH)


def effective_growth_rate(co2_mg_l: float, avg_par: float,
                          temp_c: float, ph: float) -> float:
    """Net specific growth rate (1/h) combining all limiting factors."""
    mu = monod(co2_mg_l)
    lf = light_factor(avg_par)
    tf = temperature_factor(temp_c)
    pf = ph_factor(ph)
    return mu * lf * tf * pf


def co2_consumed_kg(biomass_produced_kg: float) -> float:
    """CO₂ consumed to produce given biomass (kg)."""
    biomass_produced_kg = max(0.0, biomass_produced_kg)
    if YIELD_BIOMASS_PER_CO2 <= 0.0:
        return 0.0
    return biomass_produced_kg / YIELD_BIOMASS_PER_CO2


def o2_produced_kg(co2_consumed_kg_val: float) -> float:
    """O₂ produced from given CO₂ consumption (kg) — stoichiometric."""
    return max(0.0, co2_consumed_kg_val * O2_PER_CO2_MASS)


def led_power_kwh_per_sol(face_area_m2: float,
                          power_w_per_m2: float = LED_POWER_W_PER_M2,
                          photoperiod_h: float = LIGHT_HOURS_PER_SOL) -> float:
    """LED electrical energy consumption per sol (kWh)."""
    face_area_m2 = max(0.0, face_area_m2)
    power_w_per_m2 = max(0.0, power_w_per_m2)
    photoperiod_h = max(0.0, min(photoperiod_h, HOURS_PER_SOL))
    return face_area_m2 * power_w_per_m2 * photoperiod_h / 1000.0


def harvest_biomass_g(dilution_rate_h: float, biomass_g_l: float,
                      volume_l: float, hours: float) -> float:
    """Biomass removed by continuous dilution (g)."""
    dilution_rate_h = max(0.0, dilution_rate_h)
    biomass_g_l = max(0.0, biomass_g_l)
    volume_l = max(0.0, volume_l)
    hours = max(0.0, hours)
    return dilution_rate_h * biomass_g_l * volume_l * hours


def contamination_probability(culture_age_sols: int,
                              base_prob: float = CONTAMINATION_BASE_PROB_PER_SOL,
                              age_factor: float = CONTAMINATION_AGE_FACTOR) -> float:
    """Probability of contamination event this sol."""
    culture_age_sols = max(0, culture_age_sols)
    return min(1.0, base_prob + age_factor * culture_age_sols)


def ph_shift_from_photosynthesis(co2_consumed_g: float,
                                  volume_l: float,
                                  current_ph: float) -> float:
    """pH rise from CO₂ removal by photosynthesis (simplified)."""
    if volume_l <= 0.0:
        return current_ph
    # Removing CO₂ raises pH — ~0.1 pH per g/L CO₂ removed (linearised)
    delta = 0.1 * co2_consumed_g / volume_l
    return min(14.0, current_ph + delta)


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BioreactorState:
    """Full state of the photobioreactor."""
    sol: int = 0
    culture_age_sols: int = 0
    volume_l: float = DEFAULT_VOLUME_L
    depth_cm: float = DEFAULT_DEPTH_CM
    face_area_m2: float = DEFAULT_FACE_AREA_M2
    biomass_g_l: float = DEFAULT_INITIAL_BIOMASS_G_L
    co2_feed_g_l: float = DEFAULT_CO2_FEED_G_L
    dilution_rate_h: float = DEFAULT_DILUTION_RATE_PER_H
    temp_c: float = DEFAULT_TEMP_C
    ph: float = DEFAULT_PH
    led_on: bool = True
    contaminated: bool = False
    contamination_cleanup_remaining: int = 0
    total_biomass_harvested_kg: float = 0.0
    total_co2_consumed_kg: float = 0.0
    total_o2_produced_kg: float = 0.0
    total_led_energy_kwh: float = 0.0
    generation: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sol": self.sol,
            "culture_age_sols": self.culture_age_sols,
            "volume_l": round(self.volume_l, 2),
            "depth_cm": round(self.depth_cm, 2),
            "face_area_m2": round(self.face_area_m2, 2),
            "biomass_g_l": round(self.biomass_g_l, 4),
            "co2_feed_g_l": round(self.co2_feed_g_l, 2),
            "dilution_rate_h": round(self.dilution_rate_h, 4),
            "temp_c": round(self.temp_c, 2),
            "ph": round(self.ph, 3),
            "led_on": self.led_on,
            "contaminated": self.contaminated,
            "contamination_cleanup_remaining": self.contamination_cleanup_remaining,
            "total_biomass_harvested_kg": round(self.total_biomass_harvested_kg, 4),
            "total_co2_consumed_kg": round(self.total_co2_consumed_kg, 4),
            "total_o2_produced_kg": round(self.total_o2_produced_kg, 4),
            "total_led_energy_kwh": round(self.total_led_energy_kwh, 4),
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BioreactorState:
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class TickResult:
    """Result of a single sol tick."""
    sol: int = 0
    growth_rate_per_h: float = 0.0
    biomass_g_l: float = 0.0
    biomass_produced_g: float = 0.0
    biomass_harvested_g: float = 0.0
    co2_consumed_g: float = 0.0
    o2_produced_g: float = 0.0
    led_energy_kwh: float = 0.0
    avg_par_umol: float = 0.0
    temp_factor: float = 0.0
    ph_factor_val: float = 0.0
    light_factor_val: float = 0.0
    ph: float = 0.0
    contaminated: bool = False
    contamination_event: bool = False
    washout: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sol": self.sol,
            "growth_rate_per_h": round(self.growth_rate_per_h, 6),
            "biomass_g_l": round(self.biomass_g_l, 4),
            "biomass_produced_g": round(self.biomass_produced_g, 2),
            "biomass_harvested_g": round(self.biomass_harvested_g, 2),
            "co2_consumed_g": round(self.co2_consumed_g, 2),
            "o2_produced_g": round(self.o2_produced_g, 2),
            "led_energy_kwh": round(self.led_energy_kwh, 4),
            "avg_par_umol": round(self.avg_par_umol, 2),
            "temp_factor": round(self.temp_factor, 4),
            "ph_factor_val": round(self.ph_factor_val, 4),
            "light_factor_val": round(self.light_factor_val, 4),
            "ph": round(self.ph, 3),
            "contaminated": self.contaminated,
            "contamination_event": self.contamination_event,
            "washout": self.washout,
        }


# ---------------------------------------------------------------------------
# Factory & tick
# ---------------------------------------------------------------------------

def create_bioreactor(**kwargs: Any) -> BioreactorState:
    """Create a new bioreactor with optional overrides."""
    return BioreactorState(**{k: v for k, v in kwargs.items()
                              if k in BioreactorState.__dataclass_fields__})


def tick(state: BioreactorState, rng_roll: float = -1.0) -> TickResult:
    """Advance the bioreactor by one sol.

    Parameters
    ----------
    state : BioreactorState
        Mutable reactor state — modified in place.
    rng_roll : float
        Random number in [0, 1) for contamination check.
        Pass -1.0 to skip contamination (deterministic mode).

    Returns
    -------
    TickResult with this sol's metrics.
    """
    state.sol += 1
    state.culture_age_sols += 1
    result = TickResult(sol=state.sol)

    # --- Contamination cleanup ---
    if state.contamination_cleanup_remaining > 0:
        state.contamination_cleanup_remaining -= 1
        if state.contamination_cleanup_remaining == 0:
            state.contaminated = False
            state.biomass_g_l = DEFAULT_INITIAL_BIOMASS_G_L
            state.ph = DEFAULT_PH
            state.culture_age_sols = 0
            state.generation += 1
        result.contaminated = state.contaminated
        result.biomass_g_l = state.biomass_g_l
        result.ph = state.ph
        return result

    # --- Contamination check ---
    if 0.0 <= rng_roll < 1.0:
        prob = contamination_probability(state.culture_age_sols)
        if rng_roll < prob:
            state.contaminated = True
            state.contamination_cleanup_remaining = CLEANUP_SOLS
            result.contamination_event = True
            result.contaminated = True
            result.biomass_g_l = state.biomass_g_l
            result.ph = state.ph
            return result

    # --- LED lighting ---
    if state.led_on:
        incident_par = LED_PAR_UMOL
        energy_kwh = led_power_kwh_per_sol(state.face_area_m2)
    else:
        incident_par = 0.0
        energy_kwh = 0.0

    result.led_energy_kwh = energy_kwh
    state.total_led_energy_kwh += energy_kwh

    # --- Average PAR (Beer-Lambert self-shading) ---
    avg_par = beer_lambert(incident_par, state.biomass_g_l, state.depth_cm)
    result.avg_par_umol = avg_par

    # --- Growth limiting factors ---
    tf = temperature_factor(state.temp_c)
    pf = ph_factor(state.ph)
    lf = light_factor(avg_par)
    result.temp_factor = tf
    result.ph_factor_val = pf
    result.light_factor_val = lf

    # --- Effective growth rate ---
    co2_mg_l = state.co2_feed_g_l * 1000.0  # g/L → mg/L
    mu = effective_growth_rate(co2_mg_l, avg_par, state.temp_c, state.ph)

    # Contamination penalty (if contaminated but not in cleanup)
    if state.contaminated:
        mu *= CONTAMINATION_PRODUCTIVITY_FACTOR

    result.growth_rate_per_h = mu

    # --- Biomass growth (Euler integration over sol) ---
    growth_hours = LIGHT_HOURS_PER_SOL  # growth only during lit period
    biomass_before = state.biomass_g_l * state.volume_l  # total g
    delta_biomass_g = mu * state.biomass_g_l * state.volume_l * growth_hours
    delta_biomass_g = max(0.0, delta_biomass_g)

    # --- Harvest (continuous dilution over full sol) ---
    harvested_g = harvest_biomass_g(state.dilution_rate_h,
                                     state.biomass_g_l,
                                     state.volume_l,
                                     HOURS_PER_SOL)

    # --- Net biomass change ---
    new_total_g = biomass_before + delta_biomass_g - harvested_g

    # Washout check
    if new_total_g < 0.01 * state.volume_l:  # < 0.01 g/L
        result.washout = True
        new_total_g = max(0.0, new_total_g)

    # Density ceiling
    new_biomass_g_l = new_total_g / state.volume_l
    new_biomass_g_l = min(new_biomass_g_l, MAX_BIOMASS_G_L)
    new_biomass_g_l = max(0.0, new_biomass_g_l)

    state.biomass_g_l = new_biomass_g_l
    result.biomass_g_l = new_biomass_g_l
    result.biomass_produced_g = delta_biomass_g
    result.biomass_harvested_g = harvested_g

    # --- CO₂ consumed and O₂ produced (stoichiometric) ---
    co2_g = co2_consumed_kg(delta_biomass_g / 1000.0) * 1000.0  # back to g
    o2_g = o2_produced_kg(co2_g / 1000.0) * 1000.0
    result.co2_consumed_g = co2_g
    result.o2_produced_g = o2_g
    state.total_co2_consumed_kg += co2_g / 1000.0
    state.total_o2_produced_kg += o2_g / 1000.0
    state.total_biomass_harvested_kg += harvested_g / 1000.0

    # --- pH shift from photosynthesis ---
    state.ph = ph_shift_from_photosynthesis(co2_g, state.volume_l, state.ph)
    result.ph = state.ph

    return result


def run_simulation(state: BioreactorState, sols: int = 100,
                   rng_roll: float = -1.0) -> list[TickResult]:
    """Run the bioreactor for multiple sols (deterministic by default)."""
    return [tick(state, rng_roll) for _ in range(sols)]
