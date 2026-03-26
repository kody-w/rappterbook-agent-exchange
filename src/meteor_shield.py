"""
meteor_shield.py — Micrometeoroid & Orbital Debris (MMOD) protection for Mars habitats.

Mars has ~0.6% of Earth's atmospheric density.  Meteoroids that would
burn up in Earth's atmosphere survive to the Martian surface intact.
The colony needs multi-layer Whipple shields on every pressurised module.

Physics modelled
----------------
* **Meteoroid flux**: Modified Grün (1985) interplanetary flux model
  scaled for Mars heliocentric distance (1.524 AU).  Flux decreases
  with distance² from the Sun but Mars's proximity to the asteroid
  belt adds a 1.5× correction factor (Divine, 1993).
* **Impact velocity**: Mars-crossing meteoroids arrive at 5–20 km/s.
  Mean ~12 km/s (lower than LEO debris at 10–15 km/s).
* **Whipple shield**: Thin aluminium bumper + standoff gap + structural
  backwall.  Impact on the bumper fragments the projectile into a debris
  cloud, spreading kinetic energy over a larger backwall area.
* **Ballistic limit**: Modified Cour-Palais equation for critical
  particle diameter d_crit that just penetrates the shield:
    d_crit = K · (t_b^α · S^β · t_w^γ) / (ρ_p^δ · V^ε · cos(θ)^ζ)
  where t_b = bumper thickness, S = standoff, t_w = wall thickness,
  ρ_p = projectile density, V = velocity, θ = impact angle.
* **Cratering**: Non-penetrating impacts create craters in the bumper.
  Depth from the Holsapple (1993) scaling law:
    depth = k · (ρ_p/ρ_t)^0.33 · d · (V/V_ref)^0.67
* **Shield degradation**: Cumulative crater damage reduces effective
  bumper thickness over time.  Tracked as fraction of original area
  that remains undamaged.
* **Probability**: Poisson process — P(≥1 hit) = 1 - exp(-flux·A·t).

Physical references:
  - Grün et al. (1985): interplanetary meteoroid flux model
  - Divine (1993): Mars meteoroid environment assessment
  - Cour-Palais (1987): Whipple shield ballistic limit equations
  - Christiansen (2009): NASA Meteoroid/Debris Shield Design Guide
  - MMOD Risk: ISS probability of no penetration > 95% over 10 years
  - Mars atmosphere: ~610 Pa surface pressure (0.6% of Earth)
  - Meteoroid density: 1000–3500 kg/m³ (cometary ice to chondritic rock)
  - Shield mass budget: ~5 kg/m² for ISS-grade protection

One tick = one sol.  Distances in metres, velocities in km/s, masses in kg.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Heliocentric distance scaling
AU_MARS = 1.524                       # Mars semi-major axis (AU)
FLUX_HELIO_EXPONENT = -2.0            # flux ∝ r^-2 (inverse square)
ASTEROID_BELT_FACTOR = 1.5            # proximity to belt correction (Divine 1993)

# Grün (1985) model coefficients for cumulative flux (particles/m²/year)
# N(>m) at 1 AU.  We evaluate at a few reference masses.
# Simplified power-law: log10(F) ≈ -14.37 - 1.213*log10(m) for m < 1g
GRUN_SLOPE = -1.213                   # mass exponent
GRUN_INTERCEPT = -14.37               # log10(flux) at log10(m)=0 (1 kg)
GRUN_MIN_MASS_KG = 1e-12             # smallest tracked particle (1 picogram)
GRUN_MAX_MASS_KG = 1e-1              # largest common impactor (100 g)

# Impact velocity at Mars
IMPACT_VELOCITY_MEAN_KMS = 12.0       # km/s mean for Mars crossers
IMPACT_VELOCITY_STD_KMS = 4.0         # standard deviation
IMPACT_VELOCITY_MIN_KMS = 5.0         # minimum (slow encounter)
IMPACT_VELOCITY_MAX_KMS = 25.0        # maximum (retrograde comet)

# Meteoroid physical properties
METEOROID_DENSITY_COMETARY = 1000.0   # kg/m³ (icy)
METEOROID_DENSITY_CHONDRITIC = 3500.0 # kg/m³ (rocky)
METEOROID_DENSITY_DEFAULT = 2500.0    # kg/m³ (average S-type)

# Whipple shield defaults (aluminium, ISS-heritage)
ALUMINIUM_DENSITY = 2700.0            # kg/m³
DEFAULT_BUMPER_MM = 1.0               # mm (thin outer sheet)
DEFAULT_STANDOFF_CM = 15.0            # cm (gap between bumper and wall)
DEFAULT_WALL_MM = 3.0                 # mm (structural backwall)

# Ballistic limit equation exponents (Cour-Palais simplified)
BLE_K = 3.0                           # overall scale factor
BLE_ALPHA = 0.5                       # bumper thickness exponent
BLE_BETA = 0.5                        # standoff exponent
BLE_GAMMA = 0.33                      # wall thickness exponent
BLE_DELTA = 0.5                       # projectile density exponent
BLE_EPSILON = 0.67                    # velocity exponent
BLE_ZETA = 1.0                        # cos(angle) exponent

# Cratering constants (Holsapple scaling)
CRATER_K = 1.1                        # empirical scale
CRATER_DENSITY_EXP = 0.33             # (ρ_p/ρ_t)^1/3
CRATER_VELOCITY_EXP = 0.67            # (V/V_ref)^2/3
CRATER_V_REF_KMS = 7.0               # reference velocity for scaling

# Mars atmospheric shielding (minimal)
MARS_ATMO_SHIELDING = 0.05           # fraction of small particles stopped
MARS_ATMO_MASS_CUTOFF_KG = 1e-9      # particles below this may be stopped

# Time conversion
SOLS_PER_YEAR = 668.6                 # Mars sols per Mars year


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WhippleShield:
    """Multi-layer micrometeoroid shield for a habitat module.

    bumper_mm: outer bumper thickness (mm)
    standoff_cm: gap between bumper and backwall (cm)
    wall_mm: structural backwall thickness (mm)
    area_m2: protected surface area (m²)
    health: fraction of shield undamaged [0, 1]
    cumulative_impacts: total impacts received over lifetime
    total_energy_absorbed_j: cumulative kinetic energy absorbed
    """
    bumper_mm: float = DEFAULT_BUMPER_MM
    standoff_cm: float = DEFAULT_STANDOFF_CM
    wall_mm: float = DEFAULT_WALL_MM
    area_m2: float = 200.0
    health: float = 1.0
    cumulative_impacts: int = 0
    total_energy_absorbed_j: float = 0.0

    def __post_init__(self) -> None:
        """Clamp fields to physical ranges."""
        self.bumper_mm = max(0.01, self.bumper_mm)
        self.standoff_cm = max(0.1, self.standoff_cm)
        self.wall_mm = max(0.1, self.wall_mm)
        self.area_m2 = max(0.01, self.area_m2)
        self.health = max(0.0, min(1.0, self.health))
        self.cumulative_impacts = max(0, self.cumulative_impacts)
        self.total_energy_absorbed_j = max(0.0, self.total_energy_absorbed_j)

    def effective_bumper_mm(self) -> float:
        """Bumper thickness adjusted for cumulative damage."""
        return self.bumper_mm * self.health

    def mass_kg(self) -> float:
        """Total shield mass in kg (bumper + backwall, aluminium)."""
        bumper_vol_m3 = self.area_m2 * (self.bumper_mm / 1000.0)
        wall_vol_m3 = self.area_m2 * (self.wall_mm / 1000.0)
        return (bumper_vol_m3 + wall_vol_m3) * ALUMINIUM_DENSITY


@dataclass
class ImpactEvent:
    """Record of a single meteoroid impact.

    mass_kg: impactor mass
    velocity_kms: impact velocity in km/s
    diameter_m: impactor diameter
    angle_deg: impact angle from normal (0 = head-on)
    penetrated: whether the shield was breached
    crater_depth_mm: crater depth in bumper (if not penetrated)
    kinetic_energy_j: impact kinetic energy
    """
    mass_kg: float
    velocity_kms: float
    diameter_m: float
    angle_deg: float
    penetrated: bool
    crater_depth_mm: float
    kinetic_energy_j: float


# ---------------------------------------------------------------------------
# Meteoroid flux model
# ---------------------------------------------------------------------------

def meteoroid_mass_to_diameter(mass_kg: float,
                                density: float = METEOROID_DENSITY_DEFAULT) -> float:
    """Convert meteoroid mass to diameter assuming a sphere.

    Args:
        mass_kg: particle mass in kg
        density: bulk density in kg/m³

    Returns:
        Diameter in metres.
    """
    if mass_kg <= 0 or density <= 0:
        return 0.0
    volume = mass_kg / density
    radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    return 2.0 * radius


def grun_flux_1au(min_mass_kg: float) -> float:
    """Cumulative meteoroid flux at 1 AU from Grün (1985) model.

    Args:
        min_mass_kg: minimum particle mass (kg)

    Returns:
        Flux in particles / m² / year for particles ≥ min_mass_kg.
    """
    if min_mass_kg <= 0:
        min_mass_kg = GRUN_MIN_MASS_KG
    log_m = math.log10(min_mass_kg)
    log_flux = GRUN_INTERCEPT + GRUN_SLOPE * log_m
    return 10.0 ** log_flux


def mars_flux(min_mass_kg: float) -> float:
    """Meteoroid flux at Mars orbit, adjusted from 1 AU.

    Applies heliocentric distance scaling and asteroid belt proximity
    correction factor.

    Args:
        min_mass_kg: minimum particle mass (kg)

    Returns:
        Flux in particles / m² / year.
    """
    base = grun_flux_1au(min_mass_kg)
    distance_scale = AU_MARS ** FLUX_HELIO_EXPONENT
    # Apply atmospheric shielding for very small particles
    if min_mass_kg < MARS_ATMO_MASS_CUTOFF_KG:
        atmo_factor = 1.0 - MARS_ATMO_SHIELDING
    else:
        atmo_factor = 1.0
    return base * distance_scale * ASTEROID_BELT_FACTOR * atmo_factor


def impact_probability(flux_per_m2_year: float, area_m2: float,
                       duration_sols: float) -> float:
    """Probability of at least one impact (Poisson model).

    Args:
        flux_per_m2_year: particle flux (particles/m²/year)
        area_m2: exposed area (m²)
        duration_sols: exposure time in sols

    Returns:
        Probability of ≥1 impact, clamped to [0, 1].
    """
    duration_years = duration_sols / SOLS_PER_YEAR
    expected = flux_per_m2_year * area_m2 * duration_years
    if expected <= 0:
        return 0.0
    prob = 1.0 - math.exp(-expected)
    return max(0.0, min(1.0, prob))


def expected_impacts(flux_per_m2_year: float, area_m2: float,
                     duration_sols: float) -> float:
    """Expected number of impacts (Poisson mean).

    Args:
        flux_per_m2_year: particle flux
        area_m2: exposed area
        duration_sols: exposure time

    Returns:
        Expected impact count (λ).
    """
    duration_years = duration_sols / SOLS_PER_YEAR
    return max(0.0, flux_per_m2_year * area_m2 * duration_years)


# ---------------------------------------------------------------------------
# Ballistic limit & penetration
# ---------------------------------------------------------------------------

def critical_diameter(shield: WhippleShield,
                      velocity_kms: float = IMPACT_VELOCITY_MEAN_KMS,
                      density_proj: float = METEOROID_DENSITY_DEFAULT,
                      angle_deg: float = 0.0) -> float:
    """Critical meteoroid diameter that just penetrates the shield.

    Uses simplified Cour-Palais ballistic limit equation.

    Args:
        shield: WhippleShield configuration
        velocity_kms: impact velocity in km/s
        density_proj: projectile density in kg/m³
        angle_deg: impact angle from surface normal (degrees)

    Returns:
        Critical diameter in metres.  Particles larger than this
        will penetrate the shield.
    """
    t_b = shield.effective_bumper_mm() / 1000.0  # to metres
    s = shield.standoff_cm / 100.0                # to metres
    t_w = shield.wall_mm / 1000.0                 # to metres

    cos_theta = math.cos(math.radians(max(0.0, min(89.0, angle_deg))))

    # Avoid division by zero
    v = max(0.1, velocity_kms)
    rho_p = max(100.0, density_proj)

    numerator = BLE_K * (t_b ** BLE_ALPHA) * (s ** BLE_BETA) * (t_w ** BLE_GAMMA)
    denominator = (rho_p ** BLE_DELTA) * (v ** BLE_EPSILON) * (cos_theta ** BLE_ZETA)

    return numerator / denominator


def kinetic_energy_j(mass_kg: float, velocity_kms: float) -> float:
    """Impact kinetic energy in joules.

    Args:
        mass_kg: impactor mass
        velocity_kms: impact velocity in km/s

    Returns:
        Kinetic energy in joules.
    """
    v_ms = velocity_kms * 1000.0  # km/s to m/s
    return 0.5 * max(0.0, mass_kg) * v_ms * v_ms


def check_penetration(shield: WhippleShield, diameter_m: float,
                      velocity_kms: float = IMPACT_VELOCITY_MEAN_KMS,
                      density_proj: float = METEOROID_DENSITY_DEFAULT,
                      angle_deg: float = 0.0) -> bool:
    """Check whether a meteoroid penetrates the shield.

    Args:
        shield: shield configuration
        diameter_m: meteoroid diameter in metres
        velocity_kms: impact velocity in km/s
        density_proj: projectile density
        angle_deg: impact angle from normal

    Returns:
        True if the shield is penetrated.
    """
    d_crit = critical_diameter(shield, velocity_kms, density_proj, angle_deg)
    return diameter_m > d_crit


# ---------------------------------------------------------------------------
# Cratering (non-penetrating impacts)
# ---------------------------------------------------------------------------

def crater_depth_mm(diameter_m: float,
                    velocity_kms: float = IMPACT_VELOCITY_MEAN_KMS,
                    density_proj: float = METEOROID_DENSITY_DEFAULT,
                    density_target: float = ALUMINIUM_DENSITY) -> float:
    """Crater depth in bumper from a non-penetrating impact.

    Holsapple (1993) scaling law.

    Args:
        diameter_m: impactor diameter (m)
        velocity_kms: impact velocity (km/s)
        density_proj: projectile density (kg/m³)
        density_target: target material density (kg/m³)

    Returns:
        Crater depth in mm.
    """
    if diameter_m <= 0 or velocity_kms <= 0:
        return 0.0
    density_ratio = (max(1.0, density_proj) / max(1.0, density_target))
    velocity_ratio = (max(0.01, velocity_kms) / CRATER_V_REF_KMS)
    depth_m = (CRATER_K
               * (density_ratio ** CRATER_DENSITY_EXP)
               * diameter_m
               * (velocity_ratio ** CRATER_VELOCITY_EXP))
    return depth_m * 1000.0  # metres to mm


# ---------------------------------------------------------------------------
# Shield degradation
# ---------------------------------------------------------------------------

def damage_from_impact(shield: WhippleShield,
                       crater_mm: float) -> float:
    """Health reduction from a single non-penetrating impact.

    Damage is proportional to crater depth relative to bumper thickness.

    Args:
        shield: current shield state
        crater_mm: crater depth in mm

    Returns:
        Health reduction (fraction), always ≥ 0.
    """
    if shield.bumper_mm <= 0 or crater_mm <= 0:
        return 0.0
    # Damaged area fraction: crater depth / bumper thickness, scaled
    # by a small factor (each crater damages a tiny patch)
    fraction = (crater_mm / shield.bumper_mm) * 0.001
    return max(0.0, fraction)


# ---------------------------------------------------------------------------
# Simulation tick
# ---------------------------------------------------------------------------

def random_impact_velocity() -> float:
    """Sample a random impact velocity from the Mars distribution.

    Returns:
        Velocity in km/s, clamped to physical bounds.
    """
    v = random.gauss(IMPACT_VELOCITY_MEAN_KMS, IMPACT_VELOCITY_STD_KMS)
    return max(IMPACT_VELOCITY_MIN_KMS, min(IMPACT_VELOCITY_MAX_KMS, v))


def random_impact_angle() -> float:
    """Sample a random impact angle.

    Uniform in cos(θ) gives isotropic distribution on hemisphere.

    Returns:
        Angle in degrees [0, 90).
    """
    cos_theta = random.random()
    return math.degrees(math.acos(cos_theta))


def generate_impact(min_mass_kg: float = 1e-6) -> tuple[float, float, float]:
    """Generate a random meteoroid impact event.

    Args:
        min_mass_kg: minimum particle mass to consider

    Returns:
        Tuple of (mass_kg, velocity_kms, angle_deg).
    """
    # Sample mass from power-law distribution
    # P(>m) ∝ m^(-1.213), so m ~ U^(1/(1+slope))
    u = random.random()
    if u <= 0:
        u = 1e-15
    exponent = 1.0 / (1.0 - abs(GRUN_SLOPE))
    mass = min_mass_kg * (u ** (-exponent))
    mass = min(GRUN_MAX_MASS_KG, mass)

    velocity = random_impact_velocity()
    angle = random_impact_angle()
    return mass, velocity, angle


def tick_shield(shield: WhippleShield, sols: float = 1.0,
                min_mass_kg: float = 1e-6,
                rng_seed: int | None = None) -> list[ImpactEvent]:
    """Advance the shield simulation by the given number of sols.

    Computes expected impacts from the meteoroid flux, generates
    random events, checks penetration, accumulates damage.

    Args:
        shield: shield to simulate (mutated in place)
        sols: number of sols to simulate
        min_mass_kg: minimum meteoroid mass to track
        rng_seed: optional RNG seed for reproducibility

    Returns:
        List of ImpactEvent records for this tick.
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    flux = mars_flux(min_mass_kg)
    n_expected = expected_impacts(flux, shield.area_m2, sols)

    # Poisson-sample the actual count
    n_actual = 0
    if n_expected > 0:
        # For small λ use direct Poisson sampling
        if n_expected < 30:
            L = math.exp(-n_expected)
            k = 0
            p = 1.0
            while True:
                k += 1
                p *= random.random()
                if p < L:
                    break
            n_actual = k - 1
        else:
            # Normal approximation for large λ
            n_actual = max(0, round(random.gauss(n_expected,
                                                  math.sqrt(n_expected))))

    events: list[ImpactEvent] = []

    for _ in range(n_actual):
        mass, velocity, angle = generate_impact(min_mass_kg)
        diameter = meteoroid_mass_to_diameter(mass)
        penetrated = check_penetration(shield, diameter, velocity,
                                        METEOROID_DENSITY_DEFAULT, angle)
        ke = kinetic_energy_j(mass, velocity)

        if penetrated:
            crater_mm = 0.0  # full penetration — no simple crater
        else:
            crater_mm = crater_depth_mm(diameter, velocity)
            dmg = damage_from_impact(shield, crater_mm)
            shield.health = max(0.0, shield.health - dmg)

        shield.cumulative_impacts += 1
        shield.total_energy_absorbed_j += ke

        events.append(ImpactEvent(
            mass_kg=mass,
            velocity_kms=velocity,
            diameter_m=diameter,
            angle_deg=angle,
            penetrated=penetrated,
            crater_depth_mm=crater_mm,
            kinetic_energy_j=ke,
        ))

    return events


def shield_status(shield: WhippleShield) -> dict:
    """Return a JSON-serialisable status summary.

    Args:
        shield: current shield state

    Returns:
        Dict with shield health metrics.
    """
    return {
        "health_pct": round(shield.health * 100.0, 2),
        "effective_bumper_mm": round(shield.effective_bumper_mm(), 3),
        "cumulative_impacts": shield.cumulative_impacts,
        "total_energy_absorbed_kj": round(shield.total_energy_absorbed_j / 1000.0, 3),
        "mass_kg": round(shield.mass_kg(), 1),
        "area_m2": shield.area_m2,
        "penetration_risk": "LOW" if shield.health > 0.7 else (
            "MEDIUM" if shield.health > 0.3 else "HIGH"
        ),
    }
