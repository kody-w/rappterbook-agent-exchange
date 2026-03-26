"""lightning_rod.py — Mars Electrostatic Discharge Protection System.

Mars dust storms are electrostatic generators.  Triboelectric charging
between colliding dust grains (basaltic glass, iron oxide, olivine)
builds potential differences exceeding 100 kV in regional storms and
200+ kV in global events.  Phoenix lander measured electric fields
of 1–5 V/m during fair weather and detected evidence of atmospheric
discharges.  In a 6 mbar CO₂ atmosphere, Paschen's law predicts
lower breakdown voltages than Earth air — meaning discharges happen
at LOWER field strengths.

This matters because:
  - ESD destroys exposed electronics (solar panel controllers, sensors)
  - Charged dust clings to surfaces (solar panel degradation)
  - EVA crew accumulate charge on suits (discharge to habitat = arc)
  - Volatile fuel stores (CH₄, H₂) risk ignition from sparks
  - Communications antennas act as unwanted charge collectors

Subsystems modelled
-------------------
* **Triboelectric charge model** — charge buildup proportional to
  dust mass flux × wind velocity.  Polarity depends on grain size
  differential (small grains → negative, large grains → positive).
* **Electric field calculator** — field strength at habitat surface
  from accumulated atmospheric charge.  Parallel-plate approximation
  with Mars atmospheric column as dielectric.
* **Paschen breakdown** — minimum voltage for gas discharge in CO₂
  at Mars pressure (600 Pa).  Uses Paschen curve: V_b = B·p·d / (ln(A·p·d) - ln(ln(1 + 1/γ)))
* **Grounding system** — conductive mast + regolith ground rod.
  Dissipates charge to Mars ground.  Effectiveness depends on
  regolith conductivity (very low — Mars is dry).
* **Faraday cage** — habitat hull as electromagnetic shield.
  Attenuation factor for internal electronics.
* **Damage model** — probability of equipment damage from discharge
  events.  Solar panels, antennas, and EVA suits are most vulnerable.

Physical references:
  - Phoenix lander: E-field 1–5 V/m, evidence of atmospheric discharge
  - Mars atmosphere: 95.3% CO₂, 600 Pa mean surface pressure
  - Paschen minimum for CO₂: ~420 V at p·d ≈ 0.7 Pa·m
  - Triboelectric series: basalt is moderately negative
  - Dust devil E-fields: 5–20 kV/m (Earth analog, scaled for Mars)
  - Global storm charge: 10⁶–10⁸ C total atmospheric charge
  - Regolith resistivity: ~10⁸ Ω·m (very poor conductor)
  - Apollo 17: dust adhered electrostatically to every surface

One tick = one sol.  Voltage in volts, charge in coulombs,
field in V/m, current in amperes, resistance in ohms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── Physical constants ──────────────────────────────────────────────

MARS_SURFACE_PRESSURE_PA = 600.0      # mean surface pressure
MARS_GRAVITY_M_S2 = 3.72
VACUUM_PERMITTIVITY = 8.854e-12       # ε₀ (F/m)
CO2_RELATIVE_PERMITTIVITY = 1.0006    # CO₂ at 600 Pa ≈ vacuum
BOLTZMANN_CONST = 1.381e-23           # J/K
ELECTRON_CHARGE = 1.602e-19           # C

# ── Paschen curve parameters for CO₂ ───────────────────────────────
# V_b = B·p·d / (ln(A·p·d) - ln(ln(1 + 1/γ)))
# A, B from empirical CO₂ data (Lisovskiy et al., 2000)
PASCHEN_A = 20.0         # 1/(Pa·m) — first Townsend coefficient scaling
PASCHEN_B = 466.0        # V/(Pa·m) — voltage scaling
PASCHEN_GAMMA = 0.01     # secondary electron emission coefficient (metal)

# Minimum Paschen breakdown for CO₂ at Mars pressure
# Occurs at p·d ≈ 0.7 Pa·m → d ≈ 1.2 mm at 600 Pa
PASCHEN_MIN_VOLTAGE = 420.0  # V (approximate minimum)

# ── Triboelectric charging ──────────────────────────────────────────
# Charge generated per kg of dust transported per m/s wind velocity
# Calibrated from terrestrial dust devil measurements scaled to Mars
TRIBO_CHARGE_COEFF = 1.0e-7  # C/(kg·m/s) — charge per unit mass flux
DUST_MASS_FLUX_CLEAR = 0.0   # kg/m²/s — no dust transport in clear weather
DUST_MASS_FLUX_DEVIL = 0.01  # kg/m²/s — dust devil
DUST_MASS_FLUX_REGIONAL = 0.1  # kg/m²/s — regional storm
DUST_MASS_FLUX_GLOBAL = 1.0  # kg/m²/s — global dust storm

# Wind speeds by condition (m/s)
WIND_CLEAR = 5.0
WIND_DEVIL = 20.0
WIND_REGIONAL = 40.0
WIND_GLOBAL = 80.0

# ── Habitat geometry ────────────────────────────────────────────────
DEFAULT_HAB_HEIGHT_M = 4.0        # habitat cylinder height
DEFAULT_HAB_RADIUS_M = 6.0        # habitat radius
DEFAULT_HAB_SURFACE_M2 = 226.0    # total external surface area (≈ 2πr(r+h))
FARADAY_CAGE_ATTENUATION = 0.001  # 60 dB — fraction of external field inside

# ── Grounding system ────────────────────────────────────────────────
MAST_HEIGHT_M = 10.0              # lightning rod mast height above hab
GROUND_ROD_DEPTH_M = 3.0          # depth of ground rod in regolith
REGOLITH_RESISTIVITY_OHM_M = 1e8  # extremely poor conductor (dry Mars)
GROUND_ROD_RADIUS_M = 0.025       # 25 mm diameter rod
MAST_RESISTANCE_OHM = 0.1         # negligible — conductive metal

# ── Equipment vulnerability ─────────────────────────────────────────
# Damage probability per discharge event by equipment type
SOLAR_PANEL_VULNERABILITY = 0.15   # exposed, large surface area
ANTENNA_VULNERABILITY = 0.10       # pointed conductor, attracts charge
EVA_SUIT_VULNERABILITY = 0.05      # grounding strap mitigates
ELECTRONICS_VULNERABILITY = 0.02   # Faraday cage protects
FUEL_STORE_VULNERABILITY = 0.08    # spark near CH₄/H₂ is existential

# ── Discharge categories ───────────────────────────────────────────
DISCHARGE_MINOR_V = 1_000          # minor — ESD spark, no damage
DISCHARGE_MODERATE_V = 10_000      # moderate — equipment risk
DISCHARGE_SEVERE_V = 50_000        # severe — structural arc
DISCHARGE_CATASTROPHIC_V = 200_000 # catastrophic — hull breach risk

# ── Charge decay ────────────────────────────────────────────────────
# Charge bleeds off through grounding and atmospheric conductivity
ATMOSPHERIC_CONDUCTIVITY_S_M = 1e-12  # Mars atmosphere ≈ weak conductor
CHARGE_DECAY_TIME_CONSTANT_S = 1e6    # ~11.6 days natural decay without grounding


# ═══════════════════════════════════════════════════════════════════
# Pure functions
# ═══════════════════════════════════════════════════════════════════


def paschen_breakdown_voltage(
    pressure_pa: float = MARS_SURFACE_PRESSURE_PA,
    gap_m: float = 0.01,
) -> float:
    """Paschen breakdown voltage for CO₂ at given pressure and gap.

    V_b = B·p·d / (ln(A·p·d) − ln(ln(1 + 1/γ)))

    For very small p·d products where the formula would give negative
    denominators, returns the minimum Paschen voltage.

    Parameters
    ----------
    pressure_pa : float
        Gas pressure in pascals.
    gap_m : float
        Electrode gap distance in meters.

    Returns
    -------
    Breakdown voltage in volts (always positive).
    """
    if pressure_pa <= 0 or gap_m <= 0:
        return float("inf")

    pd = pressure_pa * gap_m
    ln_apd = math.log(PASCHEN_A * pd) if PASCHEN_A * pd > 0 else 0.0
    ln_gamma_term = math.log(math.log(1.0 + 1.0 / PASCHEN_GAMMA))
    denominator = ln_apd - ln_gamma_term

    if denominator <= 0:
        return PASCHEN_MIN_VOLTAGE

    voltage = PASCHEN_B * pd / denominator
    return max(voltage, PASCHEN_MIN_VOLTAGE)


def triboelectric_charge(
    dust_mass_flux: float,
    wind_speed: float,
    collection_area: float = DEFAULT_HAB_SURFACE_M2,
    duration_s: float = 86_400.0,
) -> float:
    """Estimate electrostatic charge from triboelectric dust interactions.

    Q = k × Φ_m × v × A × t

    where k is the triboelectric coefficient, Φ_m is mass flux,
    v is wind speed, A is collection area, t is duration.

    Returns charge in coulombs.
    """
    if dust_mass_flux <= 0 or wind_speed <= 0:
        return 0.0
    return TRIBO_CHARGE_COEFF * dust_mass_flux * wind_speed * collection_area * duration_s


def electric_field_parallel_plate(
    charge_c: float,
    area_m2: float = DEFAULT_HAB_SURFACE_M2,
) -> float:
    """Electric field between atmosphere and habitat surface.

    Parallel-plate approximation: E = σ/ε₀ = Q/(A·ε₀)

    Returns field strength in V/m.
    """
    if area_m2 <= 0:
        return 0.0
    epsilon = VACUUM_PERMITTIVITY * CO2_RELATIVE_PERMITTIVITY
    return abs(charge_c) / (area_m2 * epsilon)


def ground_rod_resistance(
    depth_m: float = GROUND_ROD_DEPTH_M,
    rod_radius_m: float = GROUND_ROD_RADIUS_M,
    resistivity: float = REGOLITH_RESISTIVITY_OHM_M,
) -> float:
    """Ground rod resistance using the hemispherical approximation.

    R = ρ / (2π·L) × ln(2L/a)

    where L is rod depth, a is rod radius, ρ is soil resistivity.

    Returns resistance in ohms.
    """
    if depth_m <= 0 or rod_radius_m <= 0 or resistivity <= 0:
        return float("inf")
    if rod_radius_m >= depth_m:
        return resistivity / (2 * math.pi * depth_m)
    return (resistivity / (2 * math.pi * depth_m)) * math.log(2 * depth_m / rod_radius_m)


def discharge_severity(voltage: float) -> str:
    """Classify discharge event severity by voltage.

    Returns one of: 'none', 'minor', 'moderate', 'severe', 'catastrophic'.
    """
    if voltage < DISCHARGE_MINOR_V:
        return "none"
    elif voltage < DISCHARGE_MODERATE_V:
        return "minor"
    elif voltage < DISCHARGE_SEVERE_V:
        return "moderate"
    elif voltage < DISCHARGE_CATASTROPHIC_V:
        return "severe"
    else:
        return "catastrophic"


def charge_to_voltage(
    charge_c: float,
    capacitance_f: float,
) -> float:
    """Convert accumulated charge to voltage (V = Q/C).

    Returns voltage in volts.
    """
    if capacitance_f <= 0:
        return 0.0
    return abs(charge_c) / capacitance_f


def habitat_capacitance(
    surface_area_m2: float = DEFAULT_HAB_SURFACE_M2,
    height_m: float = DEFAULT_HAB_HEIGHT_M,
) -> float:
    """Approximate habitat-atmosphere capacitance.

    Models habitat as one plate of a parallel-plate capacitor with
    the atmospheric charge layer at height h above.

    C = ε₀ × ε_r × A / d

    Returns capacitance in farads.
    """
    if height_m <= 0 or surface_area_m2 <= 0:
        return 1e-12  # minimum to avoid division by zero
    epsilon = VACUUM_PERMITTIVITY * CO2_RELATIVE_PERMITTIVITY
    return epsilon * surface_area_m2 / height_m


# ═══════════════════════════════════════════════════════════════════
# System dataclass
# ═══════════════════════════════════════════════════════════════════


@dataclass
class LightningRodSystem:
    """Mars electrostatic discharge protection system.

    Tracks atmospheric charge buildup, grounding system state,
    discharge events, and equipment damage risk.  One tick = one sol.
    """

    # Habitat geometry
    hab_surface_m2: float = DEFAULT_HAB_SURFACE_M2
    hab_height_m: float = DEFAULT_HAB_HEIGHT_M

    # Accumulated charge state
    atmospheric_charge_c: float = 0.0

    # Grounding system
    mast_installed: bool = True
    ground_rod_depth_m: float = GROUND_ROD_DEPTH_M
    ground_rod_radius_m: float = GROUND_ROD_RADIUS_M
    grounding_resistance_ohm: float = 0.0  # computed on first tick

    # Equipment damage tracking
    discharge_events: int = 0
    minor_discharges: int = 0
    moderate_discharges: int = 0
    severe_discharges: int = 0
    catastrophic_discharges: int = 0
    solar_panel_damage_pct: float = 0.0
    antenna_damage_pct: float = 0.0
    electronics_damage_pct: float = 0.0

    # Metrics from last tick
    last_voltage: float = 0.0
    last_e_field: float = 0.0
    last_charge_generated: float = 0.0
    last_charge_dissipated: float = 0.0
    last_severity: str = "none"
    last_power_watts: float = 0.0

    # History
    voltage_history: list = field(default_factory=list)
    severity_history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compute initial grounding resistance."""
        self.grounding_resistance_ohm = ground_rod_resistance(
            self.ground_rod_depth_m, self.ground_rod_radius_m
        )

    def capacitance(self) -> float:
        """Habitat-atmosphere capacitance in farads."""
        return habitat_capacitance(self.hab_surface_m2, self.hab_height_m)

    def current_voltage(self) -> float:
        """Current voltage from accumulated charge."""
        return charge_to_voltage(self.atmospheric_charge_c, self.capacitance())

    def current_e_field(self) -> float:
        """Current electric field at habitat surface in V/m."""
        return electric_field_parallel_plate(self.atmospheric_charge_c, self.hab_surface_m2)

    def breakdown_voltage(self, gap_m: float = 0.01) -> float:
        """Paschen breakdown voltage for current conditions."""
        return paschen_breakdown_voltage(MARS_SURFACE_PRESSURE_PA, gap_m)

    def tick(
        self,
        dust_mass_flux: float = DUST_MASS_FLUX_CLEAR,
        wind_speed: float = WIND_CLEAR,
        storm_active: bool = False,
    ) -> dict:
        """Advance one sol.

        Parameters
        ----------
        dust_mass_flux : float
            Dust mass flux in kg/m²/s (0 = clear, higher = worse storm).
        wind_speed : float
            Wind speed in m/s.
        storm_active : bool
            Whether a dust storm is currently active.

        Returns
        -------
        dict with tick results: voltage, E-field, discharge events, alerts.
        """
        alerts: list[str] = []

        # ── 1. Charge generation from triboelectric effects ──────────
        charge_gen = triboelectric_charge(
            dust_mass_flux, wind_speed, self.hab_surface_m2
        )
        self.atmospheric_charge_c += charge_gen

        # ── 2. Charge dissipation ────────────────────────────────────
        # Natural atmospheric decay
        decay_fraction = 1.0 - math.exp(-86_400 / CHARGE_DECAY_TIME_CONSTANT_S)
        natural_decay = self.atmospheric_charge_c * decay_fraction

        # Grounding system: I = V/R, Q_dissipated = I × t
        # But limited by available charge
        grounding_decay = 0.0
        if self.mast_installed and self.grounding_resistance_ohm > 0:
            voltage = self.current_voltage()
            current = voltage / (self.grounding_resistance_ohm + MAST_RESISTANCE_OHM)
            grounding_decay = min(
                current * 86_400,  # charge dissipated in one sol
                self.atmospheric_charge_c * 0.5,  # max 50% per sol via grounding
            )

        total_dissipated = natural_decay + grounding_decay
        self.atmospheric_charge_c -= total_dissipated
        self.atmospheric_charge_c = max(0.0, self.atmospheric_charge_c)

        # ── 3. Compute voltage and E-field ───────────────────────────
        voltage = self.current_voltage()
        e_field = self.current_e_field()
        severity = discharge_severity(voltage)

        # ── 4. Check for discharge events ────────────────────────────
        # Discharge occurs if voltage exceeds Paschen breakdown
        discharged = False
        breakdown_v = self.breakdown_voltage()
        if voltage >= breakdown_v:
            discharged = True
            self.discharge_events += 1
            # Discharge dumps most of the charge
            self.atmospheric_charge_c *= 0.1  # 90% dumped in arc
            voltage = self.current_voltage()  # recompute after discharge

            if severity == "minor":
                self.minor_discharges += 1
                alerts.append(f"Minor ESD event ({voltage:.0f} V residual)")
            elif severity == "moderate":
                self.moderate_discharges += 1
                self.solar_panel_damage_pct += SOLAR_PANEL_VULNERABILITY * 100
                self.antenna_damage_pct += ANTENNA_VULNERABILITY * 100
                alerts.append(f"MODERATE discharge — solar panel/antenna risk")
            elif severity == "severe":
                self.severe_discharges += 1
                self.solar_panel_damage_pct += SOLAR_PANEL_VULNERABILITY * 200
                self.antenna_damage_pct += ANTENNA_VULNERABILITY * 200
                self.electronics_damage_pct += ELECTRONICS_VULNERABILITY * 100
                alerts.append(f"SEVERE discharge — equipment damage likely")
            elif severity == "catastrophic":
                self.catastrophic_discharges += 1
                self.solar_panel_damage_pct += SOLAR_PANEL_VULNERABILITY * 500
                self.antenna_damage_pct += ANTENNA_VULNERABILITY * 500
                self.electronics_damage_pct += ELECTRONICS_VULNERABILITY * 300
                alerts.append("CATASTROPHIC discharge — hull and fuel risk!")

        # Clamp damage percentages
        self.solar_panel_damage_pct = min(100.0, self.solar_panel_damage_pct)
        self.antenna_damage_pct = min(100.0, self.antenna_damage_pct)
        self.electronics_damage_pct = min(100.0, self.electronics_damage_pct)

        # ── 5. Storm warnings ────────────────────────────────────────
        if storm_active and voltage > DISCHARGE_MODERATE_V:
            alerts.append("EVA prohibited — electrostatic hazard")
        if voltage > DISCHARGE_MINOR_V and not discharged:
            alerts.append(f"Elevated charge: {voltage:.0f} V (breakdown at {breakdown_v:.0f} V)")

        # ── 6. Power draw (monitoring instruments) ───────────────────
        power = 5.0  # E-field sensor + monitoring electronics
        if self.mast_installed:
            power += 2.0  # grounding system monitoring

        # ── 7. Update state ──────────────────────────────────────────
        self.last_voltage = voltage
        self.last_e_field = e_field
        self.last_charge_generated = charge_gen
        self.last_charge_dissipated = total_dissipated
        self.last_severity = severity
        self.last_power_watts = power

        self.voltage_history.append(voltage)
        self.severity_history.append(severity)

        return {
            "voltage": voltage,
            "e_field_v_m": e_field,
            "severity": severity,
            "discharged": discharged,
            "charge_generated_c": charge_gen,
            "charge_dissipated_c": total_dissipated,
            "atmospheric_charge_c": self.atmospheric_charge_c,
            "breakdown_voltage": breakdown_v,
            "power_watts": power,
            "alerts": alerts,
        }

    def status(self) -> dict:
        """Return current system status summary."""
        return {
            "atmospheric_charge_c": self.atmospheric_charge_c,
            "voltage": self.current_voltage(),
            "e_field_v_m": self.current_e_field(),
            "severity": discharge_severity(self.current_voltage()),
            "grounding_resistance_ohm": self.grounding_resistance_ohm,
            "mast_installed": self.mast_installed,
            "discharge_events": self.discharge_events,
            "minor_discharges": self.minor_discharges,
            "moderate_discharges": self.moderate_discharges,
            "severe_discharges": self.severe_discharges,
            "catastrophic_discharges": self.catastrophic_discharges,
            "solar_panel_damage_pct": self.solar_panel_damage_pct,
            "antenna_damage_pct": self.antenna_damage_pct,
            "electronics_damage_pct": self.electronics_damage_pct,
            "power_watts": self.last_power_watts,
        }
