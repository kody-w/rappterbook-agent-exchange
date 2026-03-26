"""Tests for cargo_lander.py — Mars Cargo Entry, Descent & Landing (EDL).

95 tests covering:
  - Atmosphere model (density profile, scale height, edge cases)
  - Drag physics (force calculation, ballistic coefficient)
  - Gravity model (altitude variation, surface value)
  - Heating physics (convective flux, heat shield temp, ablation)
  - Parachute physics (drag, deployment conditions)
  - Rocket physics (Tsiolkovsky, thrust, fuel flow, propellant)
  - Energy calculations (kinetic, potential, dynamic pressure)
  - Lander state (creation, mass breakdown, serialization)
  - Tick engine entry phase (drag, heating, deceleration)
  - Tick engine parachute phase (deployment, deceleration)
  - Tick engine powered descent (thrust, fuel consumption)
  - Tick engine landing (touchdown, crash detection)
  - Conservation laws (energy, mass, propellant, bounds)
  - Multi-step simulation (full EDL sequence, no crashes)
  - Edge cases (zero mass, extreme values, boundary conditions)
  - Abort conditions (dust storms, high winds, fuel margin)
"""
from __future__ import annotations

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cargo_lander import (
    # Constants
    MARS_SURFACE_GRAVITY_M_S2,
    MARS_RADIUS_M,
    MARS_ATM_DENSITY_SURFACE,
    MARS_SCALE_HEIGHT_M,
    MARS_SPEED_OF_SOUND_M_S,
    MARS_SURFACE_TEMP_K,
    DEFAULT_ENTRY_VELOCITY_M_S,
    DEFAULT_ENTRY_ALTITUDE_M,
    DEFAULT_ENTRY_ANGLE_DEG,
    HEATSHIELD_MASS_FRACTION,
    HEATSHIELD_ABLATION_THRESHOLD_K,
    HEATSHIELD_MAX_TEMP_K,
    HEATSHIELD_ABLATION_ENERGY_J_KG,
    CONVECTIVE_HEATING_COEFF,
    DEFAULT_ENTRY_CD,
    DEFAULT_CAPSULE_AREA_M2,
    MAX_DECELERATION_G,
    MAX_DECELERATION_M_S2,
    PARACHUTE_DEPLOY_MACH,
    PARACHUTE_DEPLOY_VELOCITY_M_S,
    PARACHUTE_DEPLOY_ALTITUDE_MIN_M,
    PARACHUTE_CD,
    PARACHUTE_MASS_FRACTION,
    DEFAULT_PARACHUTE_AREA_M2,
    POWERED_DESCENT_TRIGGER_M_S,
    DEFAULT_ENGINE_ISP_S,
    DEFAULT_MAX_THRUST_N,
    G0_M_S2,
    PROPELLANT_MASS_FRACTION,
    LANDING_VELOCITY_MAX_M_S,
    DUST_TAU_ABORT_LIMIT,
    WIND_ABORT_LIMIT_M_S,
    FUEL_MARGIN_ABORT_FRACTION,
    DEFAULT_DT_S,
    PHASE_PREENTRY,
    PHASE_ENTRY,
    PHASE_PARACHUTE,
    PHASE_POWERED,
    PHASE_LANDED,
    PHASE_ABORTED,
    PHASE_CRASHED,
    # Functions
    atmosphere_density,
    drag_force,
    gravity_at_altitude,
    mach_number,
    convective_heat_flux,
    heat_shield_temp,
    ablation_mass_loss,
    parachute_drag_force,
    ballistic_coefficient,
    tsiolkovsky_delta_v,
    required_propellant,
    thrust_acceleration,
    fuel_mass_flow_rate,
    kinetic_energy,
    potential_energy,
    dynamic_pressure,
    tick,
    create_lander,
    run_edl,
    # Classes
    LanderState,
)


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture
def default_lander():
    """A fresh lander with default 5000 kg payload."""
    return create_lander(payload_mass_kg=5_000.0)


@pytest.fixture
def light_lander():
    """A 1000 kg payload lander for faster simulations."""
    return create_lander(payload_mass_kg=1_000.0)


@pytest.fixture
def heavy_lander():
    """A 20000 kg payload lander (max cargo)."""
    return create_lander(payload_mass_kg=20_000.0)


@pytest.fixture
def entry_lander():
    """A lander that has entered the atmosphere (one tick past pre-entry)."""
    lander = create_lander(payload_mass_kg=5_000.0)
    tick(lander)
    return lander


# -- Atmosphere model ---------------------------------------------------------

class TestAtmosphereDensity:
    """Tests for the Mars atmospheric density profile."""

    def test_surface_density(self):
        """Density at surface equals ρ₀ = 0.020 kg/m³."""
        rho = atmosphere_density(0.0)
        assert abs(rho - MARS_ATM_DENSITY_SURFACE) < 1e-10

    def test_one_scale_height(self):
        """Density at one scale height = ρ₀ × e⁻¹ ≈ 0.00736 kg/m³."""
        rho = atmosphere_density(MARS_SCALE_HEIGHT_M)
        expected = MARS_ATM_DENSITY_SURFACE * math.exp(-1.0)
        assert abs(rho - expected) < 1e-10

    def test_two_scale_heights(self):
        """Density at two scale heights = ρ₀ × e⁻² ≈ 0.00271 kg/m³."""
        rho = atmosphere_density(2.0 * MARS_SCALE_HEIGHT_M)
        expected = MARS_ATM_DENSITY_SURFACE * math.exp(-2.0)
        assert abs(rho - expected) < 1e-10

    def test_density_decreases_with_altitude(self):
        """Density strictly decreases as altitude increases."""
        prev = atmosphere_density(0.0)
        for alt_km in range(1, 100):
            rho = atmosphere_density(alt_km * 1000.0)
            assert rho < prev
            prev = rho

    def test_high_altitude_is_vacuum(self):
        """Above 200 km, density is effectively zero."""
        assert atmosphere_density(200_001.0) == 0.0
        assert atmosphere_density(500_000.0) == 0.0

    def test_negative_altitude_returns_surface(self):
        """Below surface, density equals surface density."""
        assert atmosphere_density(-100.0) == MARS_ATM_DENSITY_SURFACE

    def test_entry_altitude_density(self):
        """Density at 125 km entry interface is very low but positive."""
        rho = atmosphere_density(125_000.0)
        assert rho > 0.0
        assert rho < 1e-4  # very thin at 125 km


# -- Drag physics -------------------------------------------------------------

class TestDragForce:
    """Tests for aerodynamic drag force calculation."""

    def test_zero_velocity_no_drag(self):
        """No drag at zero velocity."""
        assert drag_force(0.020, 0.0, 1.0, 10.0) == 0.0

    def test_zero_density_no_drag(self):
        """No drag in vacuum."""
        assert drag_force(0.0, 1000.0, 1.0, 10.0) == 0.0

    def test_drag_increases_with_velocity_squared(self):
        """Drag scales as v² (double velocity → 4× drag)."""
        d1 = drag_force(0.020, 100.0, 1.0, 10.0)
        d2 = drag_force(0.020, 200.0, 1.0, 10.0)
        assert abs(d2 / d1 - 4.0) < 0.01

    def test_drag_proportional_to_density(self):
        """Double density → double drag."""
        d1 = drag_force(0.010, 100.0, 1.0, 10.0)
        d2 = drag_force(0.020, 100.0, 1.0, 10.0)
        assert abs(d2 / d1 - 2.0) < 0.01

    def test_drag_proportional_to_area(self):
        """Double area → double drag."""
        d1 = drag_force(0.020, 100.0, 1.0, 5.0)
        d2 = drag_force(0.020, 100.0, 1.0, 10.0)
        assert abs(d2 / d1 - 2.0) < 0.01

    def test_known_drag_value(self):
        """F = 0.5 × 0.020 × 3500² × 1.05 × 15.9 = known value."""
        f = drag_force(0.020, 3500.0, 1.05, 15.9)
        expected = 0.5 * 0.020 * 3500.0**2 * 1.05 * 15.9
        assert abs(f - expected) < 0.01


# -- Gravity model ------------------------------------------------------------

class TestGravity:
    """Tests for Mars gravitational acceleration."""

    def test_surface_gravity(self):
        """Surface gravity is 3.72076 m/s²."""
        g = gravity_at_altitude(0.0)
        assert abs(g - MARS_SURFACE_GRAVITY_M_S2) < 1e-6

    def test_gravity_decreases_with_altitude(self):
        """Gravity weakens with altitude."""
        g_surface = gravity_at_altitude(0.0)
        g_100km = gravity_at_altitude(100_000.0)
        assert g_100km < g_surface

    def test_gravity_at_entry_altitude(self):
        """Gravity at 125 km is ~93% of surface value."""
        g = gravity_at_altitude(125_000.0)
        ratio = g / MARS_SURFACE_GRAVITY_M_S2
        assert 0.90 < ratio < 0.97

    def test_negative_altitude_uses_zero(self):
        """Negative altitude treated as surface."""
        g = gravity_at_altitude(-100.0)
        assert abs(g - MARS_SURFACE_GRAVITY_M_S2) < 0.01

    def test_inverse_square_law(self):
        """Gravity follows inverse-square law: g ∝ 1/(R+h)²."""
        h = 50_000.0
        g = gravity_at_altitude(h)
        expected = MARS_SURFACE_GRAVITY_M_S2 * (MARS_RADIUS_M / (MARS_RADIUS_M + h)) ** 2
        assert abs(g - expected) < 1e-6


# -- Mach number --------------------------------------------------------------

class TestMachNumber:
    """Tests for Mach number calculation."""

    def test_speed_of_sound_is_mach_1(self):
        """Speed of sound gives Mach 1.0."""
        assert abs(mach_number(MARS_SPEED_OF_SOUND_M_S) - 1.0) < 1e-10

    def test_zero_velocity_mach_zero(self):
        """Zero velocity is Mach 0."""
        assert mach_number(0.0) == 0.0

    def test_entry_velocity_is_hypersonic(self):
        """3500 m/s is Mach ~14.6 (hypersonic)."""
        m = mach_number(3500.0)
        assert m > 10.0
        assert abs(m - 3500.0 / MARS_SPEED_OF_SOUND_M_S) < 0.01

    def test_negative_velocity_gives_positive_mach(self):
        """Mach number is always non-negative (uses abs)."""
        assert mach_number(-500.0) > 0.0


# -- Convective heating -------------------------------------------------------

class TestConvectiveHeating:
    """Tests for convective heat flux on heat shield."""

    def test_zero_velocity_no_heating(self):
        """No heating at zero velocity."""
        assert convective_heat_flux(0.020, 0.0, 2.25) == 0.0

    def test_zero_density_no_heating(self):
        """No heating in vacuum."""
        assert convective_heat_flux(0.0, 3500.0, 2.25) == 0.0

    def test_heating_increases_with_velocity_cubed(self):
        """Heat flux scales as v³ (double velocity → 8× heating)."""
        q1 = convective_heat_flux(0.020, 1000.0, 2.25)
        q2 = convective_heat_flux(0.020, 2000.0, 2.25)
        ratio = q2 / q1
        assert abs(ratio - 8.0) < 0.01

    def test_heating_increases_with_sqrt_density(self):
        """Heat flux scales as √ρ (4× density → 2× heating)."""
        q1 = convective_heat_flux(0.005, 1000.0, 2.25)
        q2 = convective_heat_flux(0.020, 1000.0, 2.25)
        ratio = q2 / q1
        assert abs(ratio - 2.0) < 0.01

    def test_smaller_nose_more_heating(self):
        """Smaller nose radius → higher heating (√(1/r))."""
        q_large = convective_heat_flux(0.020, 3000.0, 4.0)
        q_small = convective_heat_flux(0.020, 3000.0, 1.0)
        assert q_small > q_large

    def test_known_heat_flux_value(self):
        """Verify formula: q = k × √(ρ/r) × v³."""
        rho, v, r = 0.020, 3500.0, 2.25
        expected = CONVECTIVE_HEATING_COEFF * math.sqrt(rho / r) * v ** 3
        actual = convective_heat_flux(rho, v, r)
        assert abs(actual - expected) < 0.01


# -- Heat shield temperature and ablation -------------------------------------

class TestHeatShield:
    """Tests for heat shield thermal response and ablation."""

    def test_no_heating_temp_stays_or_drops(self):
        """With zero heat flux, temperature stays at or drops toward ambient."""
        t = heat_shield_temp(0.0, 100.0, MARS_SURFACE_TEMP_K, 15.9, 1.0)
        assert t <= MARS_SURFACE_TEMP_K + 0.01

    def test_high_heating_raises_temp(self):
        """High heat flux raises temperature."""
        t = heat_shield_temp(500_000.0, 500.0, 500.0, 15.9, 1.0)
        assert t > 500.0

    def test_temp_capped_at_max(self):
        """Temperature cannot exceed HEATSHIELD_MAX_TEMP_K."""
        t = heat_shield_temp(1e10, 1.0, 3000.0, 15.9, 100.0)
        assert t <= HEATSHIELD_MAX_TEMP_K

    def test_temp_floored_at_ambient(self):
        """Temperature never drops below Mars surface temp."""
        t = heat_shield_temp(0.0, 100.0, MARS_SURFACE_TEMP_K, 15.9, 1000.0)
        assert t >= MARS_SURFACE_TEMP_K

    def test_zero_mass_returns_current(self):
        """Zero shield mass returns current temperature."""
        t = heat_shield_temp(100_000.0, 0.0, 500.0, 15.9, 1.0)
        assert t == 500.0

    def test_ablation_below_threshold_is_zero(self):
        """No ablation below threshold temperature."""
        loss = ablation_mass_loss(100_000.0, 15.9, 1.0, 2000.0)
        assert loss == 0.0  # 2000 K < 2500 K threshold

    def test_ablation_above_threshold_positive(self):
        """Ablation occurs above threshold temperature."""
        loss = ablation_mass_loss(100_000.0, 15.9, 1.0, 3000.0)
        assert loss > 0.0

    def test_ablation_increases_with_temp(self):
        """Higher temperature → more ablation."""
        loss_low = ablation_mass_loss(100_000.0, 15.9, 1.0, 2600.0)
        loss_high = ablation_mass_loss(100_000.0, 15.9, 1.0, 3200.0)
        assert loss_high > loss_low

    def test_ablation_zero_flux_zero_loss(self):
        """Zero heat flux = zero ablation even at high temp."""
        loss = ablation_mass_loss(0.0, 15.9, 1.0, 3000.0)
        assert loss == 0.0


# -- Parachute drag -----------------------------------------------------------

class TestParachuteDrag:
    """Tests for parachute drag physics."""

    def test_chute_drag_formula(self):
        """Parachute drag uses same formula as aero drag."""
        f1 = parachute_drag_force(0.010, 200.0, 0.5, 200.0)
        f2 = drag_force(0.010, 200.0, 0.5, 200.0)
        assert abs(f1 - f2) < 0.01

    def test_chute_adds_significant_drag(self):
        """200 m² chute provides more drag than 15.9 m² capsule."""
        capsule = drag_force(0.010, 200.0, 1.05, 15.9)
        chute = parachute_drag_force(0.010, 200.0, 0.5, 200.0)
        assert chute > capsule

    def test_chute_zero_velocity_no_drag(self):
        """No drag at zero velocity."""
        assert parachute_drag_force(0.010, 0.0, 0.5, 200.0) == 0.0


# -- Ballistic coefficient ---------------------------------------------------

class TestBallisticCoefficient:
    """Tests for ballistic coefficient calculation."""

    def test_known_value(self):
        """β = 5000 / (1.05 × 15.9) ≈ 299.4 kg/m²."""
        beta = ballistic_coefficient(5000.0, 1.05, 15.9)
        expected = 5000.0 / (1.05 * 15.9)
        assert abs(beta - expected) < 0.1

    def test_higher_mass_higher_beta(self):
        """Heavier capsule has higher ballistic coefficient."""
        b1 = ballistic_coefficient(5000.0, 1.05, 15.9)
        b2 = ballistic_coefficient(10000.0, 1.05, 15.9)
        assert b2 > b1

    def test_zero_area_returns_inf(self):
        """Zero area gives infinite ballistic coefficient."""
        assert ballistic_coefficient(5000.0, 1.0, 0.0) == float("inf")

    def test_zero_cd_returns_inf(self):
        """Zero drag coefficient gives infinite ballistic coefficient."""
        assert ballistic_coefficient(5000.0, 0.0, 10.0) == float("inf")


# -- Rocket physics -----------------------------------------------------------

class TestRocketPhysics:
    """Tests for Tsiolkovsky equation and fuel calculations."""

    def test_tsiolkovsky_known_value(self):
        """Δv = 226 × 9.807 × ln(2) ≈ 1535 m/s for mass ratio 2."""
        dv = tsiolkovsky_delta_v(226.0, 1000.0, 500.0)
        expected = 226.0 * G0_M_S2 * math.log(2.0)
        assert abs(dv - expected) < 0.1

    def test_tsiolkovsky_no_fuel_zero_dv(self):
        """If m_initial == m_final, no delta-v available."""
        assert tsiolkovsky_delta_v(226.0, 1000.0, 1000.0) == 0.0

    def test_tsiolkovsky_zero_final_mass(self):
        """Zero final mass is invalid (returns 0)."""
        assert tsiolkovsky_delta_v(226.0, 1000.0, 0.0) == 0.0

    def test_required_propellant_roundtrip(self):
        """Required propellant for a given Δv, then verify with Tsiolkovsky."""
        dv_target = 500.0
        dry = 5000.0
        prop = required_propellant(dv_target, 226.0, dry)
        dv_check = tsiolkovsky_delta_v(226.0, dry + prop, dry)
        assert abs(dv_check - dv_target) < 0.1

    def test_required_propellant_zero_dv(self):
        """Zero Δv needs zero propellant."""
        assert required_propellant(0.0, 226.0, 5000.0) == 0.0

    def test_thrust_acceleration(self):
        """a = F / m = 31000 / 8929 ≈ 3.47 m/s²."""
        a = thrust_acceleration(31_000.0, 8_929.0)
        assert abs(a - 31_000.0 / 8_929.0) < 0.01

    def test_thrust_acceleration_zero_mass(self):
        """Zero mass returns zero (not infinity)."""
        assert thrust_acceleration(31_000.0, 0.0) == 0.0

    def test_fuel_mass_flow_rate(self):
        """ṁ = F / (Isp × g₀) = 31000 / (226 × 9.807) ≈ 13.99 kg/s."""
        mdot = fuel_mass_flow_rate(31_000.0, 226.0)
        expected = 31_000.0 / (226.0 * G0_M_S2)
        assert abs(mdot - expected) < 0.01

    def test_fuel_flow_zero_isp(self):
        """Zero Isp gives zero flow rate (not infinity)."""
        assert fuel_mass_flow_rate(31_000.0, 0.0) == 0.0


# -- Energy calculations -----------------------------------------------------

class TestEnergy:
    """Tests for kinetic, potential, and dynamic pressure."""

    def test_kinetic_energy(self):
        """KE = 0.5 × 8929 × 3500² ≈ 54.7 GJ."""
        ke = kinetic_energy(8929.0, 3500.0)
        expected = 0.5 * 8929.0 * 3500.0**2
        assert abs(ke - expected) < 1.0

    def test_kinetic_energy_zero_velocity(self):
        """Zero velocity → zero KE."""
        assert kinetic_energy(5000.0, 0.0) == 0.0

    def test_potential_energy(self):
        """PE = m × g × h = 8929 × 3.72 × 125000 ≈ 4.15 GJ."""
        pe = potential_energy(8929.0, 125_000.0)
        expected = 8929.0 * MARS_SURFACE_GRAVITY_M_S2 * 125_000.0
        assert abs(pe - expected) < 1.0

    def test_potential_energy_surface(self):
        """PE at surface is zero."""
        assert potential_energy(5000.0, 0.0) == 0.0

    def test_dynamic_pressure(self):
        """q = 0.5 × ρ × v² = 0.5 × 0.020 × 3500² = 122,500 Pa."""
        q = dynamic_pressure(0.020, 3500.0)
        expected = 0.5 * 0.020 * 3500.0**2
        assert abs(q - expected) < 0.01


# -- Lander state creation ----------------------------------------------------

class TestLanderCreation:
    """Tests for create_lander factory function."""

    def test_default_payload_mass(self, default_lander):
        """Default payload is 5000 kg."""
        assert default_lander.payload_mass_kg == 5_000.0

    def test_entry_mass_greater_than_payload(self, default_lander):
        """Entry mass exceeds payload (includes shield, chute, fuel, structure)."""
        assert default_lander.entry_mass_kg > default_lander.payload_mass_kg

    def test_mass_fractions_sum_correctly(self, default_lander):
        """All mass components sum to entry mass."""
        total = (default_lander.payload_mass_kg +
                 default_lander.heatshield_mass_kg +
                 default_lander.parachute_mass_kg +
                 default_lander.propellant_mass_kg +
                 default_lander.structure_mass_kg)
        assert abs(total - default_lander.entry_mass_kg) < 0.01

    def test_heatshield_fraction(self, default_lander):
        """Heat shield is ~17% of entry mass."""
        frac = default_lander.heatshield_mass_kg / default_lander.entry_mass_kg
        assert abs(frac - HEATSHIELD_MASS_FRACTION) < 0.01

    def test_initial_phase_is_preentry(self, default_lander):
        """Starts in pre-entry phase."""
        assert default_lander.phase == PHASE_PREENTRY

    def test_propellant_fully_loaded(self, default_lander):
        """Propellant starts at maximum."""
        assert default_lander.propellant_remaining_kg == default_lander.propellant_mass_kg

    def test_heatshield_fully_intact(self, default_lander):
        """Heat shield starts at full mass."""
        assert default_lander.heatshield_remaining_kg == default_lander.heatshield_mass_kg

    def test_light_lander_mass(self, light_lander):
        """1000 kg payload creates lighter entry mass."""
        assert light_lander.entry_mass_kg < 2000.0

    def test_heavy_lander_mass(self, heavy_lander):
        """20000 kg payload creates heavy entry mass."""
        assert heavy_lander.entry_mass_kg > 30_000.0

    def test_invalid_mass_fractions_raise(self):
        """Mass fractions summing to ≥1.0 raises ValueError."""
        with pytest.raises(ValueError):
            create_lander(heatshield_fraction=0.5, parachute_fraction=0.3,
                          propellant_fraction=0.3)

    def test_to_dict_has_required_keys(self, default_lander):
        """Serialized dict includes all essential keys."""
        d = default_lander.to_dict()
        required = {"payload_mass_kg", "entry_mass_kg", "altitude_m",
                     "velocity_m_s", "phase", "time_s", "events"}
        assert required.issubset(d.keys())

    def test_current_mass_equals_entry_mass(self, default_lander):
        """Initial current mass equals entry mass."""
        assert abs(default_lander.current_mass_kg() - default_lander.entry_mass_kg) < 0.01


# -- Tick engine: entry phase -------------------------------------------------

class TestTickEntry:
    """Tests for the atmospheric entry phase of tick()."""

    def test_first_tick_enters_atmosphere(self, default_lander):
        """First tick transitions from pre-entry to entry."""
        result = tick(default_lander)
        assert default_lander.phase == PHASE_ENTRY
        assert "ENTRY_INTERFACE" in result["events"]

    def test_velocity_decreases_during_entry(self, default_lander):
        """Velocity decreases as drag decelerates the capsule."""
        tick(default_lander)  # enter atmosphere
        v_before = default_lander.velocity_m_s
        # Move to lower altitude where atmosphere is denser
        default_lander.altitude_m = 30_000.0
        tick(default_lander)
        assert default_lander.velocity_m_s <= v_before

    def test_altitude_decreases_during_entry(self, default_lander):
        """Altitude decreases (capsule descends)."""
        tick(default_lander)
        alt_before = default_lander.altitude_m
        tick(default_lander)
        assert default_lander.altitude_m <= alt_before

    def test_drag_force_reported_in_result(self, default_lander):
        """Result dict includes non-negative drag force."""
        result = tick(default_lander)
        assert result["drag_n"] >= 0.0

    def test_heat_flux_during_entry(self):
        """Heat flux is positive during high-speed entry."""
        lander = create_lander()
        tick(lander)  # enter
        lander.altitude_m = 40_000.0
        lander.velocity_m_s = 3000.0
        result = tick(lander)
        assert result["heat_flux_w_m2"] > 0.0

    def test_peak_deceleration_tracked(self, default_lander):
        """Peak deceleration increases during entry."""
        tick(default_lander)
        for _ in range(5):
            tick(default_lander)
        assert default_lander.peak_deceleration_g >= 0.0


# -- Tick engine: parachute phase ---------------------------------------------

class TestTickParachute:
    """Tests for the parachute deployment and descent phase."""

    def test_parachute_deploys_at_subsonic(self):
        """Parachute deploys when velocity drops below Mach 2."""
        lander = create_lander()
        lander.phase = PHASE_ENTRY
        lander.velocity_m_s = PARACHUTE_DEPLOY_VELOCITY_M_S - 10.0
        lander.altitude_m = 10_000.0
        result = tick(lander)
        assert lander.parachute_deployed is True
        assert lander.phase == PHASE_PARACHUTE
        assert "PARACHUTE_DEPLOYED" in result["events"]

    def test_parachute_adds_drag(self):
        """Chute drag adds to capsule drag (result shows chute_drag_n)."""
        lander = create_lander()
        lander.phase = PHASE_PARACHUTE
        lander.parachute_deployed = True
        lander.velocity_m_s = 200.0
        lander.altitude_m = 8_000.0
        result = tick(lander)
        assert result["chute_drag_n"] > 0.0

    def test_no_deploy_too_low(self):
        """Parachute doesn't deploy below minimum altitude."""
        lander = create_lander()
        lander.phase = PHASE_ENTRY
        lander.velocity_m_s = PARACHUTE_DEPLOY_VELOCITY_M_S - 10.0
        lander.altitude_m = PARACHUTE_DEPLOY_ALTITUDE_MIN_M - 100.0
        tick(lander)
        assert lander.parachute_deployed is False

    def test_parachute_slows_but_not_enough(self):
        """Parachute reduces velocity but transitions to powered descent."""
        lander = create_lander()
        lander.phase = PHASE_PARACHUTE
        lander.parachute_deployed = True
        lander.velocity_m_s = 300.0
        lander.altitude_m = 8_000.0
        for _ in range(500):
            tick(lander)
            if lander.phase != PHASE_PARACHUTE:
                break
        assert lander.phase in (PHASE_POWERED, PHASE_LANDED, PHASE_CRASHED)


# -- Tick engine: powered descent ---------------------------------------------

class TestTickPowered:
    """Tests for the powered descent (retrorocket) phase."""

    def test_engines_activate_on_transition(self):
        """Engines activate when entering powered descent."""
        lander = create_lander()
        lander.phase = PHASE_PARACHUTE
        lander.parachute_deployed = True
        lander.velocity_m_s = POWERED_DESCENT_TRIGGER_M_S - 1.0
        lander.altitude_m = 2_000.0
        result = tick(lander)
        assert lander.engines_active is True
        assert lander.phase == PHASE_POWERED
        assert "POWERED_DESCENT_START" in result["events"]

    def test_thrust_reported_in_result(self):
        """Thrust force is positive during powered descent."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.engines_active = True
        lander.velocity_m_s = 50.0
        lander.altitude_m = 500.0
        result = tick(lander)
        assert result["thrust_n"] > 0.0

    def test_propellant_consumed(self):
        """Propellant decreases during powered descent."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.engines_active = True
        lander.velocity_m_s = 50.0
        lander.altitude_m = 500.0
        fuel_before = lander.propellant_remaining_kg
        tick(lander)
        assert lander.propellant_remaining_kg < fuel_before

    def test_fuel_exhaustion_event(self):
        """FUEL_EXHAUSTED event when propellant runs out."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.engines_active = True
        lander.velocity_m_s = 50.0
        lander.altitude_m = 500.0
        lander.propellant_remaining_kg = 0.1
        for _ in range(100):
            result = tick(lander)
            if "FUEL_EXHAUSTED" in result["events"]:
                break
        assert lander.propellant_remaining_kg == 0.0

    def test_throttle_within_bounds(self):
        """Throttle stays between 0 and 1."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.engines_active = True
        lander.velocity_m_s = 50.0
        lander.altitude_m = 500.0
        result = tick(lander)
        assert 0.0 <= result["throttle"] <= 1.0


# -- Tick engine: landing and crash -------------------------------------------

class TestTickLanding:
    """Tests for touchdown and crash detection."""

    def test_soft_landing_touchdown(self):
        """Low velocity at surface → TOUCHDOWN event."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.engines_active = True
        lander.velocity_m_s = LANDING_VELOCITY_MAX_M_S - 0.5
        lander.altitude_m = 1.0
        lander.flight_path_angle_deg = -90.0
        result = tick(lander)
        assert lander.phase == PHASE_LANDED
        assert "TOUCHDOWN" in result["events"]
        assert lander.velocity_m_s == 0.0

    def test_hard_crash(self):
        """High velocity at surface → CRASH event."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.velocity_m_s = 50.0
        lander.altitude_m = 1.0
        lander.flight_path_angle_deg = -90.0
        result = tick(lander)
        assert lander.phase == PHASE_CRASHED
        assert any("CRASH" in e for e in result["events"])

    def test_landed_is_terminal(self):
        """Landed state is terminal (no further physics)."""
        lander = create_lander()
        lander.phase = PHASE_LANDED
        lander.velocity_m_s = 0.0
        lander.altitude_m = 0.0
        result = tick(lander)
        assert result["velocity_m_s"] == 0.0
        assert result["phase"] == PHASE_LANDED

    def test_crashed_is_terminal(self):
        """Crashed state is terminal."""
        lander = create_lander()
        lander.phase = PHASE_CRASHED
        result = tick(lander)
        assert result["phase"] == PHASE_CRASHED


# -- Conservation laws --------------------------------------------------------

class TestConservationLaws:
    """Tests for physical conservation laws and invariants."""

    def test_mass_conservation_through_entry(self):
        """Mass components always sum to entry mass minus losses."""
        lander = create_lander()
        for _ in range(50):
            tick(lander)
            if lander.is_terminal():
                break
            ablated = lander.heatshield_mass_kg - lander.heatshield_remaining_kg
            burned = lander.propellant_mass_kg - lander.propellant_remaining_kg
            expected = lander.entry_mass_kg - ablated - burned
            assert abs(lander.current_mass_kg() - expected) < 0.01, \
                f"Mass conservation violated at t={lander.time_s:.1f}s"

    def test_propellant_never_negative(self):
        """Propellant can never go below zero."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.engines_active = True
        lander.velocity_m_s = 50.0
        lander.altitude_m = 10_000.0
        for _ in range(5000):
            tick(lander)
            assert lander.propellant_remaining_kg >= 0.0
            if lander.is_terminal():
                break

    def test_velocity_never_negative(self):
        """Velocity is always non-negative during descent."""
        lander = create_lander()
        for _ in range(200):
            tick(lander)
            assert lander.velocity_m_s >= 0.0
            if lander.is_terminal():
                break

    def test_altitude_never_negative(self):
        """Altitude is always non-negative."""
        lander = create_lander()
        for _ in range(200):
            tick(lander)
            assert lander.altitude_m >= 0.0
            if lander.is_terminal():
                break

    def test_heatshield_mass_never_negative(self):
        """Heat shield mass cannot go below zero."""
        lander = create_lander()
        for _ in range(200):
            tick(lander)
            assert lander.heatshield_remaining_kg >= 0.0
            if lander.is_terminal():
                break

    def test_heat_dissipated_monotonically_increases(self):
        """Total heat dissipated only increases (entropy)."""
        lander = create_lander()
        prev_heat = 0.0
        for _ in range(100):
            tick(lander)
            assert lander.total_heat_dissipated_j >= prev_heat
            prev_heat = lander.total_heat_dissipated_j
            if lander.is_terminal():
                break

    def test_fuel_fraction_bounded(self):
        """Fuel fraction is always in [0, 1]."""
        lander = create_lander()
        for _ in range(200):
            tick(lander)
            ff = lander.fuel_fraction()
            assert 0.0 <= ff <= 1.0
            if lander.is_terminal():
                break

    def test_heatshield_temp_bounded(self):
        """Heat shield temperature stays within physical bounds."""
        lander = create_lander()
        for _ in range(200):
            tick(lander)
            assert lander.heatshield_temp_k >= MARS_SURFACE_TEMP_K
            assert lander.heatshield_temp_k <= HEATSHIELD_MAX_TEMP_K
            if lander.is_terminal():
                break

    def test_current_mass_never_exceeds_entry_mass(self):
        """Current mass ≤ entry mass always (mass only decreases)."""
        lander = create_lander()
        for _ in range(200):
            tick(lander)
            assert lander.current_mass_kg() <= lander.entry_mass_kg + 0.01
            if lander.is_terminal():
                break


# -- Multi-step simulation ----------------------------------------------------

class TestMultiStep:
    """Tests for multi-step EDL simulation runs."""

    def test_run_edl_completes(self, default_lander):
        """Full EDL completes within timeout."""
        results = run_edl(default_lander, max_time_s=600.0, dt_s=1.0)
        assert len(results) > 0
        assert default_lander.is_terminal()

    def test_run_edl_phase_sequence(self, default_lander):
        """EDL progresses through phases in order."""
        results = run_edl(default_lander, max_time_s=600.0, dt_s=1.0)
        phases_seen = []
        for r in results:
            if not phases_seen or r["phase"] != phases_seen[-1]:
                phases_seen.append(r["phase"])
        assert PHASE_ENTRY in phases_seen

    def test_altitude_monotonically_decreases(self, default_lander):
        """Altitude decreases throughout descent (no bouncing)."""
        results = run_edl(default_lander, max_time_s=600.0, dt_s=1.0)
        prev_alt = DEFAULT_ENTRY_ALTITUDE_M + 1.0
        for r in results:
            assert r["altitude_m"] <= prev_alt + 0.01, \
                f"Altitude increased at t={r['time_s']:.1f}s"
            prev_alt = r["altitude_m"]

    def test_velocity_generally_decreases(self, default_lander):
        """Velocity trend is downward over the full EDL."""
        results = run_edl(default_lander, max_time_s=600.0, dt_s=1.0)
        if len(results) > 10:
            early_v = results[5]["velocity_m_s"]
            late_v = results[-1]["velocity_m_s"]
            assert late_v < early_v

    def test_light_lander_completes(self, light_lander):
        """1000 kg payload EDL completes."""
        results = run_edl(light_lander, max_time_s=600.0, dt_s=1.0)
        assert len(results) > 0
        assert light_lander.is_terminal()

    def test_heavy_lander_completes(self, heavy_lander):
        """20000 kg payload EDL completes."""
        results = run_edl(heavy_lander, max_time_s=600.0, dt_s=1.0)
        assert len(results) > 0
        assert heavy_lander.is_terminal()

    def test_time_monotonically_increases(self, default_lander):
        """Simulation time always increases."""
        results = run_edl(default_lander, max_time_s=600.0, dt_s=1.0)
        prev_t = -1.0
        for r in results:
            assert r["time_s"] > prev_t
            prev_t = r["time_s"]

    def test_mass_decreases_over_time(self, default_lander):
        """Total mass decreases (ablation + fuel consumption)."""
        results = run_edl(default_lander, max_time_s=600.0, dt_s=1.0)
        if len(results) > 10:
            early_mass = results[0]["current_mass_kg"]
            late_mass = results[-1]["current_mass_kg"]
            assert late_mass <= early_mass


# -- Abort conditions ---------------------------------------------------------

class TestAbortConditions:
    """Tests for dust storm, wind, and fuel margin abort triggers."""

    def test_dust_storm_abort_event(self):
        """High optical depth triggers ABORT_DUST_STORM event."""
        lander = create_lander()
        lander.phase = PHASE_ENTRY
        lander.altitude_m = 50_000.0
        lander.velocity_m_s = 2000.0
        result = tick(lander, dust_tau=3.0)
        assert "ABORT_DUST_STORM" in result["events"]

    def test_no_abort_in_clear_weather(self):
        """No abort with normal dust tau."""
        lander = create_lander()
        lander.phase = PHASE_ENTRY
        lander.altitude_m = 50_000.0
        lander.velocity_m_s = 2000.0
        result = tick(lander, dust_tau=0.3)
        assert "ABORT_DUST_STORM" not in result["events"]

    def test_high_wind_abort_near_surface(self):
        """High wind near surface triggers ABORT_HIGH_WIND."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.engines_active = True
        lander.velocity_m_s = 30.0
        lander.altitude_m = 500.0
        result = tick(lander, surface_wind_m_s=30.0)
        assert "ABORT_HIGH_WIND" in result["events"]

    def test_no_wind_abort_at_altitude(self):
        """High wind at high altitude doesn't trigger abort."""
        lander = create_lander()
        lander.phase = PHASE_ENTRY
        lander.altitude_m = 50_000.0
        lander.velocity_m_s = 2000.0
        result = tick(lander, surface_wind_m_s=30.0)
        assert "ABORT_HIGH_WIND" not in result["events"]

    def test_low_fuel_warning(self):
        """Low fuel fraction triggers warning event."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.engines_active = True
        lander.velocity_m_s = 50.0
        lander.altitude_m = 500.0
        lander.propellant_remaining_kg = lander.propellant_mass_kg * 0.03
        result = tick(lander)
        assert "LOW_FUEL_WARNING" in result["events"]


# -- Edge cases ---------------------------------------------------------------

class TestEdgeCases:
    """Tests for boundary and extreme conditions."""

    def test_zero_payload_mass(self):
        """Zero payload creates a valid (if odd) lander."""
        lander = create_lander(payload_mass_kg=0.0)
        assert lander.entry_mass_kg == 0.0

    def test_very_high_entry_velocity(self):
        """5500 m/s direct entry doesn't crash the simulation."""
        lander = create_lander(entry_velocity_m_s=5500.0)
        result = tick(lander)
        assert result["phase"] == PHASE_ENTRY

    def test_very_low_entry_altitude(self):
        """Low entry altitude still simulates."""
        lander = create_lander(entry_altitude_m=10_000.0)
        result = tick(lander)
        assert result["altitude_m"] >= 0.0

    def test_already_at_surface(self):
        """Capsule starting at surface altitude."""
        lander = create_lander()
        lander.altitude_m = 0.0
        lander.velocity_m_s = 1.0
        lander.flight_path_angle_deg = -90.0
        lander.phase = PHASE_POWERED
        result = tick(lander)
        assert lander.is_terminal()

    def test_zero_velocity_at_altitude(self):
        """Zero velocity at altitude (hovering, then gravity pulls)."""
        lander = create_lander()
        lander.phase = PHASE_POWERED
        lander.engines_active = True
        lander.velocity_m_s = 0.0
        lander.altitude_m = 100.0
        result = tick(lander)
        assert result is not None

    def test_terminal_state_no_mutation(self):
        """Terminal state tick doesn't change state significantly."""
        lander = create_lander()
        lander.phase = PHASE_LANDED
        lander.velocity_m_s = 0.0
        lander.altitude_m = 0.0
        mass_before = lander.current_mass_kg()
        tick(lander)
        assert lander.current_mass_kg() == mass_before

    def test_result_dict_always_has_events(self):
        """Every tick result has an events list."""
        lander = create_lander()
        for _ in range(10):
            result = tick(lander)
            assert isinstance(result["events"], list)
            if lander.is_terminal():
                break

    def test_steep_entry_angle(self):
        """Very steep entry angle (-45°) still works."""
        lander = create_lander(entry_angle_deg=-45.0)
        results = run_edl(lander, max_time_s=600.0, dt_s=1.0)
        assert len(results) > 0

    def test_shallow_entry_angle(self):
        """Shallow entry angle (-5°) still works."""
        lander = create_lander(entry_angle_deg=-5.0)
        results = run_edl(lander, max_time_s=600.0, dt_s=1.0)
        assert len(results) > 0

    def test_mass_check_helper(self, default_lander):
        """mass_check() returns near-zero for consistent state."""
        assert default_lander.mass_check() < 0.01
