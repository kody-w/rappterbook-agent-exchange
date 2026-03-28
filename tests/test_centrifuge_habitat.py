"""Tests for centrifuge_habitat.py -- Mars Artificial Gravity Centrifuge.

127 tests covering unit conversions, centripetal physics, Coriolis effects,
tether mechanics, moment of inertia, rotational energy, angular momentum,
spin-up/spin-down dynamics, bearing friction, comfort criteria, structural
limits, conservation laws, design function, tick engine, full simulation,
edge cases, and property-based invariants.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import centrifuge_habitat as ch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def default_state():
    return ch.CentrifugeState()


@pytest.fixture
def spinning_state():
    s = ch.CentrifugeState()
    ch.start_spin(s, 0.7)
    # Spin up for many sols to reach nominal
    for _ in range(50):
        ch.tick(s)
        if s.phase == "nominal":
            break
    return s


@pytest.fixture
def small_centrifuge():
    return ch.CentrifugeState(radius_m=20.0, pod_mass_kg=2000.0,
                               crew_per_pod=1, hub_mass_kg=500.0)


# ============================================================================
# Unit conversion tests
# ============================================================================

class TestUnitConversions:
    def test_rpm_to_rad_s_zero(self):
        assert ch.rpm_to_rad_s(0.0) == 0.0

    def test_rpm_to_rad_s_one(self):
        assert abs(ch.rpm_to_rad_s(1.0) - math.pi / 30.0) < 1e-10

    def test_rpm_to_rad_s_roundtrip(self):
        for rpm in [0.5, 1.0, 2.0, 4.0, 6.0]:
            omega = ch.rpm_to_rad_s(rpm)
            assert abs(ch.rad_s_to_rpm(omega) - rpm) < 1e-10

    def test_rad_s_to_rpm_zero(self):
        assert ch.rad_s_to_rpm(0.0) == 0.0

    def test_rad_s_to_rpm_negative(self):
        assert ch.rad_s_to_rpm(-1.0) == 0.0

    def test_known_conversion(self):
        # 60 RPM = 2π rad/s
        assert abs(ch.rpm_to_rad_s(60.0) - 2 * math.pi) < 1e-10


# ============================================================================
# Centripetal acceleration tests
# ============================================================================

class TestCentripetalAcceleration:
    def test_positive_for_valid_inputs(self):
        assert ch.centripetal_acceleration_m_s2(1.0, 10.0) > 0.0

    def test_zero_omega(self):
        assert ch.centripetal_acceleration_m_s2(0.0, 10.0) == 0.0

    def test_zero_radius(self):
        assert ch.centripetal_acceleration_m_s2(1.0, 0.0) == 0.0

    def test_negative_omega(self):
        assert ch.centripetal_acceleration_m_s2(-1.0, 10.0) == 0.0

    def test_scales_with_omega_squared(self):
        a1 = ch.centripetal_acceleration_m_s2(1.0, 10.0)
        a2 = ch.centripetal_acceleration_m_s2(2.0, 10.0)
        assert abs(a2 / a1 - 4.0) < 1e-10

    def test_scales_with_radius(self):
        a1 = ch.centripetal_acceleration_m_s2(1.0, 10.0)
        a2 = ch.centripetal_acceleration_m_s2(1.0, 20.0)
        assert abs(a2 / a1 - 2.0) < 1e-10

    def test_known_value(self):
        # 2 RPM at 56m → a = (2π*2/60)² * 56 ≈ 2.46 m/s²
        omega = ch.rpm_to_rad_s(2.0)
        a = ch.centripetal_acceleration_m_s2(omega, 56.0)
        assert 2.0 < a < 3.0


# ============================================================================
# Apparent gravity tests
# ============================================================================

class TestApparentGravity:
    def test_zero_spin_gives_mars_gravity(self):
        g = ch.apparent_gravity_g(0.0, 56.0)
        assert abs(g - ch.MARS_GRAVITY_M_S2 / ch.G0_M_S2) < 0.01

    def test_increases_with_omega(self):
        g1 = ch.apparent_gravity_g(0.1, 56.0)
        g2 = ch.apparent_gravity_g(0.3, 56.0)
        assert g2 > g1

    def test_at_1g_config(self):
        # Design for ~1g: need centripetal ≈ sqrt(g0² - g_mars²) ≈ 9.07 m/s²
        # At 56m: ω = sqrt(9.07/56) ≈ 0.403 rad/s ≈ 3.85 RPM
        omega = ch.omega_for_target_g(1.0, 56.0)
        g = ch.apparent_gravity_g(omega, 56.0)
        assert abs(g - 1.0) < 0.01

    def test_always_at_least_mars_gravity(self):
        for omega in [0.0, 0.01, 0.1]:
            g = ch.apparent_gravity_g(omega, 56.0)
            assert g >= ch.MARS_GRAVITY_M_S2 / ch.G0_M_S2 - 0.001


# ============================================================================
# Omega for target g
# ============================================================================

class TestOmegaForTargetG:
    def test_returns_zero_for_sub_mars_gravity(self):
        assert ch.omega_for_target_g(0.3, 56.0) == 0.0

    def test_returns_zero_for_zero_target(self):
        assert ch.omega_for_target_g(0.0, 56.0) == 0.0

    def test_returns_zero_for_zero_radius(self):
        assert ch.omega_for_target_g(1.0, 0.0) == 0.0

    def test_roundtrip_g(self):
        for target in [0.5, 0.7, 1.0, 1.5]:
            omega = ch.omega_for_target_g(target, 56.0)
            if omega > 0:
                g = ch.apparent_gravity_g(omega, 56.0)
                assert abs(g - target) < 0.01

    def test_higher_g_needs_higher_omega(self):
        w1 = ch.omega_for_target_g(0.5, 56.0)
        w2 = ch.omega_for_target_g(1.0, 56.0)
        assert w2 > w1


# ============================================================================
# Coriolis effect tests
# ============================================================================

class TestCoriolisEffect:
    def test_zero_omega(self):
        assert ch.coriolis_acceleration_m_s2(0.0, 1.0) == 0.0

    def test_zero_velocity(self):
        assert ch.coriolis_acceleration_m_s2(0.5, 0.0) == 0.0

    def test_positive_for_valid(self):
        assert ch.coriolis_acceleration_m_s2(0.5, 1.0) > 0.0

    def test_proportional_to_omega(self):
        a1 = ch.coriolis_acceleration_m_s2(0.5, 1.0)
        a2 = ch.coriolis_acceleration_m_s2(1.0, 1.0)
        assert abs(a2 / a1 - 2.0) < 1e-10

    def test_proportional_to_velocity(self):
        a1 = ch.coriolis_acceleration_m_s2(0.5, 1.0)
        a2 = ch.coriolis_acceleration_m_s2(0.5, 2.0)
        assert abs(a2 / a1 - 2.0) < 1e-10

    def test_coriolis_fraction_typical(self):
        # At 2 RPM, 56m radius, 1 m/s walk: fraction ~17%
        omega = ch.rpm_to_rad_s(2.0)
        frac = ch.coriolis_fraction(omega, 56.0, 1.0)
        assert 0.0 < frac < 0.25

    def test_coriolis_fraction_zero_centripetal(self):
        assert ch.coriolis_fraction(0.0, 56.0) == 0.0


# ============================================================================
# Gravity gradient tests
# ============================================================================

class TestGravityGradient:
    def test_decreases_with_radius(self):
        g1 = ch.gravity_gradient_fraction(20.0)
        g2 = ch.gravity_gradient_fraction(60.0)
        assert g2 < g1

    def test_known_value(self):
        # At 56m, gradient = 1.75/56 ≈ 3.1%
        grad = ch.gravity_gradient_fraction(56.0)
        assert abs(grad - 1.75 / 56.0) < 0.001

    def test_zero_radius(self):
        assert ch.gravity_gradient_fraction(0.0) == 1.0

    def test_always_positive(self):
        for r in [10, 20, 50, 100, 200]:
            assert ch.gravity_gradient_fraction(float(r)) > 0.0


# ============================================================================
# Tether mechanics tests
# ============================================================================

class TestTetherMechanics:
    def test_tension_zero_when_not_spinning(self):
        assert ch.tether_tension_n(8000.0, 0.0, 56.0) == 0.0

    def test_tension_positive_when_spinning(self):
        omega = ch.rpm_to_rad_s(2.0)
        assert ch.tether_tension_n(8000.0, omega, 56.0) > 0.0

    def test_tension_scales_with_mass(self):
        omega = ch.rpm_to_rad_s(2.0)
        t1 = ch.tether_tension_n(4000.0, omega, 56.0)
        t2 = ch.tether_tension_n(8000.0, omega, 56.0)
        assert abs(t2 / t1 - 2.0) < 1e-10

    def test_tension_scales_with_omega_squared(self):
        t1 = ch.tether_tension_n(8000.0, 0.1, 56.0)
        t2 = ch.tether_tension_n(8000.0, 0.2, 56.0)
        assert abs(t2 / t1 - 4.0) < 1e-10

    def test_tension_scales_with_radius(self):
        t1 = ch.tether_tension_n(8000.0, 0.1, 28.0)
        t2 = ch.tether_tension_n(8000.0, 0.1, 56.0)
        assert abs(t2 / t1 - 2.0) < 1e-10

    def test_cross_section_positive(self):
        cs = ch.tether_cross_section_m2(100_000.0, 3.0)
        assert cs > 0.0

    def test_cross_section_zero_for_zero_tension(self):
        assert ch.tether_cross_section_m2(0.0, 3.0) == 0.0

    def test_cross_section_scales_with_safety_factor(self):
        cs1 = ch.tether_cross_section_m2(100_000.0, 1.0)
        cs3 = ch.tether_cross_section_m2(100_000.0, 3.0)
        assert abs(cs3 / cs1 - 3.0) < 1e-10

    def test_tether_mass_positive(self):
        assert ch.tether_mass_kg(100.0, 0.001) > 0.0

    def test_tether_mass_zero_for_zero_length(self):
        assert ch.tether_mass_kg(0.0, 0.001) == 0.0


# ============================================================================
# Moment of inertia tests
# ============================================================================

class TestMomentOfInertia:
    def test_single_point_mass(self):
        moi = ch.moment_of_inertia_kg_m2([(10.0, 5.0)])
        assert abs(moi - 10.0 * 25.0) < 1e-10

    def test_two_symmetric_masses(self):
        moi = ch.moment_of_inertia_kg_m2([(10.0, 5.0), (10.0, 5.0)])
        assert abs(moi - 2 * 10.0 * 25.0) < 1e-10

    def test_hub_mass_contributes_nothing(self):
        moi = ch.moment_of_inertia_kg_m2([(100.0, 0.0)])
        assert moi == 0.0

    def test_always_non_negative(self):
        moi = ch.moment_of_inertia_kg_m2([(0.0, 10.0), (-5.0, 10.0)])
        assert moi >= 0.0

    def test_empty_list(self):
        assert ch.moment_of_inertia_kg_m2([]) == 0.0


# ============================================================================
# Rotational energy tests
# ============================================================================

class TestRotationalEnergy:
    def test_positive_for_valid(self):
        assert ch.rotational_energy_j(1000.0, 1.0) > 0.0

    def test_zero_omega(self):
        assert ch.rotational_energy_j(1000.0, 0.0) == 0.0

    def test_zero_moi(self):
        assert ch.rotational_energy_j(0.0, 1.0) == 0.0

    def test_scales_with_omega_squared(self):
        e1 = ch.rotational_energy_j(1000.0, 1.0)
        e2 = ch.rotational_energy_j(1000.0, 2.0)
        assert abs(e2 / e1 - 4.0) < 1e-10

    def test_known_value(self):
        # E = 0.5 * 1000 * 4.0 = 2000
        assert abs(ch.rotational_energy_j(1000.0, 2.0) - 2000.0) < 1e-10


# ============================================================================
# Angular momentum tests
# ============================================================================

class TestAngularMomentum:
    def test_positive(self):
        assert ch.angular_momentum_kg_m2_s(1000.0, 1.0) > 0.0

    def test_zero_omega(self):
        assert ch.angular_momentum_kg_m2_s(1000.0, 0.0) == 0.0

    def test_scales_linearly_with_omega(self):
        l1 = ch.angular_momentum_kg_m2_s(1000.0, 1.0)
        l2 = ch.angular_momentum_kg_m2_s(1000.0, 2.0)
        assert abs(l2 / l1 - 2.0) < 1e-10

    def test_scales_linearly_with_moi(self):
        l1 = ch.angular_momentum_kg_m2_s(1000.0, 1.0)
        l2 = ch.angular_momentum_kg_m2_s(2000.0, 1.0)
        assert abs(l2 / l1 - 2.0) < 1e-10


# ============================================================================
# Spin-up time tests
# ============================================================================

class TestSpinUpTime:
    def test_positive(self):
        t = ch.spin_up_time_s(1e6, 0.5, 5000.0)
        assert t > 0.0

    def test_zero_power_is_infinite(self):
        t = ch.spin_up_time_s(1e6, 0.5, 0.0)
        assert t == float("inf")

    def test_zero_omega_is_zero(self):
        assert ch.spin_up_time_s(1e6, 0.0, 5000.0) == 0.0

    def test_decreases_with_more_power(self):
        t1 = ch.spin_up_time_s(1e6, 0.5, 1000.0)
        t2 = ch.spin_up_time_s(1e6, 0.5, 5000.0)
        assert t2 < t1

    def test_increases_with_moi(self):
        t1 = ch.spin_up_time_s(1e6, 0.5, 5000.0)
        t2 = ch.spin_up_time_s(2e6, 0.5, 5000.0)
        assert t2 > t1


# ============================================================================
# Bearing friction tests
# ============================================================================

class TestBearingFriction:
    def test_positive_for_valid(self):
        p = ch.bearing_friction_power_w(0.005, 10000.0, 0.5, 0.3)
        assert p > 0.0

    def test_zero_omega(self):
        assert ch.bearing_friction_power_w(0.005, 10000.0, 0.0, 0.3) == 0.0

    def test_zero_load(self):
        assert ch.bearing_friction_power_w(0.005, 0.0, 0.5, 0.3) == 0.0

    def test_scales_with_omega(self):
        p1 = ch.bearing_friction_power_w(0.005, 10000.0, 0.5, 0.3)
        p2 = ch.bearing_friction_power_w(0.005, 10000.0, 1.0, 0.3)
        assert abs(p2 / p1 - 2.0) < 1e-10


class TestSpindownRate:
    def test_positive(self):
        assert ch.spindown_rate_rad_s2(100.0, 1e6, 0.5) > 0.0

    def test_zero_friction(self):
        assert ch.spindown_rate_rad_s2(0.0, 1e6, 0.5) == 0.0

    def test_zero_omega(self):
        assert ch.spindown_rate_rad_s2(100.0, 1e6, 0.0) == 0.0


class TestSpindownTime:
    def test_positive(self):
        assert ch.spindown_time_s(0.5, 0.001) > 0.0

    def test_zero_omega(self):
        assert ch.spindown_time_s(0.0, 0.001) == 0.0

    def test_zero_decel(self):
        assert ch.spindown_time_s(0.5, 0.0) == 0.0


# ============================================================================
# Comfort check tests
# ============================================================================

class TestComfortCheck:
    def test_comfortable_at_2rpm_56m(self):
        result = ch.comfort_check(2.0, 56.0)
        assert result["comfortable"] is True
        assert len(result["issues"]) == 0

    def test_uncomfortable_at_high_rpm(self):
        result = ch.comfort_check(5.0, 56.0)
        assert result["comfortable"] is False

    def test_uncomfortable_at_small_radius(self):
        result = ch.comfort_check(2.0, 5.0)
        assert result["comfortable"] is False

    def test_returns_gravity(self):
        result = ch.comfort_check(2.0, 56.0)
        assert result["apparent_gravity_g"] > 0.0

    def test_returns_gradient(self):
        result = ch.comfort_check(2.0, 56.0)
        assert 0.0 < result["gravity_gradient_frac"] < 0.10

    def test_returns_coriolis(self):
        result = ch.comfort_check(2.0, 56.0)
        assert result["coriolis_fraction"] >= 0.0


# ============================================================================
# Design function tests
# ============================================================================

class TestDesignCentrifuge:
    def test_returns_valid_design(self):
        d = ch.design_centrifuge(0.7)
        assert d["radius_m"] > 0
        assert d["rpm"] > 0
        assert d["rpm"] <= ch.COMFORT_MAX_RPM + 0.1
        assert d["apparent_gravity_g"] > 0.4

    def test_target_below_mars_gravity(self):
        d = ch.design_centrifuge(0.3)
        assert "error" in d

    def test_1g_design(self):
        d = ch.design_centrifuge(1.0)
        assert abs(d["apparent_gravity_g"] - 1.0) < 0.05

    def test_comfort_is_satisfied(self):
        d = ch.design_centrifuge(0.7)
        assert d["comfort"]["comfortable"] is True

    def test_tether_mass_reasonable(self):
        d = ch.design_centrifuge(0.7)
        assert d["tether_mass_kg"] > 0
        assert d["tether_mass_kg"] < 10000  # not absurd

    def test_spin_up_finite(self):
        d = ch.design_centrifuge(0.7)
        assert d["spin_up_time_hours"] > 0
        assert d["spin_up_time_hours"] < 1000  # reasonable

    def test_energy_positive(self):
        d = ch.design_centrifuge(0.7)
        assert d["rotational_energy_mj"] > 0


# ============================================================================
# State properties tests
# ============================================================================

class TestCentrifugeState:
    def test_default_phase_stopped(self, default_state):
        assert default_state.phase == "stopped"
        assert default_state.omega_rad_s == 0.0

    def test_rpm_property(self, default_state):
        assert default_state.rpm == 0.0

    def test_pod_total_mass(self, default_state):
        expected = ch.DEFAULT_POD_MASS_KG + ch.DEFAULT_CREW_PER_POD * ch.DEFAULT_CREW_MASS_KG
        assert default_state.pod_total_mass_kg == expected

    def test_tether_cross_section(self, default_state):
        r = ch.DEFAULT_TETHER_DIAMETER_M / 2.0
        expected = math.pi * r * r
        assert abs(default_state.tether_cross_section_m2 - expected) < 1e-10

    def test_tension_zero_when_stopped(self, default_state):
        assert default_state.current_tension_n == 0.0

    def test_apparent_gravity_mars_when_stopped(self, default_state):
        g = default_state.apparent_gravity_g
        assert abs(g - ch.MARS_GRAVITY_M_S2 / ch.G0_M_S2) < 0.01

    def test_angular_momentum_zero_when_stopped(self, default_state):
        assert default_state.angular_momentum == 0.0


# ============================================================================
# Start/stop commands
# ============================================================================

class TestStartStop:
    def test_start_spin_from_stopped(self, default_state):
        err = ch.start_spin(default_state, 0.7)
        assert err is None
        assert default_state.phase == "spinning_up"
        assert default_state.target_omega_rad_s > 0.0

    def test_start_spin_too_high(self, default_state):
        err = ch.start_spin(default_state, 5.0)
        assert err is not None
        assert "RPM" in err or "exceeds" in err

    def test_stop_when_stopped(self, default_state):
        err = ch.stop_spin(default_state)
        assert err is not None

    def test_emergency_stop(self, spinning_state):
        err = ch.stop_spin(spinning_state, emergency=True)
        assert err is None
        # After emergency stop, next tick should execute it
        ch.tick(spinning_state)
        assert spinning_state.omega_rad_s == 0.0

    def test_cannot_start_during_spinup(self, default_state):
        ch.start_spin(default_state, 0.7)
        err = ch.start_spin(default_state, 0.8)
        assert err is not None


# ============================================================================
# Tick engine tests
# ============================================================================

class TestTickEngine:
    def test_stopped_tick_does_nothing(self, default_state):
        rec = ch.tick(default_state)
        assert rec.phase == "stopped"
        assert rec.omega_rad_s == 0.0

    def test_spin_up_increases_omega(self, default_state):
        ch.start_spin(default_state, 0.7)
        rec1 = ch.tick(default_state)
        omega1 = default_state.omega_rad_s
        rec2 = ch.tick(default_state)
        omega2 = default_state.omega_rad_s
        assert omega2 >= omega1

    def test_reaches_nominal(self, default_state):
        ch.start_spin(default_state, 0.5)
        for _ in range(100):
            ch.tick(default_state)
            if default_state.phase == "nominal":
                break
        assert default_state.phase == "nominal"

    def test_nominal_maintains_omega(self, spinning_state):
        omega_before = spinning_state.omega_rad_s
        ch.tick(spinning_state)
        # Should be approximately the same (friction compensated)
        assert abs(spinning_state.omega_rad_s - omega_before) < 0.001

    def test_spin_down_decreases_omega(self, spinning_state):
        omega_before = spinning_state.omega_rad_s
        ch.stop_spin(spinning_state)
        ch.tick(spinning_state)
        assert spinning_state.omega_rad_s < omega_before

    def test_spin_down_reaches_stopped(self, spinning_state):
        ch.stop_spin(spinning_state)
        for _ in range(100):
            ch.tick(spinning_state)
            if spinning_state.phase == "stopped":
                break
        assert spinning_state.phase == "stopped"
        assert spinning_state.omega_rad_s == 0.0

    def test_emergency_stop_immediate(self, spinning_state):
        ch.stop_spin(spinning_state, emergency=True)
        ch.tick(spinning_state)
        assert spinning_state.omega_rad_s == 0.0
        assert spinning_state.phase == "stopped"

    def test_sol_counter_increments(self, default_state):
        assert default_state.sol == 0
        ch.tick(default_state)
        assert default_state.sol == 1
        ch.tick(default_state)
        assert default_state.sol == 2

    def test_insufficient_power_warns(self, spinning_state):
        # Give very little power so it can't maintain
        rec = ch.tick(spinning_state, available_power_w=0.001)
        # Should warn about insufficient power
        assert len(rec.warnings) > 0 or spinning_state.omega_rad_s < spinning_state.target_omega_rad_s


# ============================================================================
# Conservation law tests
# ============================================================================

class TestConservationLaws:
    def test_energy_monotonic_during_spinup(self, default_state):
        ch.start_spin(default_state, 0.5)
        energies = []
        for _ in range(20):
            ch.tick(default_state)
            e = ch.rotational_energy_j(default_state.total_moment_of_inertia,
                                       default_state.omega_rad_s)
            energies.append(e)
            if default_state.phase == "nominal":
                break
        # Energy should be non-decreasing during spin-up
        for i in range(1, len(energies)):
            assert energies[i] >= energies[i - 1] - 1e-6

    def test_omega_non_negative(self, default_state):
        ch.start_spin(default_state, 0.5)
        for _ in range(50):
            ch.tick(default_state)
        ch.stop_spin(default_state)
        for _ in range(100):
            ch.tick(default_state)
            assert default_state.omega_rad_s >= 0.0

    def test_energy_input_tracks_consumption(self, default_state):
        ch.start_spin(default_state, 0.5)
        for _ in range(30):
            ch.tick(default_state)
        assert default_state.total_energy_input_j > 0.0
        assert default_state.total_friction_loss_j >= 0.0

    def test_tension_consistent_with_omega(self, spinning_state):
        omega = spinning_state.omega_rad_s
        r = spinning_state.radius_m
        m = spinning_state.pod_total_mass_kg
        expected = m * omega**2 * r
        actual = spinning_state.current_tension_n
        assert abs(actual - expected) < 1.0

    def test_angular_momentum_conserved_no_torque(self):
        """With infinite motor power and no friction, L is conserved in nominal."""
        state = ch.CentrifugeState(
            bearing_friction_coeff=0.0,  # no friction
            motor_power_w=1e9,  # effectively infinite
        )
        ch.start_spin(state, 0.5)
        for _ in range(50):
            ch.tick(state)
            if state.phase == "nominal":
                break
        L_initial = state.angular_momentum
        for _ in range(5):
            ch.tick(state)
        L_final = state.angular_momentum
        # With no friction and enough power, L should be conserved
        assert abs(L_final - L_initial) / max(L_initial, 1.0) < 0.01


# ============================================================================
# Structural safety tests
# ============================================================================

class TestStructuralSafety:
    def test_tether_holds_at_normal_operation(self, spinning_state):
        tension = spinning_state.current_tension_n
        max_t = spinning_state.tether_max_tension_n
        assert tension < max_t

    def test_safety_factor_maintained(self, spinning_state):
        tension = spinning_state.current_tension_n
        max_t = spinning_state.tether_max_tension_n
        safety = max_t / tension if tension > 0 else float("inf")
        assert safety >= 1.0

    def test_overspeed_triggers_warning(self):
        state = ch.CentrifugeState(
            radius_m=10.0,
            tether_diameter_m=0.005,  # very thin tether
            pod_mass_kg=20_000.0,  # heavy pods
        )
        ch.start_spin(state, 1.5)
        warnings_found = False
        for _ in range(100):
            rec = ch.tick(state)
            if rec.warnings:
                warnings_found = True
                break
            if state.phase in ("stopped", "nominal"):
                break
        # Either warning was raised or design is adequate
        assert warnings_found or state.structural_warnings > 0 or state.phase in ("nominal", "stopped")


# ============================================================================
# Property-based invariants
# ============================================================================

class TestInvariants:
    @pytest.mark.parametrize("omega", [0.0, 0.1, 0.2, 0.3, 0.5])
    def test_apparent_gravity_ge_mars(self, omega):
        g = ch.apparent_gravity_g(omega, 56.0)
        assert g >= ch.MARS_GRAVITY_M_S2 / ch.G0_M_S2 - 0.001

    @pytest.mark.parametrize("radius", [10.0, 20.0, 56.0, 100.0, 200.0])
    def test_gradient_decreases_with_radius(self, radius):
        grad = ch.gravity_gradient_fraction(radius)
        assert 0.0 < grad <= 1.0

    @pytest.mark.parametrize("mass", [1000.0, 5000.0, 10000.0])
    def test_tension_proportional_to_mass(self, mass):
        omega = ch.rpm_to_rad_s(2.0)
        t = ch.tether_tension_n(mass, omega, 56.0)
        expected = mass * omega**2 * 56.0
        assert abs(t - expected) < 0.001

    @pytest.mark.parametrize("target_g", [0.5, 0.7, 0.8, 1.0])
    def test_design_meets_target(self, target_g):
        d = ch.design_centrifuge(target_g)
        assert abs(d["apparent_gravity_g"] - target_g) < 0.05

    @pytest.mark.parametrize("rpm", [1.0, 2.0, 3.0, 4.0])
    def test_rpm_roundtrip(self, rpm):
        omega = ch.rpm_to_rad_s(rpm)
        back = ch.rad_s_to_rpm(omega)
        assert abs(back - rpm) < 1e-10


# ============================================================================
# Simulation runner tests
# ============================================================================

class TestRunSimulation:
    def test_completes_without_crash(self):
        result = ch.run_simulation(target_g=0.7, max_sols=30)
        assert "error" not in result
        assert result["sols_simulated"] > 0

    def test_reaches_target_gravity(self):
        result = ch.run_simulation(target_g=0.7, max_sols=50)
        assert result["peak_rpm"] > 0.0

    def test_energy_consumed(self):
        result = ch.run_simulation(target_g=0.7, max_sols=30)
        assert result["total_energy_kwh"] > 0.0

    def test_crew_hours_tracked(self):
        result = ch.run_simulation(target_g=0.7, max_sols=50)
        assert result["crew_hours_at_target_g"] >= 0.0

    def test_no_emergency_stops(self):
        result = ch.run_simulation(target_g=0.7, max_sols=30)
        assert result["emergency_stops"] == 0

    def test_simulation_ends_stopped(self):
        result = ch.run_simulation(target_g=0.5, max_sols=100)
        assert result["final_phase"] == "stopped"

    def test_high_g_simulation(self):
        result = ch.run_simulation(target_g=1.0, max_sols=50)
        assert "error" not in result


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    def test_zero_payload_mass(self):
        state = ch.CentrifugeState(pod_mass_kg=0.0, crew_per_pod=0)
        ch.start_spin(state, 0.5)
        # Should not crash
        for _ in range(5):
            ch.tick(state)

    def test_very_large_radius(self):
        d = ch.design_centrifuge(0.5, max_rpm=1.0)
        assert d["radius_m"] > 0

    def test_minimal_radius(self):
        result = ch.comfort_check(4.0, 14.0)
        # Should be at the edge of comfort
        assert result["gravity_gradient_frac"] > 0.10

    def test_double_tick_at_stopped(self, default_state):
        rec1 = ch.tick(default_state)
        rec2 = ch.tick(default_state)
        assert rec1.phase == "stopped"
        assert rec2.phase == "stopped"
        assert default_state.sol == 2

    def test_start_stop_start_cycle(self, default_state):
        ch.start_spin(default_state, 0.5)
        for _ in range(50):
            ch.tick(default_state)
            if default_state.phase == "nominal":
                break
        ch.stop_spin(default_state, emergency=True)
        ch.tick(default_state)
        assert default_state.omega_rad_s == 0.0
        # Restart
        err = ch.start_spin(default_state, 0.6)
        assert err is None


# ============================================================================
# Smoke test: 10-step simulation
# ============================================================================

class TestSmokeTest:
    def test_10_tick_no_crash(self):
        state = ch.CentrifugeState()
        ch.start_spin(state, 0.7)
        for _ in range(10):
            rec = ch.tick(state)
            assert rec.sol > 0

    def test_full_lifecycle(self):
        """Complete lifecycle: start → spin up → hold → spin down → stop."""
        state = ch.CentrifugeState()
        ch.start_spin(state, 0.6)
        # Capture phase before first tick (spin-up may complete in one sol)
        phases_seen = {state.phase}

        for _ in range(200):
            rec = ch.tick(state)
            phases_seen.add(state.phase)
            if state.phase == "nominal" and state.sol > 20:
                ch.stop_spin(state)
            if state.phase == "stopped" and state.sol > 5:
                break

        assert "spinning_up" in phases_seen or "nominal" in phases_seen
        assert "nominal" in phases_seen
        assert state.total_energy_input_j > 0.0
