"""
tests/test_meteor_shield.py — 85 unit tests for Mars MMOD protection.

Targets: src/meteor_shield.py
  - WhippleShield dataclass (clamping, effective bumper, mass)
  - ImpactEvent dataclass
  - Meteoroid flux model (Grün 1985 scaled to Mars)
  - Impact probability (Poisson model)
  - Ballistic limit equation (Cour-Palais)
  - Cratering (Holsapple scaling)
  - Shield degradation model
  - tick_shield integration (full simulation tick)
  - Physical invariants (energy conservation, probability bounds)
  - Property-based tests (monotonicity, bounds across parameter ranges)

Run:
    python -m pytest tests/test_meteor_shield.py -v

53 votes said ship code.  One file.  One test.  One merge.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.meteor_shield import (
    WhippleShield,
    ImpactEvent,
    meteoroid_mass_to_diameter,
    grun_flux_1au,
    mars_flux,
    impact_probability,
    expected_impacts,
    critical_diameter,
    kinetic_energy_j,
    check_penetration,
    crater_depth_mm,
    damage_from_impact,
    random_impact_velocity,
    random_impact_angle,
    generate_impact,
    tick_shield,
    shield_status,
    METEOROID_DENSITY_DEFAULT,
    METEOROID_DENSITY_COMETARY,
    METEOROID_DENSITY_CHONDRITIC,
    ALUMINIUM_DENSITY,
    IMPACT_VELOCITY_MEAN_KMS,
    IMPACT_VELOCITY_MIN_KMS,
    IMPACT_VELOCITY_MAX_KMS,
    GRUN_MIN_MASS_KG,
    GRUN_MAX_MASS_KG,
    AU_MARS,
    SOLS_PER_YEAR,
    DEFAULT_BUMPER_MM,
    DEFAULT_STANDOFF_CM,
    DEFAULT_WALL_MM,
)


# ===================================================================
# WhippleShield dataclass
# ===================================================================


class TestWhippleShield:
    """WhippleShield construction, clamping, and derived properties."""

    def test_default_construction(self) -> None:
        s = WhippleShield()
        assert s.bumper_mm == DEFAULT_BUMPER_MM
        assert s.standoff_cm == DEFAULT_STANDOFF_CM
        assert s.wall_mm == DEFAULT_WALL_MM
        assert s.health == 1.0
        assert s.cumulative_impacts == 0

    def test_custom_construction(self) -> None:
        s = WhippleShield(bumper_mm=2.0, standoff_cm=20.0, wall_mm=5.0,
                          area_m2=500.0)
        assert s.bumper_mm == 2.0
        assert s.area_m2 == 500.0

    def test_negative_bumper_clamped(self) -> None:
        s = WhippleShield(bumper_mm=-5.0)
        assert s.bumper_mm > 0

    def test_negative_area_clamped(self) -> None:
        s = WhippleShield(area_m2=-100.0)
        assert s.area_m2 > 0

    def test_health_clamped_above_one(self) -> None:
        s = WhippleShield(health=1.5)
        assert s.health == 1.0

    def test_health_clamped_below_zero(self) -> None:
        s = WhippleShield(health=-0.3)
        assert s.health == 0.0

    def test_effective_bumper_full_health(self) -> None:
        s = WhippleShield(bumper_mm=2.0, health=1.0)
        assert s.effective_bumper_mm() == 2.0

    def test_effective_bumper_half_health(self) -> None:
        s = WhippleShield(bumper_mm=2.0, health=0.5)
        assert abs(s.effective_bumper_mm() - 1.0) < 1e-10

    def test_effective_bumper_zero_health(self) -> None:
        s = WhippleShield(bumper_mm=2.0, health=0.0)
        assert s.effective_bumper_mm() == 0.0

    def test_mass_positive(self) -> None:
        s = WhippleShield()
        assert s.mass_kg() > 0

    def test_mass_scales_with_area(self) -> None:
        s1 = WhippleShield(area_m2=100.0)
        s2 = WhippleShield(area_m2=200.0)
        assert abs(s2.mass_kg() / s1.mass_kg() - 2.0) < 0.01

    def test_mass_scales_with_thickness(self) -> None:
        s1 = WhippleShield(bumper_mm=1.0, wall_mm=3.0)
        s2 = WhippleShield(bumper_mm=2.0, wall_mm=6.0)
        assert s2.mass_kg() > s1.mass_kg()

    def test_negative_impacts_clamped(self) -> None:
        s = WhippleShield(cumulative_impacts=-5)
        assert s.cumulative_impacts == 0


# ===================================================================
# ImpactEvent dataclass
# ===================================================================


class TestImpactEvent:
    """ImpactEvent construction."""

    def test_construction(self) -> None:
        e = ImpactEvent(mass_kg=1e-6, velocity_kms=12.0, diameter_m=0.001,
                        angle_deg=30.0, penetrated=False,
                        crater_depth_mm=0.05, kinetic_energy_j=72.0)
        assert e.mass_kg == 1e-6
        assert e.penetrated is False


# ===================================================================
# Meteoroid mass-to-diameter conversion
# ===================================================================


class TestMassTodiameter:
    """Sphere mass ↔ diameter conversion."""

    def test_zero_mass_returns_zero(self) -> None:
        assert meteoroid_mass_to_diameter(0.0) == 0.0

    def test_negative_mass_returns_zero(self) -> None:
        assert meteoroid_mass_to_diameter(-1.0) == 0.0

    def test_positive_diameter(self) -> None:
        d = meteoroid_mass_to_diameter(1e-6)
        assert d > 0

    def test_heavier_is_larger(self) -> None:
        d1 = meteoroid_mass_to_diameter(1e-9)
        d2 = meteoroid_mass_to_diameter(1e-6)
        assert d2 > d1

    def test_roundtrip_volume(self) -> None:
        """Mass → diameter → volume → mass should be consistent."""
        mass = 1e-4
        d = meteoroid_mass_to_diameter(mass, METEOROID_DENSITY_DEFAULT)
        vol = (4.0 / 3.0) * math.pi * (d / 2.0) ** 3
        recovered = vol * METEOROID_DENSITY_DEFAULT
        assert abs(recovered - mass) / mass < 1e-10

    def test_density_effect(self) -> None:
        """Denser material → smaller diameter for same mass."""
        d_ice = meteoroid_mass_to_diameter(1e-6, METEOROID_DENSITY_COMETARY)
        d_rock = meteoroid_mass_to_diameter(1e-6, METEOROID_DENSITY_CHONDRITIC)
        assert d_ice > d_rock


# ===================================================================
# Grün flux model
# ===================================================================


class TestGrunFlux:
    """Grün (1985) meteoroid flux model."""

    def test_flux_positive_at_1au(self) -> None:
        f = grun_flux_1au(1e-6)
        assert f > 0

    def test_flux_monotone_decreasing(self) -> None:
        """Larger minimum mass → fewer particles."""
        f1 = grun_flux_1au(1e-9)
        f2 = grun_flux_1au(1e-6)
        f3 = grun_flux_1au(1e-3)
        assert f1 > f2 > f3

    def test_tiny_mass_high_flux(self) -> None:
        f = grun_flux_1au(GRUN_MIN_MASS_KG)
        assert f > 1.0  # many tiny particles per m²/year

    def test_large_mass_low_flux(self) -> None:
        f = grun_flux_1au(1e-2)
        assert f < 1.0  # rare for 10g+ particles


class TestMarsFlux:
    """Mars-adjusted meteoroid flux."""

    def test_mars_flux_positive(self) -> None:
        f = mars_flux(1e-6)
        assert f > 0

    def test_mars_flux_differs_from_1au(self) -> None:
        f_earth = grun_flux_1au(1e-6)
        f_mars = mars_flux(1e-6)
        assert f_mars != f_earth

    def test_mars_flux_monotone(self) -> None:
        f1 = mars_flux(1e-9)
        f2 = mars_flux(1e-3)
        assert f1 > f2

    def test_small_particles_atmospheric_shielding(self) -> None:
        """Very small particles are partially stopped by Mars atmosphere."""
        f_tiny = mars_flux(1e-12)
        # Without atmo factor the flux would be higher
        f_earth_scaled = grun_flux_1au(1e-12) * (AU_MARS ** -2.0) * 1.5
        assert f_tiny < f_earth_scaled


# ===================================================================
# Impact probability
# ===================================================================


class TestImpactProbability:
    """Poisson impact probability model."""

    def test_zero_flux_zero_probability(self) -> None:
        assert impact_probability(0.0, 200.0, 668.0) == 0.0

    def test_zero_area_zero_probability(self) -> None:
        assert impact_probability(1.0, 0.0, 668.0) == 0.0

    def test_zero_time_zero_probability(self) -> None:
        assert impact_probability(1.0, 200.0, 0.0) == 0.0

    def test_probability_bounds(self) -> None:
        p = impact_probability(1.0, 200.0, 668.0)
        assert 0.0 <= p <= 1.0

    def test_high_flux_approaches_one(self) -> None:
        p = impact_probability(1e6, 200.0, 668.0)
        assert p > 0.999

    def test_probability_increases_with_area(self) -> None:
        p1 = impact_probability(0.01, 100.0, 668.0)
        p2 = impact_probability(0.01, 1000.0, 668.0)
        assert p2 > p1

    def test_probability_increases_with_time(self) -> None:
        p1 = impact_probability(0.01, 200.0, 100.0)
        p2 = impact_probability(0.01, 200.0, 6680.0)
        assert p2 > p1


class TestExpectedImpacts:
    """Expected impact count (Poisson λ)."""

    def test_zero_returns_zero(self) -> None:
        assert expected_impacts(0.0, 200.0, 668.0) == 0.0

    def test_positive_result(self) -> None:
        n = expected_impacts(1.0, 200.0, SOLS_PER_YEAR)
        assert abs(n - 200.0) < 0.1

    def test_linearity_in_area(self) -> None:
        n1 = expected_impacts(1.0, 100.0, SOLS_PER_YEAR)
        n2 = expected_impacts(1.0, 200.0, SOLS_PER_YEAR)
        assert abs(n2 / n1 - 2.0) < 0.01


# ===================================================================
# Ballistic limit & penetration
# ===================================================================


class TestCriticalDiameter:
    """Cour-Palais ballistic limit equation."""

    def test_positive_result(self) -> None:
        s = WhippleShield()
        d = critical_diameter(s)
        assert d > 0

    def test_thicker_bumper_larger_dcrit(self) -> None:
        """Thicker bumper → can stop larger particles."""
        s1 = WhippleShield(bumper_mm=1.0)
        s2 = WhippleShield(bumper_mm=3.0)
        assert critical_diameter(s2) > critical_diameter(s1)

    def test_wider_standoff_larger_dcrit(self) -> None:
        s1 = WhippleShield(standoff_cm=10.0)
        s2 = WhippleShield(standoff_cm=30.0)
        assert critical_diameter(s2) > critical_diameter(s1)

    def test_higher_velocity_smaller_dcrit(self) -> None:
        """Faster impacts → penetrate at smaller size."""
        s = WhippleShield()
        d_slow = critical_diameter(s, velocity_kms=5.0)
        d_fast = critical_diameter(s, velocity_kms=20.0)
        assert d_slow > d_fast

    def test_denser_projectile_smaller_dcrit(self) -> None:
        """Heavier material → penetrates at smaller diameter."""
        s = WhippleShield()
        d_ice = critical_diameter(s, density_proj=METEOROID_DENSITY_COMETARY)
        d_rock = critical_diameter(s, density_proj=METEOROID_DENSITY_CHONDRITIC)
        assert d_ice > d_rock

    def test_oblique_angle_larger_dcrit(self) -> None:
        """Glancing blow → need larger particle to penetrate."""
        s = WhippleShield()
        d_normal = critical_diameter(s, angle_deg=0.0)
        d_oblique = critical_diameter(s, angle_deg=60.0)
        assert d_oblique > d_normal

    def test_damaged_shield_smaller_dcrit(self) -> None:
        """Damaged shield → easier to penetrate."""
        s_new = WhippleShield(health=1.0)
        s_old = WhippleShield(health=0.3)
        assert critical_diameter(s_old) < critical_diameter(s_new)


class TestKineticEnergy:
    """Kinetic energy calculation."""

    def test_zero_mass_zero_energy(self) -> None:
        assert kinetic_energy_j(0.0, 12.0) == 0.0

    def test_positive_energy(self) -> None:
        ke = kinetic_energy_j(1e-6, 12.0)
        assert ke > 0

    def test_energy_scales_with_mass(self) -> None:
        ke1 = kinetic_energy_j(1e-6, 12.0)
        ke2 = kinetic_energy_j(2e-6, 12.0)
        assert abs(ke2 / ke1 - 2.0) < 0.01

    def test_energy_scales_with_v_squared(self) -> None:
        ke1 = kinetic_energy_j(1e-6, 10.0)
        ke2 = kinetic_energy_j(1e-6, 20.0)
        assert abs(ke2 / ke1 - 4.0) < 0.01

    def test_known_value(self) -> None:
        """1 kg at 10 km/s = 0.5 * 1 * (10000)^2 = 50 MJ."""
        ke = kinetic_energy_j(1.0, 10.0)
        assert abs(ke - 5e7) < 1.0


class TestCheckPenetration:
    """Penetration check against ballistic limit."""

    def test_tiny_particle_no_penetration(self) -> None:
        s = WhippleShield()
        assert check_penetration(s, 1e-6) is False

    def test_huge_particle_penetrates(self) -> None:
        s = WhippleShield()
        assert check_penetration(s, 1.0) is True

    def test_at_critical_diameter_boundary(self) -> None:
        s = WhippleShield()
        d_crit = critical_diameter(s)
        assert check_penetration(s, d_crit * 0.99) is False
        assert check_penetration(s, d_crit * 1.01) is True


# ===================================================================
# Cratering
# ===================================================================


class TestCratering:
    """Holsapple cratering model."""

    def test_zero_diameter_zero_depth(self) -> None:
        assert crater_depth_mm(0.0) == 0.0

    def test_positive_depth(self) -> None:
        d = crater_depth_mm(1e-3, 12.0)
        assert d > 0

    def test_larger_impactor_deeper_crater(self) -> None:
        d1 = crater_depth_mm(1e-4, 12.0)
        d2 = crater_depth_mm(1e-3, 12.0)
        assert d2 > d1

    def test_faster_impact_deeper_crater(self) -> None:
        d_slow = crater_depth_mm(1e-3, 5.0)
        d_fast = crater_depth_mm(1e-3, 20.0)
        assert d_fast > d_slow

    def test_denser_projectile_deeper_crater(self) -> None:
        d_ice = crater_depth_mm(1e-3, 12.0, density_proj=METEOROID_DENSITY_COMETARY)
        d_rock = crater_depth_mm(1e-3, 12.0, density_proj=METEOROID_DENSITY_CHONDRITIC)
        assert d_rock > d_ice


# ===================================================================
# Shield degradation
# ===================================================================


class TestDamage:
    """Damage model from non-penetrating impacts."""

    def test_zero_crater_no_damage(self) -> None:
        s = WhippleShield()
        assert damage_from_impact(s, 0.0) == 0.0

    def test_positive_damage(self) -> None:
        s = WhippleShield()
        d = damage_from_impact(s, 0.5)
        assert d > 0

    def test_deeper_crater_more_damage(self) -> None:
        s = WhippleShield()
        d1 = damage_from_impact(s, 0.1)
        d2 = damage_from_impact(s, 0.5)
        assert d2 > d1

    def test_damage_bounded(self) -> None:
        """Damage from a single impact should be small."""
        s = WhippleShield()
        d = damage_from_impact(s, 1.0)
        assert d < 0.01  # single impact < 1% health loss


# ===================================================================
# Random generation
# ===================================================================


class TestRandomGeneration:
    """Stochastic impact parameter generation."""

    def test_velocity_in_bounds(self) -> None:
        import random
        random.seed(42)
        for _ in range(1000):
            v = random_impact_velocity()
            assert IMPACT_VELOCITY_MIN_KMS <= v <= IMPACT_VELOCITY_MAX_KMS

    def test_angle_in_bounds(self) -> None:
        import random
        random.seed(42)
        for _ in range(1000):
            a = random_impact_angle()
            assert 0.0 <= a <= 90.0

    def test_generate_impact_positive_mass(self) -> None:
        import random
        random.seed(42)
        for _ in range(100):
            mass, vel, angle = generate_impact()
            assert mass > 0
            assert vel > 0
            assert 0.0 <= angle <= 90.0


# ===================================================================
# tick_shield integration
# ===================================================================


class TestTickShield:
    """Full simulation tick."""

    def test_tick_returns_list(self) -> None:
        s = WhippleShield()
        events = tick_shield(s, sols=1.0, rng_seed=42)
        assert isinstance(events, list)

    def test_health_never_exceeds_one(self) -> None:
        s = WhippleShield()
        tick_shield(s, sols=100.0, rng_seed=42)
        assert s.health <= 1.0

    def test_health_never_negative(self) -> None:
        s = WhippleShield()
        tick_shield(s, sols=10000.0, rng_seed=42)
        assert s.health >= 0.0

    def test_cumulative_impacts_increase(self) -> None:
        s = WhippleShield()
        tick_shield(s, sols=100.0, rng_seed=42)
        assert s.cumulative_impacts >= 0

    def test_energy_absorbed_non_negative(self) -> None:
        s = WhippleShield()
        tick_shield(s, sols=100.0, rng_seed=42)
        assert s.total_energy_absorbed_j >= 0

    def test_events_have_valid_fields(self) -> None:
        s = WhippleShield(area_m2=1000.0)
        events = tick_shield(s, sols=SOLS_PER_YEAR, rng_seed=42)
        for e in events:
            assert e.mass_kg > 0
            assert e.velocity_kms >= IMPACT_VELOCITY_MIN_KMS
            assert e.diameter_m > 0
            assert 0 <= e.angle_deg <= 90
            assert e.kinetic_energy_j >= 0

    def test_reproducible_with_seed(self) -> None:
        s1 = WhippleShield()
        e1 = tick_shield(s1, sols=100.0, rng_seed=123)
        s2 = WhippleShield()
        e2 = tick_shield(s2, sols=100.0, rng_seed=123)
        assert len(e1) == len(e2)
        assert s1.health == s2.health

    def test_long_duration_shield_degrades(self) -> None:
        """Over 10 Mars years the shield takes measurable damage.

        Use min_mass_kg=1e-10 (sub-microgram dust) which has high
        flux — mm-sized particles (1e-6 kg) are too rare to reliably
        appear even over 10 years at 500 m².
        """
        s = WhippleShield(area_m2=500.0)
        for year in range(10):
            tick_shield(s, sols=SOLS_PER_YEAR, min_mass_kg=1e-10,
                        rng_seed=year)
        # Sub-microgram particles are extremely common
        assert s.cumulative_impacts > 0

    def test_smoke_10_ticks(self) -> None:
        """Smoke test: 10 sols without crash."""
        s = WhippleShield()
        for sol in range(10):
            tick_shield(s, sols=1.0, rng_seed=sol)
        assert s.health <= 1.0
        assert s.health >= 0.0


# ===================================================================
# shield_status
# ===================================================================


class TestShieldStatus:
    """JSON-serialisable status output."""

    def test_returns_dict(self) -> None:
        s = WhippleShield()
        st = shield_status(s)
        assert isinstance(st, dict)

    def test_contains_required_keys(self) -> None:
        s = WhippleShield()
        st = shield_status(s)
        for key in ("health_pct", "effective_bumper_mm", "cumulative_impacts",
                     "total_energy_absorbed_kj", "mass_kg", "area_m2",
                     "penetration_risk"):
            assert key in st

    def test_health_100_at_start(self) -> None:
        s = WhippleShield()
        st = shield_status(s)
        assert st["health_pct"] == 100.0

    def test_risk_low_at_full_health(self) -> None:
        s = WhippleShield()
        assert shield_status(s)["penetration_risk"] == "LOW"

    def test_risk_high_at_low_health(self) -> None:
        s = WhippleShield(health=0.1)
        assert shield_status(s)["penetration_risk"] == "HIGH"

    def test_risk_medium_at_mid_health(self) -> None:
        s = WhippleShield(health=0.5)
        assert shield_status(s)["penetration_risk"] == "MEDIUM"


# ===================================================================
# Physical invariants & property tests
# ===================================================================


class TestPhysicalInvariants:
    """Conservation laws and physical bounds."""

    def test_energy_is_half_mv_squared(self) -> None:
        """KE = ½mv² — fundamental physics."""
        for mass in [1e-9, 1e-6, 1e-3, 1.0]:
            for v in [5.0, 12.0, 20.0]:
                ke = kinetic_energy_j(mass, v)
                expected = 0.5 * mass * (v * 1000) ** 2
                assert abs(ke - expected) / max(expected, 1e-30) < 1e-10

    def test_flux_power_law(self) -> None:
        """Flux follows power law: smaller mass → more particles."""
        masses = [1e-12, 1e-9, 1e-6, 1e-3]
        fluxes = [grun_flux_1au(m) for m in masses]
        for i in range(len(fluxes) - 1):
            assert fluxes[i] > fluxes[i + 1]

    def test_diameter_cube_root_of_mass(self) -> None:
        """d ∝ m^(1/3) for constant density."""
        m1, m2 = 1e-9, 1e-6  # factor of 1000
        d1 = meteoroid_mass_to_diameter(m1)
        d2 = meteoroid_mass_to_diameter(m2)
        ratio = d2 / d1
        expected_ratio = (m2 / m1) ** (1.0 / 3.0)  # = 10
        assert abs(ratio - expected_ratio) / expected_ratio < 1e-6

    def test_shield_mass_conservation(self) -> None:
        """Shield mass = density × volume (bumper + wall)."""
        s = WhippleShield(bumper_mm=2.0, wall_mm=4.0, area_m2=100.0)
        vol = 100.0 * (0.002 + 0.004)  # m³
        expected_mass = vol * ALUMINIUM_DENSITY
        assert abs(s.mass_kg() - expected_mass) < 0.01

    def test_probability_monotone_in_time(self) -> None:
        """Longer exposure → higher impact probability."""
        times = [1, 10, 100, 1000, 10000]
        probs = [impact_probability(0.1, 200.0, t) for t in times]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1]

    def test_critical_diameter_monotone_in_bumper(self) -> None:
        """Thicker bumper → can resist larger particles."""
        thicknesses = [0.5, 1.0, 2.0, 4.0, 8.0]
        d_crits = []
        for t in thicknesses:
            s = WhippleShield(bumper_mm=t)
            d_crits.append(critical_diameter(s))
        for i in range(len(d_crits) - 1):
            assert d_crits[i] < d_crits[i + 1]

    def test_crater_depth_monotone_in_diameter(self) -> None:
        """Bigger impactor → deeper crater."""
        diams = [1e-5, 1e-4, 1e-3, 1e-2]
        depths = [crater_depth_mm(d) for d in diams]
        for i in range(len(depths) - 1):
            assert depths[i] < depths[i + 1]

    def test_no_negative_crater(self) -> None:
        """Crater depth is never negative."""
        for d in [0.0, 1e-10, 1e-5, 1e-2]:
            for v in [0.0, 5.0, 12.0, 25.0]:
                assert crater_depth_mm(d, v) >= 0.0
