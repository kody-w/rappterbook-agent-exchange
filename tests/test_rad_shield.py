"""Tests for src/rad_shield.py — Mars radiation shielding model."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.rad_shield import (
    ShieldLayer,
    HabitatShield,
    CrewDose,
    areal_density,
    attenuation_factor,
    shielded_dose,
    biological_repair,
    career_fraction,
    tick_radiation,
    GCR_SURFACE_MSV_SOL,
    HVL_REGOLITH_G_CM2,
    HVL_WATER_G_CM2,
    HVL_POLY_G_CM2,
    DENSITY_REGOLITH,
    DENSITY_WATER,
    DENSITY_POLY,
    REPAIR_RATE_PER_SOL,
    MAX_REPAIR_MSV_SOL,
    NASA_CAREER_LIMIT_MSV,
    SEP_MILD_MSV,
    SEP_SEVERE_MSV,
    SEP_DURATION_SOLS,
)


# ===================================================================
# ShieldLayer
# ===================================================================

class TestShieldLayer:
    def test_valid_regolith(self) -> None:
        layer = ShieldLayer("regolith", 50.0)
        assert layer.material == "regolith"
        assert layer.thickness_cm == 50.0

    def test_valid_water(self) -> None:
        layer = ShieldLayer("water", 20.0)
        assert layer.material == "water"

    def test_valid_polyethylene(self) -> None:
        layer = ShieldLayer("polyethylene", 10.0)
        assert layer.material == "polyethylene"

    def test_zero_thickness(self) -> None:
        layer = ShieldLayer("regolith", 0.0)
        assert layer.thickness_cm == 0.0

    def test_negative_thickness_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ShieldLayer("regolith", -1.0)

    def test_invalid_material_raises(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            ShieldLayer("lead", 10.0)


# ===================================================================
# HabitatShield
# ===================================================================

class TestHabitatShield:
    def test_empty_shield(self) -> None:
        shield = HabitatShield()
        assert len(shield.layers) == 0

    def test_add_layer(self) -> None:
        shield = HabitatShield()
        shield.add_layer("regolith", 50.0)
        assert len(shield.layers) == 1
        assert shield.layers[0].material == "regolith"

    def test_multi_layer(self) -> None:
        shield = HabitatShield()
        shield.add_layer("regolith", 50.0)
        shield.add_layer("water", 20.0)
        shield.add_layer("polyethylene", 5.0)
        assert len(shield.layers) == 3


# ===================================================================
# CrewDose
# ===================================================================

class TestCrewDose:
    def test_defaults(self) -> None:
        dose = CrewDose()
        assert dose.cumulative_msv == 0.0
        assert dose.peak_daily_msv == 0.0
        assert dose.sols_exposed == 0

    def test_custom_init(self) -> None:
        dose = CrewDose(cumulative_msv=100.0, peak_daily_msv=5.0, sols_exposed=200)
        assert dose.cumulative_msv == 100.0


# ===================================================================
# areal_density
# ===================================================================

class TestArealDensity:
    def test_regolith(self) -> None:
        ad = areal_density("regolith", 10.0)
        assert ad == pytest.approx(10.0 * DENSITY_REGOLITH)

    def test_water(self) -> None:
        ad = areal_density("water", 18.0)
        assert ad == pytest.approx(18.0 * DENSITY_WATER)

    def test_polyethylene(self) -> None:
        ad = areal_density("polyethylene", 10.0)
        assert ad == pytest.approx(10.0 * DENSITY_POLY)

    def test_zero_thickness(self) -> None:
        assert areal_density("regolith", 0.0) == 0.0


# ===================================================================
# attenuation_factor
# ===================================================================

class TestAttenuationFactor:
    def test_no_shielding(self) -> None:
        shield = HabitatShield()
        assert attenuation_factor(shield) == 1.0

    def test_one_hvl_regolith(self) -> None:
        """One HVL of regolith should halve the dose."""
        thickness = HVL_REGOLITH_G_CM2 / DENSITY_REGOLITH
        shield = HabitatShield()
        shield.add_layer("regolith", thickness)
        assert attenuation_factor(shield) == pytest.approx(0.5, rel=1e-6)

    def test_two_hvl_regolith(self) -> None:
        """Two HVLs should quarter the dose."""
        thickness = 2.0 * HVL_REGOLITH_G_CM2 / DENSITY_REGOLITH
        shield = HabitatShield()
        shield.add_layer("regolith", thickness)
        assert attenuation_factor(shield) == pytest.approx(0.25, rel=1e-6)

    def test_one_hvl_water(self) -> None:
        thickness = HVL_WATER_G_CM2 / DENSITY_WATER
        shield = HabitatShield()
        shield.add_layer("water", thickness)
        assert attenuation_factor(shield) == pytest.approx(0.5, rel=1e-6)

    def test_one_hvl_poly(self) -> None:
        thickness = HVL_POLY_G_CM2 / DENSITY_POLY
        shield = HabitatShield()
        shield.add_layer("polyethylene", thickness)
        assert attenuation_factor(shield) == pytest.approx(0.5, rel=1e-6)

    def test_multi_layer_compounds(self) -> None:
        """Two half-value layers of different materials = 0.25 attenuation."""
        shield = HabitatShield()
        shield.add_layer("regolith", HVL_REGOLITH_G_CM2 / DENSITY_REGOLITH)
        shield.add_layer("water", HVL_WATER_G_CM2 / DENSITY_WATER)
        assert attenuation_factor(shield) == pytest.approx(0.25, rel=1e-6)

    def test_zero_thickness_layer(self) -> None:
        shield = HabitatShield()
        shield.add_layer("regolith", 0.0)
        assert attenuation_factor(shield) == 1.0

    def test_always_positive(self) -> None:
        """Attenuation factor must always be > 0 (never reaches zero)."""
        shield = HabitatShield()
        shield.add_layer("regolith", 1000.0)  # absurdly thick
        att = attenuation_factor(shield)
        assert att > 0.0
        assert att < 0.01  # but very small

    def test_monotonic_with_thickness(self) -> None:
        """More shielding = lower attenuation factor."""
        factors = []
        for t in [0, 10, 20, 50, 100]:
            shield = HabitatShield()
            shield.add_layer("regolith", float(t))
            factors.append(attenuation_factor(shield))
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1]


# ===================================================================
# shielded_dose
# ===================================================================

class TestShieldedDose:
    def test_no_shielding_full_ambient(self) -> None:
        shield = HabitatShield()
        dose = shielded_dose(GCR_SURFACE_MSV_SOL, shield, fraction_indoors=0.0)
        assert dose == pytest.approx(GCR_SURFACE_MSV_SOL)

    def test_no_shielding_all_indoors(self) -> None:
        """With no shielding layers, indoors = full ambient."""
        shield = HabitatShield()
        dose = shielded_dose(GCR_SURFACE_MSV_SOL, shield, fraction_indoors=1.0)
        assert dose == pytest.approx(GCR_SURFACE_MSV_SOL)

    def test_shielded_all_indoors(self) -> None:
        """With shielding and all time indoors, dose = ambient × attenuation."""
        shield = HabitatShield()
        shield.add_layer("regolith", HVL_REGOLITH_G_CM2 / DENSITY_REGOLITH)
        dose = shielded_dose(GCR_SURFACE_MSV_SOL, shield, fraction_indoors=1.0)
        assert dose == pytest.approx(GCR_SURFACE_MSV_SOL * 0.5, rel=1e-6)

    def test_mixed_indoor_outdoor(self) -> None:
        """85% indoors with 1 HVL shielding."""
        shield = HabitatShield()
        shield.add_layer("regolith", HVL_REGOLITH_G_CM2 / DENSITY_REGOLITH)
        dose = shielded_dose(GCR_SURFACE_MSV_SOL, shield, fraction_indoors=0.85)
        expected = GCR_SURFACE_MSV_SOL * (0.5 * 0.85 + 0.15)
        assert dose == pytest.approx(expected, rel=1e-6)

    def test_fraction_bounds(self) -> None:
        shield = HabitatShield()
        with pytest.raises(ValueError, match="fraction_indoors"):
            shielded_dose(1.0, shield, fraction_indoors=-0.1)
        with pytest.raises(ValueError, match="fraction_indoors"):
            shielded_dose(1.0, shield, fraction_indoors=1.1)

    def test_dose_always_non_negative(self) -> None:
        """Shielded dose can never be negative."""
        shield = HabitatShield()
        shield.add_layer("water", 100.0)
        for frac in [0.0, 0.5, 1.0]:
            assert shielded_dose(GCR_SURFACE_MSV_SOL, shield, frac) >= 0.0

    def test_shielding_always_reduces_dose(self) -> None:
        """Adding shielding can never increase dose over unshielded."""
        unshielded = HabitatShield()
        shielded = HabitatShield()
        shielded.add_layer("regolith", 30.0)
        for frac in [0.0, 0.5, 0.85, 1.0]:
            d_un = shielded_dose(GCR_SURFACE_MSV_SOL, unshielded, frac)
            d_sh = shielded_dose(GCR_SURFACE_MSV_SOL, shielded, frac)
            assert d_sh <= d_un + 1e-12


# ===================================================================
# biological_repair
# ===================================================================

class TestBiologicalRepair:
    def test_zero_dose_no_repair(self) -> None:
        dose = CrewDose(cumulative_msv=0.0)
        repaired = biological_repair(dose)
        assert repaired == 0.0
        assert dose.cumulative_msv == 0.0

    def test_small_dose_proportional(self) -> None:
        dose = CrewDose(cumulative_msv=10.0)
        repaired = biological_repair(dose)
        expected = 10.0 * REPAIR_RATE_PER_SOL
        assert repaired == pytest.approx(expected)
        assert dose.cumulative_msv == pytest.approx(10.0 - expected)

    def test_large_dose_capped(self) -> None:
        """Repair is capped at MAX_REPAIR_MSV_SOL."""
        dose = CrewDose(cumulative_msv=500.0)
        repaired = biological_repair(dose)
        assert repaired == pytest.approx(MAX_REPAIR_MSV_SOL)
        assert dose.cumulative_msv == pytest.approx(500.0 - MAX_REPAIR_MSV_SOL)

    def test_never_goes_negative(self) -> None:
        dose = CrewDose(cumulative_msv=0.001)
        biological_repair(dose)
        assert dose.cumulative_msv >= 0.0

    def test_repair_monotonic(self) -> None:
        """Higher dose → more repair (up to cap)."""
        repairs = []
        for d in [1.0, 10.0, 50.0, 100.0]:
            dose = CrewDose(cumulative_msv=d)
            repairs.append(biological_repair(dose))
        for i in range(len(repairs) - 1):
            assert repairs[i] <= repairs[i + 1]


# ===================================================================
# career_fraction
# ===================================================================

class TestCareerFraction:
    def test_zero(self) -> None:
        assert career_fraction(CrewDose()) == 0.0

    def test_at_limit(self) -> None:
        dose = CrewDose(cumulative_msv=NASA_CAREER_LIMIT_MSV)
        assert career_fraction(dose) == pytest.approx(1.0)

    def test_over_limit(self) -> None:
        dose = CrewDose(cumulative_msv=900.0)
        assert career_fraction(dose) > 1.0

    def test_proportional(self) -> None:
        dose = CrewDose(cumulative_msv=300.0)
        assert career_fraction(dose) == pytest.approx(0.5)


# ===================================================================
# tick_radiation — full integration
# ===================================================================

class TestTickRadiation:
    def _make_shield(self) -> HabitatShield:
        """Standard colony shield: 50 cm regolith + 10 cm water."""
        shield = HabitatShield()
        shield.add_layer("regolith", 50.0)
        shield.add_layer("water", 10.0)
        return shield

    def test_one_sol_gcr(self) -> None:
        dose = CrewDose()
        shield = self._make_shield()
        net = tick_radiation(dose, GCR_SURFACE_MSV_SOL, shield)
        assert net > 0.0
        assert dose.cumulative_msv > 0.0
        assert dose.sols_exposed == 1

    def test_dose_accumulates(self) -> None:
        dose = CrewDose()
        shield = self._make_shield()
        for _ in range(100):
            tick_radiation(dose, GCR_SURFACE_MSV_SOL, shield)
        assert dose.sols_exposed == 100
        assert dose.cumulative_msv > 0.0

    def test_peak_tracked(self) -> None:
        dose = CrewDose()
        shield = self._make_shield()
        net1 = tick_radiation(dose, GCR_SURFACE_MSV_SOL, shield)
        first_peak = dose.peak_daily_msv
        tick_radiation(dose, GCR_SURFACE_MSV_SOL * 5, shield)  # flare day
        assert dose.peak_daily_msv > first_peak  # peak should update

    def test_sep_event_increases_dose(self) -> None:
        dose_normal = CrewDose()
        dose_sep = CrewDose()
        shield = self._make_shield()
        tick_radiation(dose_normal, GCR_SURFACE_MSV_SOL, shield)
        tick_radiation(dose_sep, GCR_SURFACE_MSV_SOL, shield, sep_event=True, sep_severity=0.5)
        assert dose_sep.cumulative_msv > dose_normal.cumulative_msv

    def test_sep_severity_scaling(self) -> None:
        """Higher SEP severity → higher dose."""
        shield = self._make_shield()
        doses = []
        for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
            d = CrewDose()
            tick_radiation(d, GCR_SURFACE_MSV_SOL, shield, sep_event=True, sep_severity=sev)
            doses.append(d.cumulative_msv)
        for i in range(len(doses) - 1):
            assert doses[i] <= doses[i + 1]

    def test_sep_severity_clamped(self) -> None:
        """Severity outside [0,1] is clamped."""
        shield = self._make_shield()
        d1 = CrewDose()
        d2 = CrewDose()
        tick_radiation(d1, GCR_SURFACE_MSV_SOL, shield, sep_event=True, sep_severity=1.0)
        tick_radiation(d2, GCR_SURFACE_MSV_SOL, shield, sep_event=True, sep_severity=5.0)
        assert d1.cumulative_msv == pytest.approx(d2.cumulative_msv)

    def test_net_dose_accounts_for_repair(self) -> None:
        dose = CrewDose(cumulative_msv=100.0)
        shield = self._make_shield()
        net = tick_radiation(dose, GCR_SURFACE_MSV_SOL, shield)
        # Net = daily_dose - repair. With 100 mSv cumulative, repair > 0
        daily = shielded_dose(GCR_SURFACE_MSV_SOL, shield, 0.85)
        assert net < daily  # repair should reduce the net

    def test_no_shielding_full_dose(self) -> None:
        dose = CrewDose()
        shield = HabitatShield()
        tick_radiation(dose, GCR_SURFACE_MSV_SOL, shield)
        # With no shielding and default 85% indoors, full ambient passes
        assert dose.cumulative_msv > 0.0


# ===================================================================
# Physical invariants
# ===================================================================

class TestPhysicalInvariants:
    def test_dose_never_negative(self) -> None:
        """Cumulative dose must never go negative, even with repair."""
        dose = CrewDose(cumulative_msv=0.001)
        shield = HabitatShield()
        for _ in range(1000):
            tick_radiation(dose, 0.0, shield)  # zero ambient, only repair
        assert dose.cumulative_msv >= 0.0

    def test_shielding_bounded_zero_one(self) -> None:
        """Attenuation factor must be in (0, 1]."""
        for thickness in [0, 1, 10, 100, 1000]:
            shield = HabitatShield()
            shield.add_layer("regolith", float(thickness))
            att = attenuation_factor(shield)
            assert 0 < att <= 1.0

    def test_career_limit_reachable(self) -> None:
        """With no shielding, GCR alone hits career limit in finite sols."""
        dose = CrewDose()
        shield = HabitatShield()
        for _ in range(5000):
            tick_radiation(dose, GCR_SURFACE_MSV_SOL, shield)
        assert career_fraction(dose) > 1.0

    def test_good_shielding_extends_career(self) -> None:
        """With proper shielding, career limit takes longer to reach."""
        dose_unshielded = CrewDose()
        dose_shielded = CrewDose()
        bare = HabitatShield()
        thick = HabitatShield()
        thick.add_layer("regolith", 100.0)
        thick.add_layer("water", 20.0)
        for _ in range(365):
            tick_radiation(dose_unshielded, GCR_SURFACE_MSV_SOL, bare)
            tick_radiation(dose_shielded, GCR_SURFACE_MSV_SOL, thick)
        assert dose_shielded.cumulative_msv < dose_unshielded.cumulative_msv

    def test_conservation_dose_equals_intake_minus_repair(self) -> None:
        """Total dose ≈ sum(daily) - sum(repaired)."""
        dose = CrewDose()
        shield = HabitatShield()
        shield.add_layer("regolith", 30.0)
        total_net = 0.0
        for _ in range(100):
            net = tick_radiation(dose, GCR_SURFACE_MSV_SOL, shield)
            total_net += net
        assert dose.cumulative_msv == pytest.approx(total_net, rel=1e-6)

    def test_sep_through_shield_less_than_unshielded(self) -> None:
        """SEP dose through shielding < SEP dose unshielded."""
        bare = HabitatShield()
        thick = HabitatShield()
        thick.add_layer("regolith", 80.0)
        d1 = CrewDose()
        d2 = CrewDose()
        tick_radiation(d1, GCR_SURFACE_MSV_SOL, bare, sep_event=True, sep_severity=1.0)
        tick_radiation(d2, GCR_SURFACE_MSV_SOL, thick, sep_event=True, sep_severity=1.0)
        assert d2.cumulative_msv < d1.cumulative_msv


# ===================================================================
# Smoke test — 365 sols without crash
# ===================================================================

class TestSmoke:
    def test_one_year_no_crash(self) -> None:
        """Run 365 sols of radiation tracking without any crash."""
        shield = HabitatShield()
        shield.add_layer("regolith", 50.0)
        shield.add_layer("water", 15.0)
        dose = CrewDose()
        for sol in range(365):
            sep = (sol % 90 == 0)  # SEP every 90 sols
            sev = 0.3 if sep else 0.0
            tick_radiation(
                dose, GCR_SURFACE_MSV_SOL, shield,
                fraction_indoors=0.85,
                sep_event=sep,
                sep_severity=sev,
            )
        assert dose.sols_exposed == 365
        assert dose.cumulative_msv > 0.0
        assert dose.cumulative_msv < NASA_CAREER_LIMIT_MSV  # shielded < limit in 1 yr

    def test_multi_crew(self) -> None:
        """Track 6 crew members with different EVA schedules."""
        shield = HabitatShield()
        shield.add_layer("regolith", 40.0)
        crew = [CrewDose() for _ in range(6)]
        indoor_fracs = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
        for sol in range(365):
            for i, member in enumerate(crew):
                tick_radiation(member, GCR_SURFACE_MSV_SOL, shield,
                               fraction_indoors=indoor_fracs[i])
        # More EVA time = more dose
        for i in range(len(crew) - 1):
            assert crew[i].cumulative_msv <= crew[i + 1].cumulative_msv + 1e-9
