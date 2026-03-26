"""
Dedicated tests for src/mars_env.py — Mars environment physics.

Property-based invariants: all outputs stay within physical bounds,
conservation laws hold, determinism is guaranteed.

Run: python -m pytest tests/test_mars_env.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars_env import (
    ATMOSPHERIC_PRESSURE_KPA,
    BASE_RADIATION_MSV_SOL,
    BASE_SOLAR_FLUX_WM2,
    MEAN_TEMP_C,
    SOLAR_FLARE_EXTRA_MSV,
    SOLS_PER_MARS_YEAR,
    TEMP_AMPLITUDE_C,
    TERRAFORM_RADIATION_DAMPING,
    TERRAFORM_TEMP_BONUS_C,
    TERRAFORM_THRESHOLDS,
    DustStorm,
    MarsEnvironment,
    radiation_msv,
    season_name,
    sol_to_ls,
    solar_flux_wm2,
    surface_temperature_c,
)


# ── Pure function tests ──────────────────────────────────────────


class TestSolToLs:
    """sol_to_ls: solar longitude is in [0, 360) and wraps correctly."""

    def test_sol_zero_is_ls_zero(self) -> None:
        assert sol_to_ls(0) == 0.0

    def test_full_year_wraps(self) -> None:
        ls = sol_to_ls(int(SOLS_PER_MARS_YEAR))
        assert 0.0 <= ls < 360.0

    def test_monotonic_within_year(self) -> None:
        """Ls increases monotonically for the first Mars year."""
        prev = -1.0
        for sol in range(1, int(SOLS_PER_MARS_YEAR)):
            ls = sol_to_ls(sol)
            assert ls > prev, f"sol {sol}: Ls {ls} <= prev {prev}"
            prev = ls

    def test_bounds_for_1000_sols(self) -> None:
        """Ls always in [0, 360) regardless of sol count."""
        for sol in range(1001):
            ls = sol_to_ls(sol)
            assert 0.0 <= ls < 360.0, f"sol {sol}: Ls {ls} out of range"


class TestSeasonName:
    """season_name: four seasons map to correct Ls ranges."""

    def test_spring(self) -> None:
        assert season_name(0.0) == "spring"
        assert season_name(89.9) == "spring"

    def test_summer(self) -> None:
        assert season_name(90.0) == "summer"
        assert season_name(179.9) == "summer"

    def test_autumn(self) -> None:
        assert season_name(180.0) == "autumn"
        assert season_name(269.9) == "autumn"

    def test_winter(self) -> None:
        assert season_name(270.0) == "winter"
        assert season_name(359.9) == "winter"

    def test_all_ls_produce_valid_season(self) -> None:
        valid = {"spring", "summer", "autumn", "winter"}
        for deg in range(360):
            assert season_name(float(deg)) in valid


class TestSurfaceTemperature:
    """surface_temperature_c: bounded by physical Mars range."""

    def test_bounds_all_ls(self) -> None:
        """Temperature stays within Mars physical limits for all Ls."""
        t_min = MEAN_TEMP_C - TEMP_AMPLITUDE_C - 1.0  # small tolerance
        t_max = MEAN_TEMP_C + TEMP_AMPLITUDE_C + 1.0
        for deg in range(360):
            t = surface_temperature_c(float(deg))
            assert t_min <= t <= t_max, f"Ls {deg}: temp {t} out of [{t_min}, {t_max}]"

    def test_warmest_near_perihelion(self) -> None:
        """Peak temperature should be near Ls ~160 (70° phase shift from Ls 0)."""
        temps = [(ls, surface_temperature_c(float(ls))) for ls in range(360)]
        _, peak_temp = max(temps, key=lambda x: x[1])
        peak_ls = [ls for ls, t in temps if t == peak_temp][0]
        # Peak should be somewhere in 100-200 Ls range
        assert 100 <= peak_ls <= 200, f"Peak at Ls {peak_ls}, expected 100-200"


class TestSolarFlux:
    """solar_flux_wm2: physical constraints."""

    def test_clear_sky_flux(self) -> None:
        """Zero dust opacity gives near-base flux (tau=0.3 baseline)."""
        flux = solar_flux_wm2(0.0, 0.0)
        assert flux > 0
        assert flux <= BASE_SOLAR_FLUX_WM2

    def test_global_storm_reduces_flux(self) -> None:
        """Dust opacity 1.0 drastically reduces flux."""
        clear = solar_flux_wm2(90.0, 0.0)
        stormy = solar_flux_wm2(90.0, 1.0)
        assert stormy < clear * 0.01  # tau=8 → exp(-8) ≈ 0.0003

    def test_flux_never_negative(self) -> None:
        """Flux is always non-negative regardless of inputs."""
        for dust in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for ls in range(0, 360, 30):
                assert solar_flux_wm2(float(ls), dust) >= 0


class TestRadiation:
    """radiation_msv: physical bounds and flare effects."""

    def test_baseline_no_flare(self) -> None:
        rad = radiation_msv(0.0, False)
        assert abs(rad - BASE_RADIATION_MSV_SOL) < 0.01

    def test_flare_adds_dose(self) -> None:
        no_flare = radiation_msv(0.0, False)
        with_flare = radiation_msv(0.0, True)
        assert with_flare - no_flare >= SOLAR_FLARE_EXTRA_MSV - 0.01

    def test_dust_shields_gcr(self) -> None:
        """Higher dust opacity reduces GCR dose."""
        clear = radiation_msv(0.0, False)
        dusty = radiation_msv(1.0, False)
        assert dusty < clear

    def test_always_positive(self) -> None:
        for dust in [0.0, 0.5, 1.0]:
            for flare in [True, False]:
                assert radiation_msv(dust, flare) > 0


# ── DustStorm tests ──────────────────────────────────────────────


class TestDustStorm:
    """DustStorm: lifecycle and opacity curves."""

    def test_regional_storm_lifecycle(self) -> None:
        storm = DustStorm("regional", 10, 0.4)
        opacities = []
        while storm.tick():
            opacities.append(storm.opacity())
        assert len(opacities) == 9  # 10 sols - 1 (final tick returns False)
        assert all(0 <= o <= 0.4 for o in opacities)

    def test_global_storm_peak(self) -> None:
        storm = DustStorm("global", 60, 0.9)
        # At start, opacity ramps up (min(1.0, remaining/5))
        assert storm.opacity() == 0.9  # 60/5 > 1, capped at 1.0 * 0.9

    def test_storm_ends(self) -> None:
        """Storm always terminates after its duration."""
        storm = DustStorm("regional", 5, 0.4)
        ticks = 0
        while storm.tick():
            ticks += 1
        assert ticks == 4  # 5 sols - 1

    def test_opacity_rampdown(self) -> None:
        """Opacity decreases as storm nears end."""
        storm = DustStorm("global", 6, 0.9)
        opacities = [storm.opacity()]
        while storm.tick():
            opacities.append(storm.opacity())
        # Last few values should be lower than first
        assert opacities[-1] < opacities[0]


# ── MarsEnvironment integration tests ────────────────────────────


class TestMarsEnvironment:
    """Full environment state machine."""

    def test_deterministic(self) -> None:
        """Same seed produces identical sequences."""
        snaps_a = [MarsEnvironment(seed=99).tick() for _ in range(50)]
        snaps_b = [MarsEnvironment(seed=99).tick() for _ in range(50)]
        for a, b in zip(snaps_a, snaps_b):
            assert a == b

    def test_sol_increments(self) -> None:
        env = MarsEnvironment(seed=1)
        for i in range(1, 20):
            snap = env.tick()
            assert snap["sol"] == i

    def test_10_sol_smoke(self) -> None:
        """10 sols without crash, all fields present."""
        env = MarsEnvironment(seed=7)
        required_keys = {
            "sol", "ls", "season", "temperature_c", "solar_flux_wm2",
            "dust_opacity", "radiation_msv", "storm", "flare",
            "pressure_kpa", "terraforming_progress", "terraform_phase",
        }
        for _ in range(10):
            snap = env.tick()
            assert required_keys <= set(snap.keys())

    def test_temperature_bounds_full_year(self) -> None:
        """Temperature stays within physical Mars limits over a full year."""
        env = MarsEnvironment(seed=42)
        for _ in range(669):
            snap = env.tick()
            assert -150.0 < snap["temperature_c"] < 50.0

    def test_radiation_always_positive(self) -> None:
        env = MarsEnvironment(seed=42)
        for _ in range(669):
            snap = env.tick()
            assert snap["radiation_msv"] > 0

    def test_pressure_positive(self) -> None:
        env = MarsEnvironment(seed=42)
        for _ in range(100):
            snap = env.tick()
            assert snap["pressure_kpa"] > 0

    def test_dust_opacity_bounds(self) -> None:
        """Dust opacity in [0, 1]."""
        env = MarsEnvironment(seed=42)
        for _ in range(669):
            snap = env.tick()
            assert 0.0 <= snap["dust_opacity"] <= 1.0

    def test_ls_in_range(self) -> None:
        env = MarsEnvironment(seed=42)
        for _ in range(669):
            snap = env.tick()
            assert 0.0 <= snap["ls"] < 360.0

    def test_season_valid(self) -> None:
        valid = {"spring", "summer", "autumn", "winter"}
        env = MarsEnvironment(seed=42)
        for _ in range(669):
            snap = env.tick()
            assert snap["season"] in valid


class TestTerraforming:
    """Terraforming feedback loop."""

    def test_terraforming_starts_at_zero(self) -> None:
        env = MarsEnvironment(seed=42)
        assert env.terraforming_progress == 0.0

    def test_apply_terraforming_accumulates(self) -> None:
        env = MarsEnvironment(seed=42)
        env.apply_terraforming(0.1)
        assert abs(env.terraforming_progress - 0.1) < 1e-9
        env.apply_terraforming(0.1)
        assert abs(env.terraforming_progress - 0.2) < 1e-9

    def test_terraforming_capped_at_one(self) -> None:
        env = MarsEnvironment(seed=42)
        env.apply_terraforming(2.0)
        assert env.terraforming_progress == 1.0

    def test_terraform_phases(self) -> None:
        """Phases unlock at correct thresholds."""
        env = MarsEnvironment(seed=42)
        assert env.terraform_phase() is None

        env.apply_terraforming(0.1)
        assert env.terraform_phase() == "early_terraforming"

        env.apply_terraforming(0.2)  # now 0.3
        assert env.terraform_phase() == "atmosphere_thickening"

        env.apply_terraforming(0.2)  # now 0.5
        assert env.terraform_phase() == "liquid_water_possible"

        env.apply_terraforming(0.3)  # now 0.8
        assert env.terraform_phase() == "breathable_approach"

    def test_terraforming_warms_planet(self) -> None:
        """Full terraforming adds TERRAFORM_TEMP_BONUS_C to temperature."""
        env_cold = MarsEnvironment(seed=42)
        env_warm = MarsEnvironment(seed=42)
        env_warm.apply_terraforming(1.0)

        snap_cold = env_cold.tick()
        snap_warm = env_warm.tick()

        diff = snap_warm["temperature_c"] - snap_cold["temperature_c"]
        assert abs(diff - TERRAFORM_TEMP_BONUS_C) < 0.1

    def test_terraforming_reduces_radiation(self) -> None:
        """Full terraforming reduces radiation by TERRAFORM_RADIATION_DAMPING."""
        env_raw = MarsEnvironment(seed=42)
        env_tf = MarsEnvironment(seed=42)
        env_tf.apply_terraforming(1.0)

        snap_raw = env_raw.tick()
        snap_tf = env_tf.tick()

        # Radiation with full terraforming should be notably lower
        assert snap_tf["radiation_msv"] < snap_raw["radiation_msv"]

    def test_terraforming_increases_pressure(self) -> None:
        """Full terraforming adds pressure."""
        env_raw = MarsEnvironment(seed=42)
        env_tf = MarsEnvironment(seed=42)
        env_tf.apply_terraforming(1.0)

        snap_raw = env_raw.tick()
        snap_tf = env_tf.tick()

        assert snap_tf["pressure_kpa"] > snap_raw["pressure_kpa"]


# ── Extended physics invariants ──────────────────────────────────
# These tests validate deeper physical properties: conservation laws,
# statistical correctness, exact formula checks, and edge cases.


class TestTemperaturePhysics:
    """Deeper temperature invariants beyond simple bounds."""

    def test_annual_mean_near_minus_60(self) -> None:
        """Annual mean should be close to -60°C (NASA Mars Fact Sheet)."""
        temps = [surface_temperature_c(float(ls)) for ls in range(360)]
        mean = sum(temps) / len(temps)
        assert abs(mean - MEAN_TEMP_C) < 5.0, f"Mean {mean}°C far from {MEAN_TEMP_C}°C"

    def test_amplitude_matches_constant(self) -> None:
        """Peak-to-trough / 2 should approximate TEMP_AMPLITUDE_C."""
        temps = [surface_temperature_c(float(ls)) for ls in range(360)]
        amplitude = (max(temps) - min(temps)) / 2.0
        assert abs(amplitude - TEMP_AMPLITUDE_C) < 5.0

    def test_no_discontinuities(self) -> None:
        """Adjacent Ls values must not jump more than 2°C."""
        prev = surface_temperature_c(0.0)
        for ls in range(1, 360):
            temp = surface_temperature_c(float(ls))
            assert abs(temp - prev) < 2.0, f"Jump at Ls={ls}: {prev} -> {temp}"
            prev = temp


class TestSolarFluxPhysics:
    """Exact formula validation for solar flux."""

    def test_beer_lambert_exact(self) -> None:
        """Clear sky flux = BASE * exp(-0.3) exactly."""
        flux = solar_flux_wm2(90.0, 0.0)
        expected = BASE_SOLAR_FLUX_WM2 * math.exp(-0.3)
        assert abs(flux - expected) < 1.0

    def test_dust_strictly_monotonic(self) -> None:
        """Flux must strictly decrease as dust increases."""
        prev = solar_flux_wm2(180.0, 0.0)
        for pct in range(1, 11):
            dust = pct / 10.0
            flux = solar_flux_wm2(180.0, dust)
            assert flux < prev, f"Flux did not decrease at dust={dust}"
            prev = flux

    def test_storm_cuts_flux_99_percent(self) -> None:
        clear = solar_flux_wm2(90.0, 0.0)
        storm = solar_flux_wm2(90.0, 1.0)
        assert storm < clear * 0.01


class TestSeasonBoundaries:
    """Exact boundary conditions for season_name."""

    def test_boundary_spring_summer(self) -> None:
        assert season_name(89.9) == "spring"
        assert season_name(90.0) == "summer"

    def test_boundary_summer_autumn(self) -> None:
        assert season_name(179.9) == "summer"
        assert season_name(180.0) == "autumn"

    def test_boundary_autumn_winter(self) -> None:
        assert season_name(269.9) == "autumn"
        assert season_name(270.0) == "winter"


class TestOrbitalMechanics:
    """Deeper sol_to_ls validation."""

    def test_quarter_orbit_is_90(self) -> None:
        ls = sol_to_ls(int(SOLS_PER_MARS_YEAR / 4))
        assert 85.0 < ls < 95.0

    def test_half_orbit_is_180(self) -> None:
        ls = sol_to_ls(int(SOLS_PER_MARS_YEAR / 2))
        assert 175.0 < ls < 185.0


class TestEnvironmentStatistics:
    """Statistical properties over long runs."""

    def test_storm_occurs_in_mars_year(self) -> None:
        """At least one storm should occur in 668 sols."""
        env = MarsEnvironment(seed=42)
        storms = sum(1 for _ in range(668) if env.tick()["storm"] is not None)
        assert storms > 0, "Expected at least one storm in a Mars year"

    def test_flare_occurs_in_mars_year(self) -> None:
        """At 0.3% per sol, expect flares in 668 sols."""
        env = MarsEnvironment(seed=42)
        flares = sum(1 for _ in range(668) if env.tick()["flare"])
        assert flares > 0, "Expected at least one solar flare in a Mars year"

    def test_different_seeds_diverge_on_weather(self) -> None:
        """Different seeds must produce different storm patterns."""
        e1, e2 = MarsEnvironment(seed=1), MarsEnvironment(seed=2)
        s1 = [e1.tick()["storm"] for _ in range(668)]
        s2 = [e2.tick()["storm"] for _ in range(668)]
        assert s1 != s2

    def test_snapshot_has_all_required_keys(self) -> None:
        """Every tick snapshot must contain the full key set."""
        env = MarsEnvironment(seed=42)
        snap = env.tick()
        required = {
            "sol", "ls", "season", "temperature_c", "solar_flux_wm2",
            "dust_opacity", "radiation_msv", "storm", "flare",
            "pressure_kpa", "terraforming_progress", "terraform_phase",
        }
        assert required.issubset(snap.keys())


# ── Property-based fuzz: 50 seeds × 200 sols ────────────────────
# The immune system. Other tests are functional; these enforce
# invariants across 10,000 environment states per property.


class TestPropertyFuzz:
    """50 random seeds × 200 sols — physical invariants must hold for ALL."""

    def test_temperature_bounded(self) -> None:
        """Temperature in (-150, 50)°C across all seeds."""
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                t = env.tick()["temperature_c"]
                assert -150 < t < 50, f"seed={seed}: {t}°C"

    def test_radiation_positive(self) -> None:
        """Radiation dose is always positive."""
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                assert env.tick()["radiation_msv"] > 0

    def test_flux_positive(self) -> None:
        """Solar flux is always positive (Beer-Lambert never hits zero)."""
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                assert env.tick()["solar_flux_wm2"] > 0

    def test_ls_in_range(self) -> None:
        """Ls always in [0, 360)."""
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                ls = env.tick()["ls"]
                assert 0.0 <= ls < 360.0

    def test_pressure_positive(self) -> None:
        """Atmospheric pressure always positive."""
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                assert env.tick()["pressure_kpa"] > 0

    def test_dust_bounded(self) -> None:
        """Dust opacity always in [0, 1]."""
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                d = env.tick()["dust_opacity"]
                assert 0.0 <= d <= 1.0


# ── Storm mechanics edge cases ──────────────────────────────────


class TestStormEdgeCases:
    """DustStorm tick precision and terminal conditions."""

    def test_tick_decrements_by_one(self) -> None:
        """Each tick reduces remaining_sols by exactly 1."""
        storm = DustStorm("global", 10, 0.9)
        storm.tick()
        assert storm.remaining_sols == 9

    def test_dies_at_one_remaining(self) -> None:
        """Storm with 1 sol remaining dies on next tick."""
        storm = DustStorm("regional", 1, 0.4)
        alive = storm.tick()
        assert alive is False
        assert storm.remaining_sols == 0

    def test_opacity_formula_exact(self) -> None:
        """Verify opacity = peak * min(1, remaining/5) for last sols."""
        storm = DustStorm("regional", 3, 0.4)
        expected = 0.4 * (3.0 / 5.0)  # 0.24
        assert abs(storm.opacity() - expected) < 0.001

    def test_full_lifecycle_tick_count(self) -> None:
        """A 20-sol storm yields exactly 19 alive ticks."""
        storm = DustStorm("global", 20, 0.9)
        alive_ticks = 0
        while storm.tick():
            alive_ticks += 1
        assert alive_ticks == 19


# ── Miscellaneous gaps ──────────────────────────────────────────


class TestMiscGaps:
    """Tests for edge cases not covered elsewhere."""

    def test_sol_to_ls_overflow_safe(self) -> None:
        """Ls stays in [0, 360) even at 1 million sols."""
        for sol in [10_000, 100_000, 1_000_000]:
            ls = sol_to_ls(sol)
            assert 0.0 <= ls < 360.0, f"sol={sol} → ls={ls}"

    def test_coldest_in_winter(self) -> None:
        """Minimum temperature occurs near Ls 340° (winter)."""
        temps = {ls: surface_temperature_c(float(ls)) for ls in range(360)}
        trough_ls = min(temps, key=temps.get)
        assert 320 < trough_ls or trough_ls < 20

    def test_radiation_exact_gcr_constant(self) -> None:
        """Zero dust, no flare → radiation equals BASE_RADIATION_MSV_SOL exactly."""
        rad = radiation_msv(0.0, False)
        assert abs(rad - BASE_RADIATION_MSV_SOL) < 0.001

    def test_all_terraform_thresholds_valid(self) -> None:
        """Every threshold is in (0, 1] with a non-empty name."""
        for threshold, name in TERRAFORM_THRESHOLDS.items():
            assert 0.0 < threshold <= 1.0
            assert len(name) > 0


# ── Bugfix regression: DustStorm negative opacity ────────────────
# DustStorm.opacity() returned negative when remaining_sols < 0.
# Fixed by clamping with max(0.0, ...).


class TestDustStormOpacityClamp:
    """Regression tests for DustStorm.opacity() negative value bug."""

    def test_opacity_nonneg_after_death(self) -> None:
        """Opacity stays >= 0 even after storm ticks past its lifetime."""
        storm = DustStorm("global", duration=2, peak_opacity=0.9)
        for _ in range(10):
            assert storm.opacity() >= 0, (
                f"Negative opacity {storm.opacity()} at "
                f"remaining_sols={storm.remaining_sols}"
            )
            storm.tick()

    def test_opacity_zero_at_death(self) -> None:
        """When remaining_sols hits 0, opacity is exactly 0."""
        storm = DustStorm("regional", duration=1, peak_opacity=0.5)
        storm.tick()  # remaining_sols -> 0
        assert storm.opacity() == 0.0

    def test_opacity_zero_past_death(self) -> None:
        """Continued ticking past death keeps opacity at 0, not negative."""
        storm = DustStorm("global", duration=1, peak_opacity=0.9)
        storm.tick()  # dies
        storm.tick()  # remaining_sols = -1
        storm.tick()  # remaining_sols = -2
        assert storm.opacity() == 0.0

    def test_lifecycle_trajectory_nonneg(self) -> None:
        """Full opacity trajectory is non-negative through and past death."""
        storm = DustStorm("global", duration=8, peak_opacity=0.9)
        opacities = []
        for _ in range(15):  # tick well past death
            opacities.append(storm.opacity())
            storm.tick()
        assert all(o >= 0 for o in opacities), (
            f"Negative opacity in trajectory: {opacities}"
        )
        # After death, should be clamped to 0
        assert opacities[-1] == 0.0
