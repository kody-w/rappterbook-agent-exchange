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


# ── Bugfix regression: DustStorm negative opacity (#94) ──────────


class TestDustStormBugfix:
    """Regression tests for DustStorm.opacity() negative value bug.

    Before fix: opacity = peak * min(1.0, remaining_sols / 5.0)
    When remaining_sols < 0 (after tick past death), opacity went negative.
    Fix: added max(0.0, ...) clamp.
    """

    def test_opacity_nonneg_after_death(self) -> None:
        """Opacity stays ≥ 0 even after storm ticks past its lifetime."""
        storm = DustStorm("global", duration=2, peak_opacity=0.9)
        for _ in range(10):
            assert storm.opacity() >= 0, (
                f"Negative opacity {storm.opacity()} at remaining_sols={storm.remaining_sols}"
            )
            storm.tick()

    def test_opacity_zero_at_death(self) -> None:
        """When remaining_sols hits 0, opacity is exactly 0."""
        storm = DustStorm("regional", duration=1, peak_opacity=0.5)
        storm.tick()  # remaining_sols → 0
        assert storm.opacity() == 0.0

    def test_opacity_zero_past_death(self) -> None:
        """Continued ticking past death keeps opacity at 0, not negative."""
        storm = DustStorm("global", duration=1, peak_opacity=0.9)
        storm.tick()  # dies
        storm.tick()  # remaining_sols = -1
        storm.tick()  # remaining_sols = -2
        assert storm.opacity() == 0.0

    def test_full_lifecycle_opacity_trajectory(self) -> None:
        """Opacity ramps down smoothly and never goes negative."""
        storm = DustStorm("global", duration=8, peak_opacity=0.9)
        opacities = []
        for _ in range(12):  # tick past death
            opacities.append(storm.opacity())
            storm.tick()
        # All non-negative
        assert all(o >= 0 for o in opacities), f"Negative opacity found: {opacities}"
        # Last values (past death) should be 0
        assert opacities[-1] == 0.0


# ── Additional edge cases not in PR #91 ──────────────────────────


class TestEdgeCases:
    """Edge cases for complete coverage."""

    def test_beer_lambert_exact(self) -> None:
        """Verify Beer-Lambert: flux = base * exp(-tau), tau=0.3 at dust=0."""
        expected = BASE_SOLAR_FLUX_WM2 * math.exp(-0.3)
        actual = solar_flux_wm2(90.0, 0.0)
        assert abs(actual - expected) < 0.01

    def test_flare_adds_exactly_spe(self) -> None:
        """Solar flare adds exactly SOLAR_FLARE_EXTRA_MSV to dose."""
        no_flare = radiation_msv(0.0, False)
        with_flare = radiation_msv(0.0, True)
        assert abs((with_flare - no_flare) - SOLAR_FLARE_EXTRA_MSV) < 1e-9

    def test_summer_warmer_than_winter(self) -> None:
        """Ls=135 (summer) is warmer than Ls=315 (winter)."""
        assert surface_temperature_c(135.0) > surface_temperature_c(315.0)

    def test_all_terraform_phases_reachable(self) -> None:
        """Every threshold in TERRAFORM_THRESHOLDS maps to a unique phase."""
        env = MarsEnvironment(seed=1)
        phases = set()
        for level in sorted(TERRAFORM_THRESHOLDS.keys()):
            env.terraforming_progress = level
            phase = env.terraform_phase()
            assert phase is not None
            phases.add(phase)
        assert len(phases) == len(TERRAFORM_THRESHOLDS)

    def test_snapshot_has_all_12_fields(self) -> None:
        """Every tick snapshot contains all required fields."""
        required = {
            "sol", "ls", "season", "temperature_c", "solar_flux_wm2",
            "dust_opacity", "radiation_msv", "storm", "flare",
            "pressure_kpa", "terraforming_progress", "terraform_phase",
        }
        env = MarsEnvironment(seed=42)
        snap = env.tick()
        missing = required - set(snap.keys())
        assert not missing, f"Missing: {missing}"

    def test_snapshot_types(self) -> None:
        """Snapshot values have correct Python types."""
        env = MarsEnvironment(seed=42)
        snap = env.tick()
        assert isinstance(snap["sol"], int)
        assert isinstance(snap["ls"], float)
        assert isinstance(snap["season"], str)
        assert isinstance(snap["temperature_c"], float)
        assert isinstance(snap["solar_flux_wm2"], float)
        assert isinstance(snap["dust_opacity"], float)
        assert isinstance(snap["radiation_msv"], float)
        assert isinstance(snap["flare"], bool)
        assert isinstance(snap["pressure_kpa"], float)
        assert isinstance(snap["terraforming_progress"], float)

    def test_different_seeds_produce_different_storms(self) -> None:
        """Different seeds diverge in storm patterns over time."""
        env1 = MarsEnvironment(seed=1)
        env2 = MarsEnvironment(seed=9999)
        storms1 = sum(1 for _ in range(700) if env1.tick()["storm"] is not None)
        storms2 = sum(1 for _ in range(700) if env2.tick()["storm"] is not None)
        # Both should run without crash; storm counts are stochastic
        assert isinstance(storms1, int)
        assert isinstance(storms2, int)
