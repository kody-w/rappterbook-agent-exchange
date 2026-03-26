"""
Dedicated physics tests for mars_env.py — the Mars Barn environment engine.

Tests physical invariants, boundary conditions, conservation laws,
and determinism. Every function in mars_env.py gets tested here.

Run: python -m pytest tests/test_mars_env.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars_env import (
    MarsEnvironment,
    DustStorm,
    sol_to_ls,
    season_name,
    surface_temperature_c,
    solar_flux_wm2,
    radiation_msv,
    SOLS_PER_MARS_YEAR,
    MEAN_TEMP_C,
    TEMP_AMPLITUDE_C,
    BASE_SOLAR_FLUX_WM2,
    BASE_RADIATION_MSV_SOL,
    SOLAR_FLARE_EXTRA_MSV,
    ATMOSPHERIC_PRESSURE_KPA,
    TERRAFORM_TEMP_BONUS_C,
    TERRAFORM_PRESSURE_BONUS_KPA,
    TERRAFORM_THRESHOLDS,
)


# ─── sol_to_ls ────────────────────────────────────────────────────────

class TestSolToLs:
    """Solar longitude conversion must respect orbital mechanics."""

    def test_sol_zero_is_ls_zero(self) -> None:
        assert sol_to_ls(0) == 0.0

    def test_full_orbit_wraps(self) -> None:
        ls = sol_to_ls(int(SOLS_PER_MARS_YEAR))
        assert abs(ls) < 1.0 or abs(ls - 360.0) < 1.0

    def test_monotonically_increasing_within_year(self) -> None:
        prev = -1.0
        for sol in range(1, int(SOLS_PER_MARS_YEAR)):
            ls = sol_to_ls(sol)
            assert ls > prev, f"Ls decreased at sol {sol}: {prev} -> {ls}"
            prev = ls

    def test_always_in_range(self) -> None:
        for sol in range(0, 2000):
            ls = sol_to_ls(sol)
            assert 0.0 <= ls < 360.0, f"Ls out of range at sol {sol}: {ls}"

    def test_quarter_orbit(self) -> None:
        ls = sol_to_ls(int(SOLS_PER_MARS_YEAR / 4))
        assert 85.0 < ls < 95.0, f"Quarter orbit should be ~90°, got {ls}"

    def test_half_orbit(self) -> None:
        ls = sol_to_ls(int(SOLS_PER_MARS_YEAR / 2))
        assert 175.0 < ls < 185.0, f"Half orbit should be ~180°, got {ls}"


# ─── season_name ──────────────────────────────────────────────────────

class TestSeasonName:
    """Season names must cover the full Ls range without gaps."""

    def test_all_four_seasons_reachable(self) -> None:
        seasons = {season_name(ls) for ls in range(0, 360, 10)}
        assert seasons == {"spring", "summer", "autumn", "winter"}

    def test_spring_at_ls_0(self) -> None:
        assert season_name(0.0) == "spring"

    def test_summer_at_ls_90(self) -> None:
        assert season_name(90.0) == "summer"

    def test_autumn_at_ls_180(self) -> None:
        assert season_name(180.0) == "autumn"

    def test_winter_at_ls_270(self) -> None:
        assert season_name(270.0) == "winter"

    def test_boundary_spring_summer(self) -> None:
        assert season_name(89.9) == "spring"
        assert season_name(90.0) == "summer"


# ─── surface_temperature_c ────────────────────────────────────────────

class TestSurfaceTemperature:
    """Temperature must stay within Mars physical bounds."""

    def test_always_within_physical_bounds(self) -> None:
        """Mars surface: -140°C to +20°C (NASA). Allow model margin."""
        for ls in range(0, 360):
            temp = surface_temperature_c(float(ls))
            assert -150.0 < temp < 30.0, f"Temp {temp}°C at Ls={ls} out of bounds"

    def test_summer_warmer_than_winter(self) -> None:
        summer = surface_temperature_c(135.0)
        winter = surface_temperature_c(315.0)
        assert summer > winter, f"Summer {summer}°C should exceed winter {winter}°C"

    def test_mean_near_minus_60(self) -> None:
        """Annual mean should be close to -60°C."""
        temps = [surface_temperature_c(float(ls)) for ls in range(360)]
        mean = sum(temps) / len(temps)
        assert abs(mean - MEAN_TEMP_C) < 5.0, f"Mean {mean}°C far from {MEAN_TEMP_C}°C"

    def test_amplitude_matches_constant(self) -> None:
        """Peak-to-trough / 2 should approximate TEMP_AMPLITUDE_C."""
        temps = [surface_temperature_c(float(ls)) for ls in range(360)]
        amplitude = (max(temps) - min(temps)) / 2.0
        assert abs(amplitude - TEMP_AMPLITUDE_C) < 5.0

    def test_continuous_no_jumps(self) -> None:
        """No discontinuities — adjacent Ls values differ by < 2°C."""
        prev = surface_temperature_c(0.0)
        for ls in range(1, 360):
            temp = surface_temperature_c(float(ls))
            assert abs(temp - prev) < 2.0, f"Jump at Ls={ls}: {prev} -> {temp}"
            prev = temp


# ─── solar_flux_wm2 ──────────────────────────────────────────────────

class TestSolarFlux:
    """Solar flux must be positive and decrease with dust."""

    def test_always_positive(self) -> None:
        for ls in range(0, 360, 30):
            for dust in [0.0, 0.25, 0.5, 0.75, 1.0]:
                flux = solar_flux_wm2(float(ls), dust)
                assert flux > 0.0, f"Flux <= 0 at Ls={ls}, dust={dust}"

    def test_clear_sky_near_base(self) -> None:
        """With no dust, flux should be close to BASE_SOLAR_FLUX_WM2."""
        flux = solar_flux_wm2(90.0, 0.0)
        # tau=0.3 gives exp(-0.3) ≈ 0.74
        expected = BASE_SOLAR_FLUX_WM2 * math.exp(-0.3)
        assert abs(flux - expected) < 1.0

    def test_dust_monotonically_reduces_flux(self) -> None:
        prev_flux = solar_flux_wm2(180.0, 0.0)
        for dust_pct in range(1, 11):
            dust = dust_pct / 10.0
            flux = solar_flux_wm2(180.0, dust)
            assert flux < prev_flux, f"Flux increased at dust={dust}"
            prev_flux = flux

    def test_global_storm_drastically_reduces(self) -> None:
        clear = solar_flux_wm2(90.0, 0.0)
        storm = solar_flux_wm2(90.0, 1.0)
        assert storm < clear * 0.01, "Global storm should cut flux by >99%"


# ─── radiation_msv ────────────────────────────────────────────────────

class TestRadiation:
    """Radiation must be positive and respect physics."""

    def test_always_positive(self) -> None:
        for dust in [0.0, 0.5, 1.0]:
            for flare in [True, False]:
                rad = radiation_msv(dust, flare)
                assert rad > 0.0

    def test_flare_adds_radiation(self) -> None:
        normal = radiation_msv(0.0, False)
        flare = radiation_msv(0.0, True)
        assert flare > normal
        assert abs((flare - normal) - SOLAR_FLARE_EXTRA_MSV) < 0.01

    def test_dust_shields_gcr(self) -> None:
        """Dust reduces GCR (shielding effect)."""
        clear = radiation_msv(0.0, False)
        dusty = radiation_msv(1.0, False)
        assert dusty < clear, "Dust should reduce GCR via shielding"

    def test_baseline_matches_curiosity_data(self) -> None:
        """Clear sky, no flare ≈ 0.67 mSv/sol (Curiosity RAD)."""
        rad = radiation_msv(0.0, False)
        assert abs(rad - BASE_RADIATION_MSV_SOL) < 0.01


# ─── DustStorm ────────────────────────────────────────────────────────

class TestDustStorm:
    """Dust storm lifecycle: opacity rises, persists, dies."""

    def test_regional_storm_params(self) -> None:
        s = DustStorm("regional", 10, 0.4)
        assert s.kind == "regional"
        assert s.remaining_sols == 10
        assert s.peak_opacity == 0.4

    def test_opacity_bounded_0_to_peak(self) -> None:
        s = DustStorm("global", 20, 0.9)
        for _ in range(25):
            op = s.opacity()
            assert 0.0 <= op <= 0.9
            if not s.tick():
                break

    def test_storm_dies(self) -> None:
        s = DustStorm("regional", 3, 0.4)
        assert s.tick()  # 2 remaining
        assert s.tick()  # 1 remaining
        assert not s.tick()  # 0 remaining — dead

    def test_opacity_ramps_down(self) -> None:
        """Opacity should decrease in the final 5 sols."""
        s = DustStorm("global", 6, 0.9)
        opacities = []
        for _ in range(6):
            opacities.append(s.opacity())
            s.tick()
        # Last 5 sols should show decreasing opacity
        last_five = opacities[-5:]
        for i in range(1, len(last_five)):
            assert last_five[i] <= last_five[i - 1]


# ─── MarsEnvironment (integration) ───────────────────────────────────

class TestMarsEnvironment:
    """Full environment state machine — determinism + invariants."""

    def test_deterministic(self) -> None:
        """Same seed = identical 365-sol trajectory."""
        e1 = MarsEnvironment(seed=42)
        e2 = MarsEnvironment(seed=42)
        for _ in range(365):
            s1 = e1.tick()
            s2 = e2.tick()
            assert s1 == s2

    def test_different_seeds_diverge(self) -> None:
        """Different seeds produce different storm/flare patterns."""
        e1 = MarsEnvironment(seed=1)
        e2 = MarsEnvironment(seed=2)
        storms1, storms2 = [], []
        for _ in range(668):
            storms1.append(e1.tick()["storm"])
            storms2.append(e2.tick()["storm"])
        assert storms1 != storms2, "Different seeds should produce different weather"

    def test_sol_advances_monotonically(self) -> None:
        env = MarsEnvironment(seed=7)
        for expected in range(1, 101):
            snap = env.tick()
            assert snap["sol"] == expected

    def test_snapshot_has_all_keys(self) -> None:
        env = MarsEnvironment(seed=42)
        snap = env.tick()
        required = {
            "sol", "ls", "season", "temperature_c", "solar_flux_wm2",
            "dust_opacity", "radiation_msv", "storm", "flare",
            "pressure_kpa", "terraforming_progress", "terraform_phase",
        }
        assert required.issubset(snap.keys())

    def test_physical_bounds_365_sols(self) -> None:
        """Every sol for a full Mars year stays within physical bounds."""
        env = MarsEnvironment(seed=42)
        for _ in range(365):
            s = env.tick()
            assert -150.0 < s["temperature_c"] < 30.0
            assert s["solar_flux_wm2"] > 0.0
            assert s["radiation_msv"] > 0.0
            assert 0.0 <= s["dust_opacity"] <= 1.0
            assert s["pressure_kpa"] > 0.0
            assert 0.0 <= s["ls"] < 360.0

    def test_terraforming_applies(self) -> None:
        """apply_terraforming increases progress and modifies output."""
        env = MarsEnvironment(seed=42)
        env.tick()
        base_snap = env.tick()
        env2 = MarsEnvironment(seed=42)
        env2.apply_terraforming(0.5)
        env2.tick()
        tf_snap = env2.tick()
        assert tf_snap["temperature_c"] > base_snap["temperature_c"]
        assert tf_snap["pressure_kpa"] > base_snap["pressure_kpa"]

    def test_terraforming_capped_at_one(self) -> None:
        env = MarsEnvironment(seed=42)
        env.apply_terraforming(0.8)
        env.apply_terraforming(0.8)
        assert env.terraforming_progress == 1.0

    def test_terraform_phase_progression(self) -> None:
        env = MarsEnvironment(seed=42)
        assert env.terraform_phase() is None
        env.apply_terraforming(0.1)
        assert env.terraform_phase() == "early_terraforming"
        env.apply_terraforming(0.2)
        assert env.terraform_phase() == "atmosphere_thickening"
        env.apply_terraforming(0.2)
        assert env.terraform_phase() == "liquid_water_possible"
        env.apply_terraforming(0.3)
        assert env.terraform_phase() == "breathable_approach"

    def test_storm_generation_deterministic(self) -> None:
        """Storm events are seed-deterministic."""
        def count_storms(seed: int) -> int:
            env = MarsEnvironment(seed=seed)
            storms = 0
            for _ in range(668):
                snap = env.tick()
                if snap["storm"] is not None:
                    storms += 1
            return storms
        s1 = count_storms(42)
        s2 = count_storms(42)
        assert s1 == s2
        assert s1 > 0, "Expected at least one storm in a Mars year"

    def test_flare_occurs_in_long_sim(self) -> None:
        """At 0.3% chance per sol, expect flares in 668 sols."""
        env = MarsEnvironment(seed=42)
        flares = sum(1 for _ in range(668) if env.tick()["flare"])
        assert flares > 0, "Expected at least one solar flare in a Mars year"

    def test_pressure_positive(self) -> None:
        """Atmospheric pressure must always be positive."""
        env = MarsEnvironment(seed=42)
        for _ in range(668):
            snap = env.tick()
            assert snap["pressure_kpa"] > 0.0
