"""
Unit tests for src/mars_env.py — Mars environment physics.

One file. One test. One merge. The community ships code.

Tests every public function and class with physical-bounds invariants,
edge cases, and determinism checks. No mocks, no stubs — pure physics.

Run: python -m pytest tests/test_mars_env.py -v
"""
from __future__ import annotations

import math
import random
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
    TERRAFORM_PRESSURE_BONUS_KPA,
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


# ── sol_to_ls ────────────────────────────────────────────────────────────

class TestSolToLs:
    """Solar longitude conversion — mapping sol count to orbital position."""

    def test_sol_zero_is_ls_zero(self) -> None:
        assert sol_to_ls(0) == 0.0

    def test_full_mars_year_wraps(self) -> None:
        """One full Mars year (668.6 sols) returns to Ls ≈ 0."""
        ls = sol_to_ls(int(SOLS_PER_MARS_YEAR))
        assert ls < 1.0 or ls > 359.0  # near 0/360 boundary

    def test_quarter_year_is_ls_90(self) -> None:
        quarter = int(SOLS_PER_MARS_YEAR / 4)
        ls = sol_to_ls(quarter)
        assert 85 < ls < 95  # approximately 90°

    def test_always_in_0_360(self) -> None:
        """Ls is always in [0, 360) for any non-negative sol."""
        for sol in [0, 1, 100, 668, 1337, 5000, 100_000]:
            ls = sol_to_ls(sol)
            assert 0.0 <= ls < 360.0, f"sol={sol} → ls={ls}"

    def test_monotonic_within_year(self) -> None:
        """Ls increases monotonically within a single Mars year."""
        prev = -1.0
        for sol in range(1, int(SOLS_PER_MARS_YEAR)):
            ls = sol_to_ls(sol)
            assert ls > prev, f"sol={sol}: ls={ls} not > prev={prev}"
            prev = ls


# ── season_name ──────────────────────────────────────────────────────────

class TestSeasonName:
    """Season labeling from Ls."""

    def test_spring(self) -> None:
        assert season_name(0.0) == "spring"
        assert season_name(45.0) == "spring"
        assert season_name(89.9) == "spring"

    def test_summer(self) -> None:
        assert season_name(90.0) == "summer"
        assert season_name(135.0) == "summer"
        assert season_name(179.9) == "summer"

    def test_autumn(self) -> None:
        assert season_name(180.0) == "autumn"
        assert season_name(225.0) == "autumn"
        assert season_name(269.9) == "autumn"

    def test_winter(self) -> None:
        assert season_name(270.0) == "winter"
        assert season_name(315.0) == "winter"
        assert season_name(359.9) == "winter"

    def test_all_ls_produce_valid_season(self) -> None:
        valid = {"spring", "summer", "autumn", "winter"}
        for ls in range(360):
            s = season_name(float(ls))
            assert s in valid, f"ls={ls} → {s}"


# ── surface_temperature_c ────────────────────────────────────────────────

class TestSurfaceTemperature:
    """Temperature model — seasonal sinusoidal variation."""

    def test_physically_bounded(self) -> None:
        """Temperature stays within ±amplitude of mean for all Ls."""
        low = MEAN_TEMP_C - TEMP_AMPLITUDE_C - 1.0
        high = MEAN_TEMP_C + TEMP_AMPLITUDE_C + 1.0
        for ls in range(360):
            t = surface_temperature_c(float(ls))
            assert low < t < high, f"ls={ls} → {t}°C out of [{low}, {high}]"

    def test_warmest_in_summer(self) -> None:
        """Peak temperature occurs near Ls=160° (70° phase offset in sin)."""
        temps = {ls: surface_temperature_c(float(ls)) for ls in range(360)}
        peak_ls = max(temps, key=temps.get)
        # sin(ls - 70) peaks at ls = 160
        assert 140 < peak_ls < 180, f"peak at Ls={peak_ls}, expected ~160"

    def test_coldest_in_winter(self) -> None:
        """Minimum temperature occurs near Ls=340°."""
        temps = {ls: surface_temperature_c(float(ls)) for ls in range(360)}
        trough_ls = min(temps, key=temps.get)
        assert 320 < trough_ls or trough_ls < 20, f"trough at Ls={trough_ls}"


# ── solar_flux_wm2 ──────────────────────────────────────────────────────

class TestSolarFlux:
    """Solar flux model — Beer-Lambert dust attenuation."""

    def test_clear_sky_near_base(self) -> None:
        """Zero dust opacity → flux attenuated only by base tau=0.3."""
        flux = solar_flux_wm2(0.0, 0.0)
        expected = BASE_SOLAR_FLUX_WM2 * math.exp(-0.3)
        assert abs(flux - expected) < 0.1

    def test_global_storm_near_zero(self) -> None:
        """Full dust opacity → flux nearly zero (tau=8)."""
        flux = solar_flux_wm2(0.0, 1.0)
        assert flux < 1.0  # practically zero

    def test_always_positive(self) -> None:
        """Flux is always positive (Beer-Lambert never goes negative)."""
        for ls in range(0, 360, 30):
            for dust in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
                f = solar_flux_wm2(float(ls), dust)
                assert f > 0.0, f"ls={ls}, dust={dust} → flux={f}"

    def test_monotone_decreasing_with_dust(self) -> None:
        """More dust → less flux, for any Ls."""
        ls = 90.0
        prev = solar_flux_wm2(ls, 0.0)
        for dust in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            f = solar_flux_wm2(ls, dust)
            assert f < prev, f"dust={dust}: {f} not < {prev}"
            prev = f


# ── radiation_msv ────────────────────────────────────────────────────────

class TestRadiation:
    """Radiation dose model — GCR + solar flare."""

    def test_baseline_no_flare_no_dust(self) -> None:
        rad = radiation_msv(0.0, False)
        assert abs(rad - BASE_RADIATION_MSV_SOL) < 0.001

    def test_flare_adds_dose(self) -> None:
        no_flare = radiation_msv(0.0, False)
        with_flare = radiation_msv(0.0, True)
        assert abs(with_flare - no_flare - SOLAR_FLARE_EXTRA_MSV) < 0.001

    def test_dust_reduces_gcr(self) -> None:
        """Dust provides GCR shielding — more dust, less GCR."""
        clear = radiation_msv(0.0, False)
        dusty = radiation_msv(1.0, False)
        assert dusty < clear

    def test_always_positive(self) -> None:
        for dust in [0.0, 0.5, 1.0]:
            for flare in [True, False]:
                r = radiation_msv(dust, flare)
                assert r > 0.0


# ── DustStorm ────────────────────────────────────────────────────────────

class TestDustStorm:
    """Dust storm lifecycle — creation, opacity decay, termination."""

    def test_global_storm_peak(self) -> None:
        storm = DustStorm("global", 50, 0.9)
        assert abs(storm.opacity() - 0.9) < 0.01  # at full remaining, peak

    def test_opacity_decays_in_final_sols(self) -> None:
        """Opacity drops linearly in the last 5 sols."""
        storm = DustStorm("regional", 3, 0.4)
        # remaining=3 → opacity = 0.4 * min(1, 3/5) = 0.4 * 0.6 = 0.24
        assert abs(storm.opacity() - 0.24) < 0.001

    def test_tick_decrements_remaining(self) -> None:
        storm = DustStorm("regional", 10, 0.4)
        alive = storm.tick()
        assert alive is True
        assert storm.remaining_sols == 9

    def test_storm_dies_at_zero(self) -> None:
        storm = DustStorm("regional", 1, 0.4)
        alive = storm.tick()
        assert alive is False
        assert storm.remaining_sols == 0

    def test_full_lifecycle(self) -> None:
        """Storm runs to completion without crash."""
        storm = DustStorm("global", 20, 0.9)
        sols_alive = 0
        while storm.tick():
            sols_alive += 1
            assert storm.opacity() >= 0.0
        assert sols_alive == 19  # ticked 19 times before dying on tick 20


# ── MarsEnvironment ──────────────────────────────────────────────────────

class TestMarsEnvironment:
    """Full environment state machine — tick, storms, terraforming."""

    def test_deterministic(self) -> None:
        """Same seed → identical sol-by-sol output."""
        e1 = MarsEnvironment(seed=42)
        e2 = MarsEnvironment(seed=42)
        for _ in range(100):
            s1 = e1.tick()
            s2 = e2.tick()
            assert s1 == s2

    def test_sol_increments(self) -> None:
        env = MarsEnvironment(seed=1)
        for expected_sol in range(1, 11):
            snap = env.tick()
            assert snap["sol"] == expected_sol

    def test_snapshot_keys(self) -> None:
        """Every tick returns all required fields."""
        required = {
            "sol", "ls", "season", "temperature_c", "solar_flux_wm2",
            "dust_opacity", "radiation_msv", "storm", "flare",
            "pressure_kpa", "terraforming_progress", "terraform_phase",
        }
        env = MarsEnvironment(seed=7)
        snap = env.tick()
        assert required.issubset(snap.keys())

    def test_temperature_physical_bounds_1000_sols(self) -> None:
        """Temperature stays within Mars physical limits over 1000 sols."""
        env = MarsEnvironment(seed=99)
        for _ in range(1000):
            snap = env.tick()
            assert -150 < snap["temperature_c"] < 50, snap

    def test_radiation_always_positive_1000_sols(self) -> None:
        env = MarsEnvironment(seed=99)
        for _ in range(1000):
            snap = env.tick()
            assert snap["radiation_msv"] > 0, snap

    def test_flux_always_positive_1000_sols(self) -> None:
        env = MarsEnvironment(seed=99)
        for _ in range(1000):
            snap = env.tick()
            assert snap["solar_flux_wm2"] > 0, snap

    def test_pressure_always_positive(self) -> None:
        env = MarsEnvironment(seed=99)
        for _ in range(500):
            snap = env.tick()
            assert snap["pressure_kpa"] > 0, snap

    def test_dust_opacity_bounded(self) -> None:
        """Dust opacity in [0, 1]."""
        env = MarsEnvironment(seed=99)
        for _ in range(1000):
            snap = env.tick()
            assert 0.0 <= snap["dust_opacity"] <= 1.0, snap

    def test_storms_occur_in_storm_season(self) -> None:
        """Over many sols, storms should appear during Ls 180–330."""
        env = MarsEnvironment(seed=12345)
        storm_sols = []
        for _ in range(2000):
            snap = env.tick()
            if snap["storm"] is not None:
                storm_sols.append(snap["ls"])
        assert len(storm_sols) > 0, "No storms in 2000 sols"
        # At least some storms should be in the storm season
        in_season = [ls for ls in storm_sols if 180 <= ls <= 330]
        assert len(in_season) > 0, "No storms during storm season"

    def test_flares_rare(self) -> None:
        """Solar flares occur but are rare (< 5% of sols)."""
        env = MarsEnvironment(seed=42)
        flare_count = sum(1 for _ in range(2000) if env.tick()["flare"])
        assert 0 < flare_count < 100, f"flares={flare_count}/2000"


class TestTerraforming:
    """Terraforming feedback loop — progress modifies environment."""

    def test_initial_progress_zero(self) -> None:
        env = MarsEnvironment(seed=1)
        assert env.terraforming_progress == 0.0

    def test_apply_terraforming_accumulates(self) -> None:
        env = MarsEnvironment(seed=1)
        env.apply_terraforming(0.1)
        env.apply_terraforming(0.2)
        assert abs(env.terraforming_progress - 0.3) < 0.001

    def test_terraforming_capped_at_1(self) -> None:
        env = MarsEnvironment(seed=1)
        env.apply_terraforming(0.8)
        env.apply_terraforming(0.5)
        assert env.terraforming_progress == 1.0

    def test_terraform_phase_progression(self) -> None:
        """Phases unlock at correct thresholds."""
        env = MarsEnvironment(seed=1)
        assert env.terraform_phase() is None

        env.apply_terraforming(0.1)
        assert env.terraform_phase() == "early_terraforming"

        env.apply_terraforming(0.2)  # now 0.3
        assert env.terraform_phase() == "atmosphere_thickening"

        env.apply_terraforming(0.2)  # now 0.5
        assert env.terraform_phase() == "liquid_water_possible"

        env.apply_terraforming(0.3)  # now 0.8
        assert env.terraform_phase() == "breathable_approach"

    def test_terraforming_warms_temperature(self) -> None:
        """Full terraforming adds TERRAFORM_TEMP_BONUS_C to temperature."""
        env_cold = MarsEnvironment(seed=42)
        env_warm = MarsEnvironment(seed=42)
        env_warm.apply_terraforming(1.0)

        snap_cold = env_cold.tick()
        snap_warm = env_warm.tick()

        diff = snap_warm["temperature_c"] - snap_cold["temperature_c"]
        assert abs(diff - TERRAFORM_TEMP_BONUS_C) < 0.1

    def test_terraforming_increases_pressure(self) -> None:
        """Full terraforming adds TERRAFORM_PRESSURE_BONUS_KPA."""
        env_lo = MarsEnvironment(seed=42)
        env_hi = MarsEnvironment(seed=42)
        env_hi.apply_terraforming(1.0)

        snap_lo = env_lo.tick()
        snap_hi = env_hi.tick()

        diff = snap_hi["pressure_kpa"] - snap_lo["pressure_kpa"]
        assert abs(diff - TERRAFORM_PRESSURE_BONUS_KPA) < 0.01

    def test_terraforming_reduces_radiation(self) -> None:
        """Full terraforming reduces radiation dose."""
        env_raw = MarsEnvironment(seed=42)
        env_tf = MarsEnvironment(seed=42)
        env_tf.apply_terraforming(1.0)

        snap_raw = env_raw.tick()
        snap_tf = env_tf.tick()

        assert snap_tf["radiation_msv"] < snap_raw["radiation_msv"]

    def test_all_thresholds_documented(self) -> None:
        """Every terraform threshold has a non-empty name."""
        for threshold, name in TERRAFORM_THRESHOLDS.items():
            assert 0.0 < threshold <= 1.0
            assert len(name) > 0


# ── Property-based: 50 random seeds ─────────────────────────────────────

class TestPropertyBased:
    """Fuzz 50 random seeds — invariants must hold for all."""

    def test_50_seeds_temperature_bounded(self) -> None:
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                snap = env.tick()
                assert -150 < snap["temperature_c"] < 50

    def test_50_seeds_radiation_positive(self) -> None:
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                snap = env.tick()
                assert snap["radiation_msv"] > 0

    def test_50_seeds_flux_positive(self) -> None:
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                snap = env.tick()
                assert snap["solar_flux_wm2"] > 0

    def test_50_seeds_ls_valid(self) -> None:
        for seed in range(50):
            env = MarsEnvironment(seed=seed)
            for _ in range(200):
                snap = env.tick()
                assert 0.0 <= snap["ls"] < 360.0
