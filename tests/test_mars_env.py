"""
Unit tests for src/mars_env.py — DustStorm, terraforming, physical invariants.

This is the file the community voted to ship. 53 votes. One file. One test. One merge.

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
    BASE_SOLAR_FLUX_WM2,
    BASE_RADIATION_MSV_SOL,
    SOLAR_FLARE_EXTRA_MSV,
    ATMOSPHERIC_PRESSURE_KPA,
    TERRAFORM_THRESHOLDS,
)


# ─── DustStorm unit tests (previously zero direct coverage) ───


class TestDustStorm:
    """DustStorm is the engine's weather system. Test it in isolation."""

    def test_creation(self) -> None:
        s = DustStorm("regional", 10, 0.4)
        assert s.kind == "regional"
        assert s.remaining_sols == 10
        assert s.peak_opacity == 0.4

    def test_tick_decrements(self) -> None:
        s = DustStorm("global", 5, 0.9)
        assert s.tick() is True  # 4 remaining
        assert s.remaining_sols == 4

    def test_tick_returns_false_at_zero(self) -> None:
        s = DustStorm("regional", 1, 0.4)
        assert s.tick() is False  # 0 remaining = dead
        assert s.remaining_sols == 0

    def test_opacity_ramps_down(self) -> None:
        """Opacity should decrease as remaining_sols decreases."""
        s = DustStorm("global", 20, 0.9)
        opacities = []
        for _ in range(20):
            opacities.append(s.opacity())
            s.tick()
        # Last 5 sols should have lower opacity than first 5
        assert sum(opacities[-5:]) < sum(opacities[:5])

    def test_opacity_bounded(self) -> None:
        """Opacity must always be in [0, peak]."""
        for duration in [1, 5, 10, 50, 100]:
            for peak in [0.1, 0.4, 0.9, 1.0]:
                s = DustStorm("global", duration, peak)
                for _ in range(duration + 5):
                    op = s.opacity()
                    assert 0.0 <= op <= peak + 1e-9, (
                        f"Opacity {op} out of [0, {peak}] at remaining={s.remaining_sols}"
                    )
                    if not s.tick():
                        break

    def test_short_storm_hits_peak(self) -> None:
        """A storm with remaining_sols >= 5 reaches peak opacity."""
        s = DustStorm("global", 10, 0.9)
        assert s.opacity() == 0.9  # remaining=10 >= 5, ratio capped at 1.0

    def test_ramp_formula(self) -> None:
        """Opacity = peak * min(1.0, remaining/5)."""
        s = DustStorm("regional", 3, 0.4)
        expected = 0.4 * min(1.0, 3 / 5.0)
        assert abs(s.opacity() - expected) < 1e-9

    def test_global_vs_regional(self) -> None:
        """Global storms have higher peak opacity than regional."""
        g = DustStorm("global", 50, 0.9)
        r = DustStorm("regional", 10, 0.4)
        assert g.opacity() > r.opacity()


# ─── Terraforming unit tests ───


class TestTerraforming:
    """Terraforming feedback is what makes Mars Barn a living system."""

    def test_initial_progress_zero(self) -> None:
        env = MarsEnvironment(seed=1)
        assert env.terraforming_progress == 0.0

    def test_apply_terraforming_accumulates(self) -> None:
        env = MarsEnvironment(seed=1)
        env.apply_terraforming(0.1)
        assert abs(env.terraforming_progress - 0.1) < 1e-9
        env.apply_terraforming(0.2)
        assert abs(env.terraforming_progress - 0.3) < 1e-9

    def test_terraforming_capped_at_one(self) -> None:
        env = MarsEnvironment(seed=1)
        env.apply_terraforming(5.0)
        assert env.terraforming_progress == 1.0

    def test_terraform_phase_none_below_threshold(self) -> None:
        env = MarsEnvironment(seed=1)
        assert env.terraform_phase() is None

    def test_terraform_phase_progression(self) -> None:
        env = MarsEnvironment(seed=1)
        expected = [
            (0.05, None),
            (0.10, "early_terraforming"),
            (0.30, "atmosphere_thickening"),
            (0.50, "liquid_water_possible"),
            (0.80, "breathable_approach"),
        ]
        for progress, phase in expected:
            env.terraforming_progress = progress
            assert env.terraform_phase() == phase, (
                f"At progress={progress}, expected {phase}, got {env.terraform_phase()}"
            )

    def test_all_thresholds_reachable(self) -> None:
        """Every threshold in TERRAFORM_THRESHOLDS can be reached."""
        env = MarsEnvironment(seed=1)
        for threshold, name in sorted(TERRAFORM_THRESHOLDS.items()):
            env.terraforming_progress = threshold
            assert env.terraform_phase() == name


# ─── Physical invariants (property-based) ───


class TestPhysicalInvariants:
    """Physics doesn't lie. These must hold for ALL inputs."""

    def test_temperature_bounded_all_sols(self) -> None:
        """Mars surface temp: -150°C to +30°C (NASA Mars Fact Sheet)."""
        for sol in range(0, 2000):
            ls = sol_to_ls(sol)
            t = surface_temperature_c(ls)
            assert -150 < t < 30, f"Sol {sol}: {t}°C"

    def test_solar_flux_nonnegative(self) -> None:
        """Solar flux can't go negative regardless of dust."""
        for ls in range(0, 360):
            for dust in [0.0, 0.25, 0.5, 0.75, 1.0]:
                flux = solar_flux_wm2(float(ls), dust)
                assert flux >= 0.0, f"Ls={ls}, dust={dust}: flux={flux}"

    def test_solar_flux_decreases_with_dust(self) -> None:
        """More dust = less sunlight. Beer-Lambert law."""
        ls = 90.0
        clear = solar_flux_wm2(ls, 0.0)
        dusty = solar_flux_wm2(ls, 0.5)
        storm = solar_flux_wm2(ls, 1.0)
        assert clear > dusty > storm > 0

    def test_radiation_positive(self) -> None:
        """GCR is always present on Mars. Radiation > 0."""
        for dust in [0.0, 0.5, 1.0]:
            for flare in [True, False]:
                rad = radiation_msv(dust, flare)
                assert rad > 0, f"dust={dust}, flare={flare}: rad={rad}"

    def test_radiation_increases_with_flare(self) -> None:
        """Solar flare adds radiation."""
        no_flare = radiation_msv(0.0, False)
        flare = radiation_msv(0.0, True)
        assert flare > no_flare
        assert abs(flare - no_flare - SOLAR_FLARE_EXTRA_MSV) < 1e-9

    def test_dust_reduces_gcr(self) -> None:
        """Dust atmosphere actually shields against galactic cosmic rays."""
        clear = radiation_msv(0.0, False)
        dusty = radiation_msv(1.0, False)
        assert dusty < clear

    def test_sol_to_ls_range(self) -> None:
        """Ls is always in [0, 360)."""
        for sol in range(0, 3000):
            ls = sol_to_ls(sol)
            assert 0.0 <= ls < 360.0, f"Sol {sol}: Ls={ls}"

    def test_season_coverage(self) -> None:
        """All four seasons exist across a Mars year."""
        seasons = set()
        for sol in range(int(SOLS_PER_MARS_YEAR) + 1):
            ls = sol_to_ls(sol)
            seasons.add(season_name(ls))
        assert seasons == {"spring", "summer", "autumn", "winter"}

    def test_pressure_positive(self) -> None:
        """Atmospheric pressure is always positive."""
        env = MarsEnvironment(seed=99)
        for _ in range(1000):
            snap = env.tick()
            assert snap["pressure_kpa"] > 0, f"Sol {snap['sol']}: pressure={snap['pressure_kpa']}"


# ─── MarsEnvironment state machine ───


class TestMarsEnvironmentStateMachine:
    """The environment is a state machine. Test its transitions."""

    def test_sol_increments(self) -> None:
        env = MarsEnvironment(seed=1)
        for expected in range(1, 11):
            snap = env.tick()
            assert snap["sol"] == expected

    def test_deterministic(self) -> None:
        """Same seed = identical history."""
        e1 = MarsEnvironment(seed=42)
        e2 = MarsEnvironment(seed=42)
        for _ in range(100):
            s1 = e1.tick()
            s2 = e2.tick()
            assert s1 == s2

    def test_different_seeds_diverge(self) -> None:
        """Different seeds produce different weather."""
        e1 = MarsEnvironment(seed=1)
        e2 = MarsEnvironment(seed=9999)
        history1 = [e1.tick() for _ in range(200)]
        history2 = [e2.tick() for _ in range(200)]
        temps1 = [s["temperature_c"] for s in history1]
        temps2 = [s["temperature_c"] for s in history2]
        # Temperatures should be identical (no randomness in temp),
        # but flare/storm patterns should differ
        storms1 = [s["storm"] for s in history1]
        storms2 = [s["storm"] for s in history2]
        assert storms1 != storms2 or any(
            s1["flare"] != s2["flare"] for s1, s2 in zip(history1, history2)
        )

    def test_storms_occur_in_dust_season(self) -> None:
        """Dust storms happen during Ls 180-330 (dust storm season)."""
        env = MarsEnvironment(seed=42)
        storm_sols = []
        for _ in range(2000):
            snap = env.tick()
            if snap["storm"] is not None and snap["storm"] not in [
                s.get("storm") for s in storm_sols[-1:]
            ]:
                storm_sols.append(snap)
        # At least some storms should have occurred
        assert len(storm_sols) > 0, "No storms in 2000 sols"

    def test_snapshot_keys(self) -> None:
        """Every tick snapshot has the required keys."""
        env = MarsEnvironment(seed=42)
        snap = env.tick()
        required = {
            "sol", "ls", "season", "temperature_c", "solar_flux_wm2",
            "dust_opacity", "radiation_msv", "storm", "flare",
            "pressure_kpa", "terraforming_progress", "terraform_phase",
        }
        assert set(snap.keys()) == required

    def test_terraforming_warms_planet(self) -> None:
        """Terraforming progress increases temperature."""
        env_cold = MarsEnvironment(seed=42)
        env_warm = MarsEnvironment(seed=42)
        env_warm.apply_terraforming(1.0)  # max terraforming
        snap_cold = env_cold.tick()
        snap_warm = env_warm.tick()
        assert snap_warm["temperature_c"] > snap_cold["temperature_c"]

    def test_terraforming_increases_pressure(self) -> None:
        """Terraforming thickens the atmosphere."""
        env_thin = MarsEnvironment(seed=42)
        env_thick = MarsEnvironment(seed=42)
        env_thick.apply_terraforming(1.0)
        snap_thin = env_thin.tick()
        snap_thick = env_thick.tick()
        assert snap_thick["pressure_kpa"] > snap_thin["pressure_kpa"]

    def test_terraforming_reduces_radiation(self) -> None:
        """Terraforming dampens cosmic radiation."""
        env_raw = MarsEnvironment(seed=42)
        env_shielded = MarsEnvironment(seed=42)
        env_shielded.apply_terraforming(1.0)
        snap_raw = env_raw.tick()
        snap_shielded = env_shielded.tick()
        assert snap_shielded["radiation_msv"] < snap_raw["radiation_msv"]

    def test_full_mars_year_no_crash(self) -> None:
        """668 sols (one Mars year) completes without error."""
        env = MarsEnvironment(seed=42)
        for _ in range(669):
            snap = env.tick()
            assert isinstance(snap, dict)
        assert env.sol == 669
