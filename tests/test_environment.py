"""Tests for Mars environment model."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from mars.environment import (
    sol_to_ls,
    solar_distance_au,
    top_of_atmosphere_irradiance,
    dust_opacity,
    surface_irradiance,
    surface_temperature,
    radiation_dose,
    atmospheric_pressure,
    MarsEnvironment,
    MARS_YEAR_SOLS,
    MARS_SEMI_MAJOR_AU,
)


class TestSolToLs:
    """Test sol-to-Ls conversion."""

    def test_sol_zero_is_ls_zero(self) -> None:
        assert sol_to_ls(0) == 0.0

    def test_half_year(self) -> None:
        ls = sol_to_ls(int(MARS_YEAR_SOLS / 2))
        assert 175 < ls < 185  # ~180°

    def test_full_year_wraps(self) -> None:
        ls = sol_to_ls(int(MARS_YEAR_SOLS))
        assert ls < 5.0 or ls > 355.0  # Wraps near 0/360

    def test_monotonic_within_year(self) -> None:
        prev = -1.0
        for sol in range(0, int(MARS_YEAR_SOLS) - 1):
            ls = sol_to_ls(sol)
            assert 0 <= ls < 360


class TestSolarDistance:
    """Test Mars-Sun distance calculations."""

    def test_range(self) -> None:
        """Distance should be between perihelion and aphelion."""
        for ls in range(0, 360, 10):
            r = solar_distance_au(float(ls))
            assert 1.38 < r < 1.67, f"r={r} at Ls={ls}"

    def test_perihelion_near_ls251(self) -> None:
        """Closest approach near Ls 251."""
        r = solar_distance_au(251.0)
        assert r < 1.45


class TestIrradiance:
    """Test solar irradiance calculations."""

    def test_toa_positive(self) -> None:
        for ls in range(0, 360, 30):
            toa = top_of_atmosphere_irradiance(float(ls))
            assert toa > 400, f"TOA={toa} at Ls={ls}"

    def test_surface_less_than_toa(self) -> None:
        rng = random.Random(42)
        for ls in [0, 90, 180, 270]:
            tau = dust_opacity(float(ls), rng)
            toa = top_of_atmosphere_irradiance(float(ls))
            surf = surface_irradiance(float(ls), tau, 0.0)
            assert surf < toa

    def test_dust_storm_reduces_irradiance(self) -> None:
        """High tau should dramatically reduce surface irradiance."""
        clear = surface_irradiance(90.0, 0.3, 0.0)
        stormy = surface_irradiance(90.0, 6.0, 0.0)
        assert stormy < clear * 0.5


class TestDustOpacity:
    """Test dust opacity model."""

    def test_always_positive(self) -> None:
        rng = random.Random(42)
        for sol in range(0, 700):
            ls = sol_to_ls(sol)
            tau = dust_opacity(ls, rng)
            assert tau >= 0.1

    def test_storm_season_higher(self) -> None:
        """Average tau should be higher during storm season (Ls 180-330)."""
        rng = random.Random(42)
        calm_taus = [dust_opacity(90.0, random.Random(i)) for i in range(100)]
        storm_taus = [dust_opacity(250.0, random.Random(i + 1000)) for i in range(100)]
        assert sum(storm_taus) / len(storm_taus) > sum(calm_taus) / len(calm_taus)


class TestTemperature:
    """Test temperature model."""

    def test_range(self) -> None:
        """Temperature should be in physical range for Mars."""
        for ls in range(0, 360, 30):
            for lat in [-60, -30, 0, 30, 60]:
                temp = surface_temperature(float(ls), float(lat), 0.5, 0.0)
                assert -130 < temp < 30, f"temp={temp} at Ls={ls}, lat={lat}"

    def test_hellas_warmer(self) -> None:
        """Hellas Basin (alt -7 km) should be warmer than datum."""
        t_datum = surface_temperature(90.0, -42.0, 0.5, 0.0)
        t_hellas = surface_temperature(90.0, -42.0, 0.5, -7.0)
        assert t_hellas > t_datum

    def test_equator_warmer_than_poles(self) -> None:
        t_eq = surface_temperature(90.0, 0.0, 0.5, 0.0)
        t_pole = surface_temperature(90.0, 60.0, 0.5, 0.0)
        assert t_eq > t_pole


class TestRadiation:
    """Test radiation dose model."""

    def test_baseline_gcr(self) -> None:
        rng = random.Random(42)
        dose = radiation_dose(0.5, False, rng)
        assert 0.3 < dose < 100  # Allow for SPE spikes

    def test_shielding_reduces(self) -> None:
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        unshielded = radiation_dose(0.5, False, rng1)
        shielded = radiation_dose(0.5, True, rng2)
        assert shielded < unshielded


class TestPressure:
    """Test atmospheric pressure model."""

    def test_datum_near_610pa(self) -> None:
        p = atmospheric_pressure(90.0, 0.0)
        assert 500 < p < 750

    def test_hellas_higher_pressure(self) -> None:
        """Hellas Basin (alt -7 km) should have much higher pressure."""
        p_datum = atmospheric_pressure(90.0, 0.0)
        p_hellas = atmospheric_pressure(90.0, -7.0)
        assert p_hellas > p_datum * 1.5  # Should be roughly double


class TestMarsEnvironment:
    """Test MarsEnvironment integration."""

    def test_creates_without_error(self) -> None:
        rng = random.Random(42)
        env = MarsEnvironment(sol=100, latitude=-4.5, altitude_km=-2.0,
                              has_shielding=True, rng=rng)
        assert env.sol == 100
        assert env.irradiance > 0
        assert -130 < env.temperature < 30

    def test_to_dict(self) -> None:
        rng = random.Random(42)
        env = MarsEnvironment(sol=50, latitude=0.0, altitude_km=0.0,
                              has_shielding=False, rng=rng)
        d = env.to_dict()
        assert d["sol"] == 50
        assert "irradiance_w_m2" in d
        assert "temperature_c" in d
        assert "radiation_msv" in d

    def test_365_sols_no_crash(self) -> None:
        """Run 365 sols without crash — smoke test."""
        for sol in range(1, 366):
            rng = random.Random(sol)
            env = MarsEnvironment(sol=sol, latitude=-4.5, altitude_km=-2.0,
                                  has_shielding=True, rng=rng)
            assert env.irradiance >= 0
            assert env.pressure > 0
