#!/usr/bin/env python3
"""Tests for mars_env.py — Mars environment model."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars_env import (
    sol_to_ls,
    solar_flux,
    surface_temperature,
    atmospheric_pressure,
    dust_opacity,
    radiation_dose,
    MarsLocation,
    SolConditions,
    OLYMPUS_MONS,
    VALLES_MARINERIS,
    HELLAS_BASIN,
)


def test_sol_to_ls_range():
    """Ls must stay within [0, 360) for all sols."""
    for sol in range(0, 1400):
        ls = sol_to_ls(sol)
        assert 0.0 <= ls < 360.0, f"sol {sol} → Ls {ls} out of range"


def test_sol_to_ls_monotonic_within_year():
    """Ls should be roughly monotonically increasing within a year."""
    prev = sol_to_ls(0)
    wrap_count = 0
    for sol in range(1, 669):
        ls = sol_to_ls(sol)
        if ls < prev:
            wrap_count += 1  # wraps at 360→0
        prev = ls
    assert wrap_count <= 1, f"Ls wrapped {wrap_count} times in one year"


def test_sol_to_ls_full_cycle():
    """One Martian year (~669 sols) should cover ~360° of Ls."""
    ls_0 = sol_to_ls(0)
    ls_669 = sol_to_ls(669)
    # Should be close to where we started (within ~10°)
    diff = abs(ls_669 - ls_0)
    assert diff < 15 or diff > 345, f"Full cycle Ls diff: {diff}"


def test_solar_flux_bounds():
    """Solar flux must be in [0, ~1.2] (>1 possible near perihelion)."""
    for sol in range(0, 669):
        for lat in [-60, -30, 0, 30, 60]:
            flux = solar_flux(sol_to_ls(sol), lat)
            assert 0.0 <= flux <= 1.5, f"sol={sol} lat={lat} flux={flux}"


def test_solar_flux_equator_positive():
    """Equator should always get some solar flux."""
    for sol in range(0, 669):
        flux = solar_flux(sol_to_ls(sol), 0)
        assert flux > 0.3, f"Equator flux too low at sol {sol}: {flux}"


def test_surface_temperature_bounds():
    """Temperature must be physically reasonable for Mars [-160, 30]°C.

    Extreme lats + high elevation can push below -140°C (Olympus summit at -60°).
    """
    for sol in range(0, 669):
        for lat in [-60, 0, 60]:
            for elev in [-7, 0, 20]:
                temp = surface_temperature(sol_to_ls(sol), lat, elev)
                assert -160 <= temp <= 30, f"T={temp} at sol={sol} lat={lat} elev={elev}"


def test_atmospheric_pressure_at_datum():
    """Pressure at datum (0 km) should be ~610 Pa."""
    p = atmospheric_pressure(0.0)
    assert 600 <= p <= 620, f"Datum pressure: {p}"


def test_atmospheric_pressure_altitude():
    """Pressure decreases with altitude, increases in basins."""
    p_high = atmospheric_pressure(20.0)  # Olympus Mons summit
    p_datum = atmospheric_pressure(0.0)
    p_low = atmospheric_pressure(-7.0)  # Hellas basin

    assert p_high < p_datum < p_low
    assert p_low > 800  # Hellas should be > 800 Pa


def test_dust_opacity_baseline():
    """Baseline dust opacity should be reasonable."""
    rng = random.Random(42)
    taus = [dust_opacity(90.0, rng) for _ in range(100)]
    avg = sum(taus) / len(taus)
    assert 0.3 < avg < 2.0, f"Average dust tau: {avg}"


def test_dust_opacity_seasonal():
    """Dust season (Ls~260) should have higher average opacity."""
    rng_calm = random.Random(1)
    rng_storm = random.Random(2)

    calm_taus = [dust_opacity(90.0, random.Random(i)) for i in range(200)]
    storm_taus = [dust_opacity(260.0, random.Random(i + 1000)) for i in range(200)]

    calm_avg = sum(calm_taus) / len(calm_taus)
    storm_avg = sum(storm_taus) / len(storm_taus)
    assert storm_avg > calm_avg, f"Storm season ({storm_avg}) should exceed calm ({calm_avg})"


def test_radiation_dose_shielding():
    """Shielded locations should receive significantly less radiation."""
    rng = random.Random(42)
    unshielded = [radiation_dose(0.5, False, random.Random(i)) for i in range(100)]
    shielded = [radiation_dose(0.5, True, random.Random(i)) for i in range(100)]

    avg_un = sum(unshielded) / len(unshielded)
    avg_sh = sum(shielded) / len(shielded)
    assert avg_sh < avg_un * 0.5, f"Shielded ({avg_sh}) not much less than unshielded ({avg_un})"


def test_radiation_dose_positive():
    """Radiation dose should always be positive."""
    for i in range(200):
        rng = random.Random(i)
        dose = radiation_dose(0.5, False, rng)
        assert dose >= 0, f"Negative dose: {dose}"


def test_mars_location_serialization():
    """MarsLocation should round-trip through to_dict."""
    for loc in [OLYMPUS_MONS, VALLES_MARINERIS, HELLAS_BASIN]:
        d = loc.to_dict()
        assert "name" in d
        assert "latitude" in d
        assert "elevation_km" in d
        assert "pressure_pa" in d
        assert isinstance(d["pressure_pa"], float)


def test_sol_conditions_complete():
    """SolConditions should produce all required fields."""
    rng = random.Random(42)
    cond = SolConditions(100, OLYMPUS_MONS, rng)
    d = cond.to_dict()
    required = {"sol", "ls", "solar_flux", "temperature_c", "pressure_pa",
                "dust_tau", "radiation_msv", "solar_power_factor",
                "is_dust_storm", "is_global_storm"}
    assert required.issubset(d.keys()), f"Missing keys: {required - d.keys()}"


def test_sol_conditions_deterministic():
    """Same seed + sol should produce identical conditions."""
    cond1 = SolConditions(50, HELLAS_BASIN, random.Random(123))
    cond2 = SolConditions(50, HELLAS_BASIN, random.Random(123))
    assert cond1.to_dict() == cond2.to_dict()


def test_three_locations_distinct():
    """The three predefined locations should have different pressures."""
    pressures = [loc.pressure_pa for loc in [OLYMPUS_MONS, VALLES_MARINERIS, HELLAS_BASIN]]
    assert len(set(round(p, 1) for p in pressures)) == 3, "Locations should have distinct pressures"
    # Hellas (deepest) should have highest pressure
    assert HELLAS_BASIN.pressure_pa > VALLES_MARINERIS.pressure_pa > OLYMPUS_MONS.pressure_pa
