#!/usr/bin/env python3
"""Mars environment model — orbital mechanics, weather, radiation.

Computes per-sol environmental conditions for Mars surface locations.
Uses real orbital parameters (Allison & McEwen 2000) for solar longitude,
and empirical temperature/pressure models from Viking/Curiosity/MEDA data.
"""
from __future__ import annotations

import math
import random


# Mars orbital constants
MARS_YEAR_SOLS = 668.6  # sols per Martian year
MARS_OBLIQUITY = 25.19  # degrees
MARS_ECCENTRICITY = 0.0934
PERIHELION_LS = 251.0  # Ls of perihelion (near southern summer solstice)

# Dust storm season: Ls ~180-360 (southern spring/summer)
DUST_STORM_SEASON_START = 180.0
DUST_STORM_SEASON_END = 360.0
REGIONAL_DUST_STORM_PEAK_LS = 260.0

# Radiation constants (mSv/sol)
BASE_GCR_DOSE = 0.67  # Curiosity RAD measurement: ~0.67 mSv/day
SEP_EVENT_DOSE = 50.0  # major solar particle event


def sol_to_ls(sol: int) -> float:
    """Convert sol number to solar longitude (Ls) in degrees.

    Ls 0 = northern vernal equinox
    Ls 90 = northern summer solstice
    Ls 180 = northern autumnal equinox
    Ls 270 = northern winter solstice (southern summer)
    """
    mean_anomaly = (sol / MARS_YEAR_SOLS) * 360.0
    # Equation of center (first-order)
    eoc = (2 * MARS_ECCENTRICITY * math.sin(math.radians(mean_anomaly))
           + 1.25 * MARS_ECCENTRICITY**2 * math.sin(math.radians(2 * mean_anomaly)))
    ls = (mean_anomaly + math.degrees(eoc)) % 360.0
    return ls


def solar_flux(ls: float, latitude: float) -> float:
    """Relative solar flux at surface [0-1] accounting for season and latitude.

    Returns fraction of max possible insolation.
    """
    declination = MARS_OBLIQUITY * math.sin(math.radians(ls))
    lat_rad = math.radians(latitude)
    dec_rad = math.radians(declination)

    cos_zenith = (math.sin(lat_rad) * math.sin(dec_rad)
                  + math.cos(lat_rad) * math.cos(dec_rad))
    cos_zenith = max(0.0, min(1.0, cos_zenith))

    # Distance factor (closer at perihelion)
    true_anomaly = ls - PERIHELION_LS
    r_factor = (1 - MARS_ECCENTRICITY**2) / (1 + MARS_ECCENTRICITY * math.cos(math.radians(true_anomaly)))
    distance_factor = 1.0 / (r_factor ** 2)

    return cos_zenith * distance_factor


def surface_temperature(ls: float, latitude: float, elevation_km: float) -> float:
    """Mean surface temperature in Celsius for given conditions.

    Based on Curiosity/Viking empirical data, adjusted for latitude and elevation.
    """
    # Base annual mean: ~-60°C at equator
    base_temp = -60.0

    # Seasonal variation: ±30°C at mid-latitudes
    season_amp = 25.0 * math.cos(math.radians(latitude))
    seasonal = season_amp * math.sin(math.radians(ls - 70))  # peak near Ls=160

    # Latitude effect: -0.5°C per degree from equator
    lat_effect = -0.5 * abs(latitude)

    # Elevation: lapse rate ~-2.5°C/km (thinner atmosphere)
    elev_effect = -2.5 * elevation_km

    return base_temp + seasonal + lat_effect + elev_effect


def atmospheric_pressure(elevation_km: float) -> float:
    """Surface atmospheric pressure in Pa at given elevation.

    Mars mean ~610 Pa at datum. Scale height ~11.1 km.
    """
    return 610.0 * math.exp(-elevation_km / 11.1)


def dust_opacity(ls: float, rng: random.Random) -> float:
    """Atmospheric dust opacity (tau) for the sol.

    Normal: tau ~0.5-1.0
    Regional storm: tau ~2-4
    Global storm: tau ~6-10 (rare)
    """
    base_tau = 0.5

    # Seasonal enhancement during dust season
    if DUST_STORM_SEASON_START <= ls <= DUST_STORM_SEASON_END:
        season_factor = 1.0 + 1.5 * math.exp(
            -((ls - REGIONAL_DUST_STORM_PEAK_LS) ** 2) / (2 * 30**2)
        )
        base_tau *= season_factor

    # Stochastic dust events
    if rng.random() < 0.005:  # ~0.5% chance per sol of regional storm
        base_tau += rng.uniform(1.5, 3.5)

    if rng.random() < 0.0003:  # ~0.03% chance per sol of global storm
        base_tau += rng.uniform(5.0, 9.0)

    return round(base_tau, 3)


def radiation_dose(dust_tau: float, has_shielding: bool, rng: random.Random) -> float:
    """Radiation dose in mSv for the sol.

    Dust attenuates GCR slightly. Shielding reduces by ~80%.
    SEP events are rare but intense.
    """
    # GCR: slightly reduced by dust (dust scatters some particles)
    gcr = BASE_GCR_DOSE * (1.0 - 0.05 * min(dust_tau, 5.0))

    # Solar particle event (~2-3 per Mars year, each lasting 1-3 sols)
    sep = 0.0
    if rng.random() < 0.004:
        sep = SEP_EVENT_DOSE * rng.uniform(0.3, 1.0)

    total = gcr + sep

    if has_shielding:
        total *= 0.2  # 80% reduction from regolith/water shielding

    return round(total, 3)


class MarsLocation:
    """A specific location on Mars with fixed geography."""

    def __init__(self, name: str, latitude: float, elevation_km: float,
                 has_caves: bool = False, has_ice: bool = False):
        self.name = name
        self.latitude = latitude
        self.elevation_km = elevation_km
        self.has_caves = has_caves
        self.has_ice = has_ice
        self.pressure_pa = atmospheric_pressure(elevation_km)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "name": self.name,
            "latitude": self.latitude,
            "elevation_km": self.elevation_km,
            "has_caves": self.has_caves,
            "has_ice": self.has_ice,
            "pressure_pa": round(self.pressure_pa, 1),
        }


class SolConditions:
    """Environmental conditions for a single sol at a location."""

    def __init__(self, sol: int, location: MarsLocation, rng: random.Random):
        self.sol = sol
        self.ls = sol_to_ls(sol)
        self.solar = solar_flux(self.ls, location.latitude)
        self.temperature_c = surface_temperature(
            self.ls, location.latitude, location.elevation_km
        )
        self.pressure_pa = location.pressure_pa
        self.dust_tau = dust_opacity(self.ls, rng)
        self.radiation_msv = radiation_dose(
            self.dust_tau, location.has_caves, rng
        )
        # Solar power factor: reduced by dust and season
        self.solar_power_factor = max(0.05, self.solar * math.exp(-0.3 * self.dust_tau))
        self.is_dust_storm = self.dust_tau > 2.0
        self.is_global_storm = self.dust_tau > 6.0

    def to_dict(self) -> dict:
        """Serialize for state output."""
        return {
            "sol": self.sol,
            "ls": round(self.ls, 2),
            "solar_flux": round(self.solar, 4),
            "temperature_c": round(self.temperature_c, 1),
            "pressure_pa": round(self.pressure_pa, 1),
            "dust_tau": self.dust_tau,
            "radiation_msv": self.radiation_msv,
            "solar_power_factor": round(self.solar_power_factor, 4),
            "is_dust_storm": self.is_dust_storm,
            "is_global_storm": self.is_global_storm,
        }


# Pre-defined Mars locations for the 3 colonies
OLYMPUS_MONS = MarsLocation(
    "Olympus Mons Foothills", latitude=18.65, elevation_km=5.0,
    has_caves=False, has_ice=True,
)

VALLES_MARINERIS = MarsLocation(
    "Valles Marineris Caverns", latitude=-13.9, elevation_km=-4.0,
    has_caves=True, has_ice=True,
)

HELLAS_BASIN = MarsLocation(
    "Hellas Planitia Hub", latitude=-42.7, elevation_km=-7.15,
    has_caves=False, has_ice=True,
)
