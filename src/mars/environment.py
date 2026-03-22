"""Mars environment model — solar irradiance, dust, radiation, temperature.

All values are based on published NASA/ESA Mars mission data:
- Solar constant at Mars orbit: ~590 W/m² (varies 493–718 W/m² with eccentricity)
- Surface pressure: ~610 Pa (varies ±10% seasonally)
- GCR dose rate: ~0.67 mSv/day on surface
- Dust storm season: Ls 180–330 (southern spring/summer)
- Temperature: -60°C mean, range -125°C to +20°C
"""
from __future__ import annotations

import math
import random


# Mars orbital constants
MARS_YEAR_SOLS = 668.6  # sols per Mars year
SOLAR_CONSTANT_AU1 = 1361.0  # W/m² at 1 AU
MARS_SEMI_MAJOR_AU = 1.524
MARS_ECCENTRICITY = 0.0934
MARS_PERIHELION_LS = 251.0  # Ls of perihelion

# Surface conditions
MEAN_PRESSURE_PA = 610.0
MEAN_TEMP_C = -60.0
GCR_BASELINE_MSV_DAY = 0.67  # galactic cosmic ray dose


def sol_to_ls(sol: int) -> float:
    """Convert sol number to solar longitude Ls (0–360°).

    Ls 0 = northern spring equinox. Simplified linear mapping.
    """
    return (sol / MARS_YEAR_SOLS * 360.0) % 360.0


def solar_distance_au(ls: float) -> float:
    """Mars-Sun distance in AU at given Ls.

    Uses Kepler's equation (first-order approximation).
    """
    # True anomaly relative to perihelion
    theta = math.radians(ls - MARS_PERIHELION_LS)
    r = MARS_SEMI_MAJOR_AU * (1 - MARS_ECCENTRICITY**2) / (
        1 + MARS_ECCENTRICITY * math.cos(theta)
    )
    return r


def top_of_atmosphere_irradiance(ls: float) -> float:
    """Solar irradiance at top of Mars atmosphere (W/m²)."""
    r = solar_distance_au(ls)
    return SOLAR_CONSTANT_AU1 / (r * r)


def dust_opacity(ls: float, rng: random.Random) -> float:
    """Atmospheric dust optical depth (tau).

    Baseline tau ~0.5. Dust storm season (Ls 180–330) can spike to 2–8.
    Regional storms: tau 2–4. Global storms: tau 5–8 (rare).
    """
    # Seasonal baseline
    if 180 <= ls <= 330:
        base_tau = 0.8 + 0.5 * math.sin(math.radians((ls - 180) * 1.2))
    else:
        base_tau = 0.3 + 0.2 * math.sin(math.radians(ls))

    # Random weather variation
    weather_noise = rng.gauss(0, 0.15)

    # Dust storm probability (regional or global)
    storm_tau = 0.0
    if 180 <= ls <= 330:
        if rng.random() < 0.03:  # ~3% per sol chance of regional storm
            storm_tau = rng.uniform(1.5, 4.0)
        if rng.random() < 0.002:  # ~0.2% chance of global storm onset
            storm_tau = rng.uniform(5.0, 8.0)

    return max(0.1, base_tau + weather_noise + storm_tau)


def surface_irradiance(ls: float, tau: float, latitude_deg: float) -> float:
    """Solar irradiance reaching the surface (W/m²).

    Accounts for atmospheric absorption (Beer-Lambert) and latitude.
    """
    toa = top_of_atmosphere_irradiance(ls)
    # Atmospheric transmission (Beer-Lambert with diffuse correction)
    transmission = math.exp(-0.9 * tau) + 0.1 * math.exp(-0.1 * tau)
    # Solar declination (simplified)
    declination = 25.19 * math.sin(math.radians(ls))
    # Effective solar angle
    lat_rad = math.radians(latitude_deg)
    decl_rad = math.radians(declination)
    cos_zenith = max(0.1, math.sin(lat_rad) * math.sin(decl_rad) +
                      math.cos(lat_rad) * math.cos(decl_rad))
    # Average over a sol (integrate day/night) — roughly 0.35× peak
    daily_avg_factor = 0.35
    return toa * transmission * cos_zenith * daily_avg_factor


def surface_temperature(ls: float, latitude_deg: float,
                        tau: float, altitude_km: float) -> float:
    """Mean surface temperature (°C) for a sol.

    Higher altitude = colder. Higher tau = warmer nights but cooler days.
    Hellas Basin (alt ~ -7 km) is warmest. Equatorial sites warmest.
    """
    # Seasonal variation
    seasonal = 15.0 * math.sin(math.radians(ls - 70))
    # Latitude effect (~0.5°C per degree from equator)
    lat_effect = -0.5 * abs(latitude_deg)
    # Altitude lapse rate (~1.5°C per km, inverted: deeper = warmer)
    alt_effect = -1.5 * altitude_km
    # Dust insulation (high tau warms surface slightly)
    dust_effect = 2.0 * max(0, tau - 0.5)
    return MEAN_TEMP_C + seasonal + lat_effect + alt_effect + dust_effect


def radiation_dose(tau: float, has_shielding: bool, rng: random.Random) -> float:
    """Radiation dose rate (mSv/day).

    Dust attenuates GCR slightly. Shielding (regolith/water) reduces by ~80%.
    Solar particle events (SPE) can spike dose 10–100×.
    """
    # GCR modulated by atmospheric shielding
    gcr = GCR_BASELINE_MSV_DAY * (1.0 - 0.05 * min(tau, 5.0))

    # Solar particle event (rare but dangerous)
    spe = 0.0
    if rng.random() < 0.005:  # ~0.5% per sol
        spe = rng.uniform(5.0, 50.0)

    total = gcr + spe

    # Habitat shielding
    if has_shielding:
        total *= 0.2  # 80% reduction from regolith/water shielding

    return total


def atmospheric_pressure(ls: float, altitude_km: float) -> float:
    """Surface pressure (Pa) at given altitude and season.

    Pressure varies ±10% seasonally due to CO2 sublimation/deposition.
    Lower altitude = higher pressure (Hellas Basin ~1100 Pa).
    """
    seasonal = MEAN_PRESSURE_PA * 0.1 * math.sin(math.radians(ls - 140))
    # Barometric formula (scale height ~11 km)
    pressure = (MEAN_PRESSURE_PA + seasonal) * math.exp(-altitude_km / 11.0)
    return max(100.0, pressure)


class MarsEnvironment:
    """Mars environment state for a single sol."""

    __slots__ = ("sol", "ls", "tau", "irradiance", "temperature",
                 "radiation", "pressure", "dust_storm", "event")

    def __init__(self, sol: int, latitude: float, altitude_km: float,
                 has_shielding: bool, rng: random.Random) -> None:
        self.sol = sol
        self.ls = sol_to_ls(sol)
        self.tau = dust_opacity(self.ls, rng)
        self.dust_storm = self.tau > 3.0
        self.irradiance = surface_irradiance(self.ls, self.tau, latitude)
        self.temperature = surface_temperature(
            self.ls, latitude, self.tau, altitude_km
        )
        self.radiation = radiation_dose(self.tau, has_shielding, rng)
        self.pressure = atmospheric_pressure(self.ls, altitude_km)
        self.event = self._roll_event(rng)

    def _roll_event(self, rng: random.Random) -> str | None:
        """Roll for a random environmental event."""
        if self.dust_storm:
            return "dust_storm"
        roll = rng.random()
        if roll < 0.005:
            return "meteorite_impact"
        if roll < 0.015:
            return "solar_particle_event"
        if roll < 0.025:
            return "equipment_failure"
        if roll < 0.04:
            return "scientific_discovery"
        return None

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "sol": self.sol,
            "ls": round(self.ls, 1),
            "tau": round(self.tau, 3),
            "irradiance_w_m2": round(self.irradiance, 1),
            "temperature_c": round(self.temperature, 1),
            "radiation_msv": round(self.radiation, 3),
            "pressure_pa": round(self.pressure, 1),
            "dust_storm": self.dust_storm,
            "event": self.event,
        }
