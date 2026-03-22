"""
Mars environment model — radiation, dust storms, temperature, solar flux.

All values sourced from NASA Mars Fact Sheet and MSL/MEDA mission data.
One tick = one sol (24h 37m 22.7s).
"""
from __future__ import annotations

import math
import random


# --- Physical constants ---
SOLS_PER_MARS_YEAR = 668.6
MEAN_TEMP_C = -60.0
TEMP_AMPLITUDE_C = 50.0  # seasonal swing ±50°C
DIURNAL_AMPLITUDE_C = 35.0  # day/night swing (averaged out per sol)
BASE_SOLAR_FLUX_WM2 = 590.0  # Mars orbit average W/m²
BASE_RADIATION_MSV_SOL = 0.67  # GCR background mSv/sol (Curiosity RAD)
SOLAR_FLARE_EXTRA_MSV = 5.0  # moderate SPE
ATMOSPHERIC_PRESSURE_KPA = 0.636


def sol_to_ls(sol: int) -> float:
    """Convert sol number to solar longitude Ls (0–360°).

    Simplified: linear mapping. Real orbit is slightly eccentric but
    this is good enough for population-model fidelity.
    """
    return (sol / SOLS_PER_MARS_YEAR * 360.0) % 360.0


def season_name(ls: float) -> str:
    """Human-readable season from Ls."""
    if ls < 90:
        return "spring"
    if ls < 180:
        return "summer"
    if ls < 270:
        return "autumn"
    return "winter"


def surface_temperature_c(ls: float) -> float:
    """Mean surface temperature for the sol, driven by Ls.

    Returns midpoint of diurnal range (night is colder, day warmer).
    """
    seasonal = TEMP_AMPLITUDE_C * math.sin(math.radians(ls - 70))
    return MEAN_TEMP_C + seasonal


def solar_flux_wm2(ls: float, dust_opacity: float) -> float:
    """Available solar flux after dust attenuation.

    dust_opacity: 0 = clear, 1 = global storm (tau ~ 8).
    Beer-Lambert: flux = base * exp(-tau). tau ∈ [0.3, 8].
    """
    tau = 0.3 + dust_opacity * 7.7
    return BASE_SOLAR_FLUX_WM2 * math.exp(-tau)


def radiation_msv(dust_opacity: float, flare: bool) -> float:
    """Daily radiation dose in mSv.

    Dust actually *reduces* GCR (shielding), but flares add.
    """
    gcr = BASE_RADIATION_MSV_SOL * (1.0 - 0.15 * dust_opacity)
    spe = SOLAR_FLARE_EXTRA_MSV if flare else 0.0
    return gcr + spe


class DustStorm:
    """Active dust storm tracker."""

    __slots__ = ("kind", "remaining_sols", "peak_opacity")

    def __init__(self, kind: str, duration: int, peak_opacity: float) -> None:
        self.kind = kind  # "regional" or "global"
        self.remaining_sols = duration
        self.peak_opacity = peak_opacity

    def opacity(self) -> float:
        """Current opacity (ramps up then down)."""
        return self.peak_opacity * min(1.0, self.remaining_sols / 5.0)

    def tick(self) -> bool:
        """Advance one sol. Returns True if storm still active."""
        self.remaining_sols -= 1
        return self.remaining_sols > 0


class MarsEnvironment:
    """Mars environment state machine — advance one sol at a time."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.sol = 0
        self.storm: DustStorm | None = None
        self.flare = False

    def dust_opacity(self) -> float:
        """Current dust opacity [0, 1]."""
        if self.storm is None:
            return 0.0
        return self.storm.opacity()

    def tick(self) -> dict:
        """Advance one sol. Returns environment snapshot."""
        self.sol += 1
        ls = sol_to_ls(self.sol)

        # --- Dust storm generation ---
        if self.storm is not None:
            alive = self.storm.tick()
            if not alive:
                self.storm = None

        if self.storm is None:
            storm_season = 180 <= ls <= 330
            if storm_season:
                r = self.rng.random()
                if r < 0.005:  # global storm ~0.5%/sol
                    dur = self.rng.randint(30, 80)
                    self.storm = DustStorm("global", dur, 0.9)
                elif r < 0.05:  # regional storm ~5%/sol
                    dur = self.rng.randint(5, 20)
                    self.storm = DustStorm("regional", dur, 0.4)

        # --- Solar flare ---
        self.flare = self.rng.random() < 0.003  # ~1/year

        dust = self.dust_opacity()
        temp = surface_temperature_c(ls)
        flux = solar_flux_wm2(ls, dust)
        rad = radiation_msv(dust, self.flare)

        return {
            "sol": self.sol,
            "ls": round(ls, 2),
            "season": season_name(ls),
            "temperature_c": round(temp, 1),
            "solar_flux_wm2": round(flux, 1),
            "dust_opacity": round(dust, 3),
            "radiation_msv": round(rad, 3),
            "storm": self.storm.kind if self.storm else None,
            "flare": self.flare,
            "pressure_kpa": round(ATMOSPHERIC_PRESSURE_KPA * (1 + 0.1 * math.sin(math.radians(ls))), 3),
        }
