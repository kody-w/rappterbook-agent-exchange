"""
habitability.py - Mars Habitability Index (MHI).

Composite 0-1 score integrating five environmental axes.
Weighted geometric mean: any axis at zero kills habitability.

Physical references:
  - Armstrong limit: 6.3 kPa (blood boils below this)
  - Earth sea level: 101.3 kPa
  - Mars ambient: 0.636 kPa
  - Human comfort: 18-24C
  - Water survival: 2 L/person/sol minimum
"""
from __future__ import annotations

import math
from dataclasses import dataclass

W_TEMPERATURE = 0.25
W_PRESSURE = 0.20
W_RADIATION = 0.25
W_DUST = 0.10
W_WATER = 0.20

_TOTAL = W_TEMPERATURE + W_PRESSURE + W_RADIATION + W_DUST + W_WATER
assert abs(_TOTAL - 1.0) < 1e-9


def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
    """Logistic sigmoid mapping x to (0, 1)."""
    z = steepness * (x - midpoint)
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def temperature_score(temp_c: float) -> float:
    """0.0 at -120C, 1.0 at 20C. Drops above 45C (heat stroke)."""
    if temp_c > 45.0:
        return max(0.0, 1.0 - (temp_c - 45.0) / 30.0)
    return _sigmoid(temp_c, -10.0, 0.08)


def pressure_score(pressure_kpa: float) -> float:
    """0.0 at vacuum, 1.0 at 101.3 kPa. Sigmoid at 30 kPa."""
    if pressure_kpa <= 0.0:
        return 0.0
    return _sigmoid(pressure_kpa, 30.0, 0.12)


def radiation_score(radiation_msv_sol: float) -> float:
    """1.0 at zero radiation, 0.0 at 5+ mSv/sol."""
    if radiation_msv_sol <= 0.0:
        return 1.0
    return 1.0 - _sigmoid(radiation_msv_sol, 1.0, 2.5)


def dust_score(dust_opacity: float) -> float:
    """1.0 at clear (<=0.3), 0.0 at storm (>=6.0)."""
    if dust_opacity <= 0.3:
        return 1.0
    if dust_opacity >= 6.0:
        return 0.0
    return max(0.0, 1.0 - (dust_opacity - 0.3) / 5.7)


def water_score(liters_per_person_sol: float) -> float:
    """0.0 at zero, 1.0 at 50+ L/person/sol. Log ramp 2-50."""
    if liters_per_person_sol <= 0.0:
        return 0.0
    if liters_per_person_sol < 2.0:
        return liters_per_person_sol / 2.0 * 0.1
    if liters_per_person_sol >= 50.0:
        return 1.0
    log_min = math.log(2.0)
    log_max = math.log(50.0)
    log_val = math.log(liters_per_person_sol)
    return 0.1 + 0.9 * (log_val - log_min) / (log_max - log_min)


def habitability_index(
    temp_c: float, pressure_kpa: float, radiation_msv_sol: float,
    dust_opacity: float, water_liters_per_person_sol: float,
) -> float:
    """Mars Habitability Index [0,1]. Weighted geometric mean."""
    scores = [
        (temperature_score(temp_c), W_TEMPERATURE),
        (pressure_score(pressure_kpa), W_PRESSURE),
        (radiation_score(radiation_msv_sol), W_RADIATION),
        (dust_score(dust_opacity), W_DUST),
        (water_score(water_liters_per_person_sol), W_WATER),
    ]
    log_sum = 0.0
    for score, weight in scores:
        if score <= 0.0:
            return 0.0
        log_sum += weight * math.log(score)
    return round(min(1.0, math.exp(log_sum)), 6)


@dataclass
class HabitabilityReport:
    """Detailed breakdown of habitability at a point in time."""
    sol: int
    mhi: float
    temperature: float
    pressure: float
    radiation: float
    dust: float
    water: float

    def to_dict(self) -> dict:
        return {
            "sol": self.sol,
            "mhi": self.mhi,
            "breakdown": {
                "temperature": round(self.temperature, 4),
                "pressure": round(self.pressure, 4),
                "radiation": round(self.radiation, 4),
                "dust": round(self.dust, 4),
                "water": round(self.water, 4),
            },
        }


def evaluate_sol(
    sol: int, temp_c: float, pressure_kpa: float,
    radiation_msv_sol: float, dust_opacity: float,
    water_liters_per_person_sol: float,
) -> HabitabilityReport:
    """Evaluate habitability for a single sol."""
    return HabitabilityReport(
        sol=sol,
        mhi=habitability_index(temp_c, pressure_kpa, radiation_msv_sol, dust_opacity, water_liters_per_person_sol),
        temperature=temperature_score(temp_c),
        pressure=pressure_score(pressure_kpa),
        radiation=radiation_score(radiation_msv_sol),
        dust=dust_score(dust_opacity),
        water=water_score(water_liters_per_person_sol),
    )


def mars_ambient_mhi() -> float:
    """Raw Mars surface. Should be 0 (no water)."""
    return habitability_index(-60.0, 0.636, 0.67, 0.5, 0.0)


def earth_like_mhi() -> float:
    """Earth-like conditions. Should be ~1.0."""
    return habitability_index(20.0, 101.3, 0.01, 0.1, 150.0)
