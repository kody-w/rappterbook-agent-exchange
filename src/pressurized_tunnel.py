"""pressurized_tunnel.py -- Mars Inter-Habitat Pressurized Tunnel System.

The colony has habs, greenhouses, fabrication bays, and landing pads --
all isolated.  Crew must EVA between buildings: suit up, depressurize,
walk, repressurize, de-suit.  30 minutes each way.  5 times per day.
That is 2.5 crew-hours lost to WALKING.  On Mars, time is oxygen.

This module simulates inflatable pressurized tunnels connecting hab
modules.  Kevlar/Vectran fabric with polyethylene liner.  Crew walks
between buildings in shirtsleeves.  The colony becomes an organism
instead of a scattering of cells.

Physics modelled
----------------
* Cylindrical pressure vessel -- hoop stress sigma = P*r/t.
  Burst margin = (sigma_yield / sigma_hoop).  Must exceed 4:1.

* Leak rate -- through seals and micrometeorite punctures.
  Q_leak = C_d * A_hole * sqrt(2 * dP / rho).  Orifice flow.

* Thermal loss -- conduction through fabric wall + radiation.
  Q_cond = k * A * dT / thickness.  Q_rad = eps*sigma*A*(T^4 - T_amb^4).

* Air volume -- V = pi * r^2 * L.  Each tunnel segment holds breathable
  atmosphere that must be maintained by life support.

* UV degradation -- Martian UV (no ozone) degrades Kevlar.
  Strength loss ~0.1% per sol exposed.  Regolith shielding extends life.

* Seal fatigue -- each pressure equalization cycle stresses tunnel seals.
  Fatigue life ~10,000 cycles before replacement.

Conservation laws: pressure >= 0, air_mass >= 0, fabric_health in [0,1],
leak_rate >= 0, thermal_loss >= 0, burst_margin > 0 when pressurized.

Reference: NASA TransHab inflatable module design, Bigelow B330.
One tick = one sol.  Lengths in metres, pressures in kPa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ── Physical constants ──────────────────────────────────────────────────

STEFAN_BOLTZMANN = 5.670374419e-8     # W/(m^2 * K^4)
MARS_AMBIENT_TEMP_K = 210.0          # mean surface temperature
MARS_AMBIENT_PRESSURE_KPA = 0.636    # CO2 atmosphere
AIR_DENSITY_KG_M3 = 1.15             # hab-pressure air at 34 kPa, ~20 °C
SECONDS_PER_SOL = 88_775.0
HOURS_PER_SOL = 24.66

# ── Tunnel geometry defaults ────────────────────────────────────────────

DEFAULT_INNER_RADIUS_M = 1.2         # comfortable walking diameter 2.4 m
DEFAULT_LENGTH_M = 50.0              # 50 m between habs
DEFAULT_WALL_THICKNESS_M = 0.012     # 12 mm multi-layer fabric

# ── Structural parameters ──────────────────────────────────────────────

KEVLAR_YIELD_STRENGTH_MPA = 3600.0   # Kevlar 49 tensile strength
SAFETY_FACTOR_MINIMUM = 4.0          # NASA crewed pressure vessel standard
HAB_PRESSURE_KPA = 34.0              # same as ISS-like low-pressure hab

# ── Thermal parameters ─────────────────────────────────────────────────

FABRIC_THERMAL_CONDUCTIVITY = 0.04   # W/(m*K) -- multi-layer insulated
FABRIC_EMISSIVITY = 0.85             # outer surface IR emissivity
HAB_INTERIOR_TEMP_K = 293.0          # 20 °C interior target

# ── Degradation parameters ─────────────────────────────────────────────

UV_DEGRADATION_PER_SOL = 0.001       # 0.1% strength loss per sol
UV_SHIELDING_FACTOR = 0.05           # regolith shielding reduces to 5%
SEAL_FATIGUE_PER_CYCLE = 0.0001      # 0.01% wear per pressure cycle
SEAL_REPLACEMENT_THRESHOLD = 0.20    # replace below 20% health
MICROMETEORITE_HOLE_RATE = 0.0005    # probability of puncture per sol per m^2

# ── Leak parameters ─────────────────────────────────────────────────────

SEAL_LEAK_AREA_M2 = 1.0e-7           # baseline seal leak cross-section
DISCHARGE_COEFFICIENT = 0.61         # sharp-edged orifice
PUNCTURE_AREA_M2 = 1.0e-6            # typical micrometeorite hole


# ── Pure physics functions ──────────────────────────────────────────────

def hoop_stress_mpa(pressure_kpa: float, radius_m: float,
                     wall_thickness_m: float) -> float:
    """Hoop stress in thin-walled cylinder: sigma = P*r/t (MPa).

    Pressure in kPa, result in MPa.
    """
    if wall_thickness_m <= 0.0:
        raise ValueError(f"wall thickness must be > 0, got {wall_thickness_m}")
    if radius_m < 0.0:
        raise ValueError(f"radius must be >= 0, got {radius_m}")
    pressure_mpa = pressure_kpa / 1000.0
    return pressure_mpa * radius_m / wall_thickness_m


def burst_margin(pressure_kpa: float, radius_m: float,
                  wall_thickness_m: float,
                  fabric_health: float = 1.0) -> float:
    """Safety factor: yield_strength * health / hoop_stress.

    Returns ratio; must be > SAFETY_FACTOR_MINIMUM for crewed ops.
    """
    stress = hoop_stress_mpa(pressure_kpa, radius_m, wall_thickness_m)
    if stress <= 0.0:
        return float("inf")
    effective_strength = KEVLAR_YIELD_STRENGTH_MPA * max(0.0, fabric_health)
    return effective_strength / stress


def tunnel_volume_m3(radius_m: float, length_m: float) -> float:
    """Internal volume of cylindrical tunnel (m^3)."""
    if radius_m < 0.0 or length_m < 0.0:
        raise ValueError("dimensions must be >= 0")
    return math.pi * radius_m ** 2 * length_m


def tunnel_surface_area_m2(radius_m: float, length_m: float) -> float:
    """Lateral surface area of cylinder (m^2), excluding end caps."""
    if radius_m < 0.0 or length_m < 0.0:
        raise ValueError("dimensions must be >= 0")
    return 2.0 * math.pi * radius_m * length_m


def air_mass_kg(volume_m3: float,
                density: float = AIR_DENSITY_KG_M3) -> float:
    """Mass of air in tunnel at given density (kg)."""
    return max(0.0, volume_m3 * density)


def orifice_leak_rate_kg_s(hole_area_m2: float,
                            pressure_diff_kpa: float,
                            density: float = AIR_DENSITY_KG_M3) -> float:
    """Mass flow rate through orifice: Q = Cd * A * sqrt(2 * dP / rho).

    Returns kg/s.  dP in kPa converted to Pa.
    """
    if hole_area_m2 <= 0.0 or pressure_diff_kpa <= 0.0:
        return 0.0
    dp_pa = pressure_diff_kpa * 1000.0
    velocity = math.sqrt(2.0 * dp_pa / density)
    return DISCHARGE_COEFFICIENT * hole_area_m2 * density * velocity


def thermal_conduction_loss_w(surface_area_m2: float,
                               wall_thickness_m: float,
                               interior_temp_k: float = HAB_INTERIOR_TEMP_K,
                               exterior_temp_k: float = MARS_AMBIENT_TEMP_K
                               ) -> float:
    """Heat loss by conduction through tunnel wall (W).

    Q = k * A * dT / thickness.
    """
    if wall_thickness_m <= 0.0:
        return 0.0
    dt = max(0.0, interior_temp_k - exterior_temp_k)
    return (FABRIC_THERMAL_CONDUCTIVITY * surface_area_m2 * dt
            / wall_thickness_m)


def thermal_radiation_loss_w(surface_area_m2: float,
                              exterior_temp_k: float = MARS_AMBIENT_TEMP_K
                              ) -> float:
    """Radiative heat loss from tunnel exterior (W).

    Q = eps * sigma * A * (T_surf^4 - T_amb^4).
    Assume outer surface at ~250 K (between interior and ambient).
    """
    surface_temp_k = (HAB_INTERIOR_TEMP_K + exterior_temp_k) / 2.0
    if surface_temp_k <= exterior_temp_k:
        return 0.0
    return (FABRIC_EMISSIVITY * STEFAN_BOLTZMANN * surface_area_m2
            * (surface_temp_k ** 4 - exterior_temp_k ** 4))


def total_thermal_loss_w(surface_area_m2: float,
                          wall_thickness_m: float) -> float:
    """Combined conduction + radiation heat loss (W)."""
    cond = thermal_conduction_loss_w(surface_area_m2, wall_thickness_m)
    rad = thermal_radiation_loss_w(surface_area_m2)
    return cond + rad


def thermal_loss_kwh_per_sol(surface_area_m2: float,
                              wall_thickness_m: float) -> float:
    """Heat energy lost per sol (kWh)."""
    watts = total_thermal_loss_w(surface_area_m2, wall_thickness_m)
    return watts * SECONDS_PER_SOL / 3_600_000.0


def fabric_health_after_uv(current_health: float,
                             sols: int = 1,
                             shielded: bool = False) -> float:
    """Fabric strength remaining after UV exposure.

    Regolith shielding reduces UV degradation to 5% of unshielded rate.
    """
    rate = UV_DEGRADATION_PER_SOL
    if shielded:
        rate *= UV_SHIELDING_FACTOR
    return max(0.0, current_health - rate * sols)


def seal_health_after_cycles(current_health: float,
                              cycles: int = 1) -> float:
    """Seal integrity after pressure equalization cycles."""
    return max(0.0, current_health - SEAL_FATIGUE_PER_CYCLE * cycles)


def check_micrometeorite(surface_area_m2: float,
                          current_punctures: int,
                          rng_value: float = 0.5) -> int:
    """Deterministic puncture check.  Returns new puncture count.

    rng_value in [0,1) simulates random roll.  Probability scales
    with surface area.
    """
    prob = min(1.0, MICROMETEORITE_HOLE_RATE * surface_area_m2)
    if rng_value < prob:
        return current_punctures + 1
    return current_punctures


# ── State dataclass ─────────────────────────────────────────────────────

@dataclass
class PressurizedTunnel:
    """State of one pressurized tunnel segment."""
    sol: int = 0
    inner_radius_m: float = DEFAULT_INNER_RADIUS_M
    length_m: float = DEFAULT_LENGTH_M
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M
    internal_pressure_kpa: float = HAB_PRESSURE_KPA
    fabric_health: float = 1.0
    seal_health: float = 1.0
    regolith_shielded: bool = False
    puncture_count: int = 0
    pressure_cycles_today: int = 5    # typical daily equalization events

    # Per-sol outputs
    air_leaked_kg: float = 0.0
    thermal_loss_kwh: float = 0.0
    burst_margin_ratio: float = 0.0
    operational: bool = True

    # Cumulative
    cumulative_air_lost_kg: float = 0.0
    cumulative_thermal_kwh: float = 0.0
    total_pressure_cycles: int = 0

    events: list[str] = field(default_factory=list)


@dataclass
class TickResult:
    """Immutable snapshot of one sol for a tunnel segment."""
    sol: int = 0
    air_leaked_kg: float = 0.0
    thermal_loss_kwh: float = 0.0
    burst_margin_ratio: float = 0.0
    fabric_health: float = 1.0
    seal_health: float = 1.0
    puncture_count: int = 0
    operational: bool = True
    events: list[str] = field(default_factory=list)


# ── Tick function ───────────────────────────────────────────────────────

def tick(state: PressurizedTunnel, rng_value: float = 0.999) -> TickResult:
    """Advance one tunnel segment by one sol.

    Checks structural integrity, calculates leaks, thermal loss,
    UV degradation, seal wear, and micrometeorite risk.
    """
    state.sol += 1
    events: list[str] = []
    operational = True

    surface = tunnel_surface_area_m2(state.inner_radius_m, state.length_m)
    volume = tunnel_volume_m3(state.inner_radius_m, state.length_m)

    # ── Structural check ──
    margin = burst_margin(state.internal_pressure_kpa,
                          state.inner_radius_m,
                          state.wall_thickness_m,
                          state.fabric_health)
    state.burst_margin_ratio = margin

    if margin < SAFETY_FACTOR_MINIMUM:
        events.append(f"BURST MARGIN LOW -- {margin:.1f}x (need {SAFETY_FACTOR_MINIMUM:.0f}x)")
        if margin < 2.0:
            events.append("CRITICAL -- depressurize immediately")
            operational = False

    # ── Seal check ──
    if state.seal_health <= SEAL_REPLACEMENT_THRESHOLD:
        events.append("SEAL WORN -- replacement required")
        operational = False

    if not operational:
        state.air_leaked_kg = 0.0
        state.thermal_loss_kwh = 0.0
        state.operational = False
        state.events = events
        return TickResult(sol=state.sol, burst_margin_ratio=margin,
                          fabric_health=state.fabric_health,
                          seal_health=state.seal_health,
                          puncture_count=state.puncture_count,
                          operational=False, events=list(events))

    # ── Micrometeorite check ──
    old_punctures = state.puncture_count
    state.puncture_count = check_micrometeorite(
        surface, state.puncture_count, rng_value)
    if state.puncture_count > old_punctures:
        events.append("IMPACT -- micrometeorite puncture detected!")

    # ── Leak calculation ──
    dp = state.internal_pressure_kpa - MARS_AMBIENT_PRESSURE_KPA
    seal_leak = orifice_leak_rate_kg_s(
        SEAL_LEAK_AREA_M2 * (2.0 - state.seal_health), dp)
    puncture_leak = orifice_leak_rate_kg_s(
        PUNCTURE_AREA_M2 * state.puncture_count, dp)
    total_leak_rate = seal_leak + puncture_leak
    air_lost = total_leak_rate * SECONDS_PER_SOL
    state.air_leaked_kg = air_lost
    state.cumulative_air_lost_kg += air_lost

    total_air = air_mass_kg(volume)
    if total_air > 0 and air_lost > total_air * 0.10:
        events.append("AIR LOSS HIGH -- >10% volume per sol")

    # ── Thermal loss ──
    t_loss = thermal_loss_kwh_per_sol(surface, state.wall_thickness_m)
    state.thermal_loss_kwh = t_loss
    state.cumulative_thermal_kwh += t_loss

    # ── UV degradation ──
    old_fabric = state.fabric_health
    state.fabric_health = fabric_health_after_uv(
        state.fabric_health, sols=1, shielded=state.regolith_shielded)
    if old_fabric >= 0.50 and state.fabric_health < 0.50:
        events.append("FABRIC WARNING -- health below 50%")

    # ── Seal wear ──
    old_seal = state.seal_health
    state.seal_health = seal_health_after_cycles(
        state.seal_health, state.pressure_cycles_today)
    state.total_pressure_cycles += state.pressure_cycles_today
    if (old_seal > SEAL_REPLACEMENT_THRESHOLD
            and state.seal_health <= SEAL_REPLACEMENT_THRESHOLD):
        events.append("SEAL CRITICAL -- replacement needed next sol")

    state.operational = True
    state.events = events

    return TickResult(
        sol=state.sol, air_leaked_kg=air_lost,
        thermal_loss_kwh=t_loss, burst_margin_ratio=margin,
        fabric_health=state.fabric_health,
        seal_health=state.seal_health,
        puncture_count=state.puncture_count,
        operational=True, events=list(events),
    )


# ── Convenience runner ──────────────────────────────────────────────────

def run_simulation(sols: int = 365,
                   length_m: float = DEFAULT_LENGTH_M,
                   radius_m: float = DEFAULT_INNER_RADIUS_M,
                   shielded: bool = False,
                   ) -> list[TickResult]:
    """Run tunnel simulation for *sols* Mars sols."""
    state = PressurizedTunnel(
        length_m=length_m,
        inner_radius_m=radius_m,
        regolith_shielded=shielded,
    )
    return [tick(state) for _ in range(sols)]


if __name__ == "__main__":
    import sys
    sols = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    results = run_simulation(sols=sols, shielded=True)
    total_air = sum(r.air_leaked_kg for r in results)
    total_heat = sum(r.thermal_loss_kwh for r in results)
    last = results[-1]
    print(f"Mars Pressurized Tunnel -- {sols} sols")
    print(f"  Air lost: {total_air:.1f} kg | Thermal loss: {total_heat:.0f} kWh")
    print(f"  Fabric: {last.fabric_health:.2%} | Seal: {last.seal_health:.2%}")
    print(f"  Burst margin: {last.burst_margin_ratio:.1f}x | Punctures: {last.puncture_count}")
