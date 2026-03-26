"""propellant_depot.py — Mars Cryogenic Propellant Storage & Boil-off Management.

The return ticket sits in a tank.  fuel_production.py makes the
propellant (CH₄ + LOX via Sabatier + electrolysis); this module
keeps it cold enough to stay liquid for months until the launch window.

Mars ambient temperature averages −60 °C (213 K) — warm enough to boil
LOX (bp −183 °C / 90 K) and LCH₄ (bp −161 °C / 112 K) in minutes
without insulation.  The depot must hold cryogenics against continuous
heat leak from the environment.

Physics modelled
----------------
* **Heat leak through MLI** — Multi-Layer Insulation (60–80 layers of
  aluminized Mylar + Dacron spacers).  Effective thermal conductivity
  ~0.05 mW/m·K in vacuum.  On Mars (636 Pa CO₂ atmosphere), gas
  conduction adds ~1 mW/m·K.  Net heat flux: 0.5–2.0 W/m² depending
  on insulation quality and ambient temperature.
* **Boil-off** — Heat leak / latent heat = mass loss rate.
    LOX: L_vap = 213 kJ/kg → 1 W sustained = 0.417 kg/sol boiled off.
    LCH₄: L_vap = 510 kJ/kg → 1 W sustained = 0.174 kg/sol boiled off.
  SpaceX target: <0.1%/day for Starship tanker on Mars surface.
* **Cryocooler (zero-boil-off)** — Pulse-tube or Stirling cryocooler
  removes heat from tank, rejects to radiator.  Power required:
    P_cryo = Q_leak / COP,  where COP ≈ η_Carnot × 0.3
    η_Carnot = T_cold / (T_hot − T_cold)
  For LOX at 90 K, ambient 213 K: COP_ideal = 0.73, COP_real ≈ 0.22.
  → ~4.5 W electrical per W of heat removed.
* **Tank capacity** — Cylindrical Al-Li tanks.  LOX density 1141 kg/m³,
  LCH₄ density 423 kg/m³.  SpaceX Starship: ~1200 m³ total.
* **Launch readiness** — Propellant load must reach target mass before
  the Earth-return launch window (every 26 months).

References:
  - SpaceX Starship: ~240 t propellant (187 t LOX + 53 t CH₄)
  - NASA Zero Boil-Off (ZBO) cryocooler: Creare Inc, TRL 5 (2023)
  - LOX boiling point: 90.19 K (−182.96 °C) at 101.3 kPa
  - LCH₄ boiling point: 111.66 K (−161.49 °C) at 101.3 kPa
  - MLI in Mars atmosphere: Plachta & Kittel, NASA/TM-2003-211919
  - Mars ambient: 150–293 K, avg ~213 K, pressure 636 Pa

One tick = one sol.  Mass in kg, energy in kWh, temperature in K.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

LOX_BP_K = 90.19                   # boiling point (K) at 101.3 kPa
LCH4_BP_K = 111.66

LOX_LATENT_KJ_KG = 213.0          # latent heat of vaporization (kJ/kg)
LCH4_LATENT_KJ_KG = 510.0

LOX_DENSITY = 1141.0               # liquid density (kg/m³)
LCH4_DENSITY = 423.0

MARS_AVG_TEMP_K = 213.0
MARS_TEMP_MIN_K = 150.0
MARS_TEMP_MAX_K = 293.0

MLI_HEAT_FLUX_NOMINAL = 1.0        # W/m² through standard 60-layer MLI

CARNOT_EFFICIENCY_FRACTION = 0.30   # real COP / ideal COP
CRYOCOOLER_MIN_POWER_KW = 0.5

TARGET_LOX_KG = 187_000.0          # SpaceX Starship-class
TARGET_LCH4_KG = 53_000.0

SOL_SECONDS = 88_775.0             # 24h 39m 35s
SOL_HOURS = SOL_SECONDS / 3600.0


# ---------------------------------------------------------------------------
# Pure physics functions
# ---------------------------------------------------------------------------

def heat_leak_watts(surface_area_m2: float, flux_w_m2: float) -> float:
    """Total heat leak into a tank from insulation performance."""
    return max(0.0, surface_area_m2 * flux_w_m2)


def boiloff_rate_kg_per_sol(heat_leak_w: float, latent_kj_kg: float) -> float:
    """Mass of cryogen boiled off per sol from sustained heat leak."""
    if latent_kj_kg <= 0.0 or heat_leak_w <= 0.0:
        return 0.0
    return heat_leak_w * SOL_SECONDS / (latent_kj_kg * 1000.0)


def cryocooler_power_kw(
    heat_load_w: float,
    cold_temp_k: float,
    hot_temp_k: float,
) -> float:
    """Electrical power for cryocooler to remove heat at cold_temp.

    COP_real = (T_cold / (T_hot - T_cold)) × η_fraction
    P = Q / COP_real
    """
    if cold_temp_k <= 0.0 or cold_temp_k >= hot_temp_k:
        return 0.0
    cop_ideal = cold_temp_k / (hot_temp_k - cold_temp_k)
    cop_real = cop_ideal * CARNOT_EFFICIENCY_FRACTION
    if cop_real <= 0.0:
        return float("inf")
    power_w = heat_load_w / cop_real
    return max(CRYOCOOLER_MIN_POWER_KW, power_w / 1000.0)


def tank_surface_area_m2(volume_m3: float) -> float:
    """Surface area of a cylindrical tank (L/D = 2).

    V = π/2 × D³,  A = 5π/2 × D².
    """
    if volume_m3 <= 0.0:
        return 0.0
    d = (2.0 * volume_m3 / math.pi) ** (1.0 / 3.0)
    return 2.5 * math.pi * d * d


def fill_fraction(current_kg: float, capacity_kg: float) -> float:
    """Fraction of tank filled [0, 1]."""
    if capacity_kg <= 0.0:
        return 0.0
    return max(0.0, min(1.0, current_kg / capacity_kg))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CryoTank:
    """A single cryogenic storage tank."""

    label: str
    capacity_kg: float
    current_kg: float = 0.0
    boiling_point_k: float = 90.0
    latent_kj_kg: float = 213.0
    density_kg_m3: float = 1141.0
    insulation_flux_w_m2: float = MLI_HEAT_FLUX_NOMINAL
    insulation_health: float = 1.0
    total_boiloff_kg: float = 0.0

    @property
    def volume_m3(self) -> float:
        return self.capacity_kg / max(self.density_kg_m3, 1.0)

    @property
    def surface_area_m2(self) -> float:
        return tank_surface_area_m2(self.volume_m3)

    @property
    def fill_pct(self) -> float:
        return fill_fraction(self.current_kg, self.capacity_kg) * 100.0

    def effective_heat_flux(self) -> float:
        """Heat flux accounting for insulation degradation."""
        return self.insulation_flux_w_m2 / max(self.insulation_health, 0.1)

    def compute_heat_leak(self, ambient_temp_k: float = MARS_AVG_TEMP_K) -> float:
        """Heat leak in watts, scaled by temperature difference."""
        delta_t = max(0.0, ambient_temp_k - self.boiling_point_k)
        reference_delta = MARS_AVG_TEMP_K - self.boiling_point_k
        if reference_delta <= 0.0:
            return 0.0
        temp_scale = delta_t / reference_delta
        flux = self.effective_heat_flux() * temp_scale
        return heat_leak_watts(self.surface_area_m2, flux)

    def tick(
        self,
        added_kg: float = 0.0,
        cryocooler_active: bool = False,
        ambient_temp_k: float = MARS_AVG_TEMP_K,
        available_power_kw: float = float("inf"),
    ) -> dict:
        """Advance one sol of cryogenic storage."""
        self.current_kg = min(self.capacity_kg, self.current_kg + max(0.0, added_kg))

        q_leak_w = self.compute_heat_leak(ambient_temp_k)

        cryo_power_kw = 0.0
        q_removed_w = 0.0
        if cryocooler_active and q_leak_w > 0:
            power_needed = cryocooler_power_kw(
                q_leak_w, self.boiling_point_k, ambient_temp_k
            )
            if power_needed <= available_power_kw:
                q_removed_w = q_leak_w
                cryo_power_kw = power_needed
            else:
                fraction = available_power_kw / max(power_needed, 1e-12)
                q_removed_w = q_leak_w * fraction
                cryo_power_kw = available_power_kw

        q_net_w = max(0.0, q_leak_w - q_removed_w)
        boiloff_kg = boiloff_rate_kg_per_sol(q_net_w, self.latent_kj_kg)
        boiloff_kg = min(boiloff_kg, self.current_kg)

        self.current_kg -= boiloff_kg
        self.total_boiloff_kg += boiloff_kg
        self.insulation_health = max(0.3, self.insulation_health - 0.0001)

        return {
            "label": self.label,
            "current_kg": round(self.current_kg, 2),
            "fill_pct": round(self.fill_pct, 2),
            "heat_leak_w": round(q_leak_w, 3),
            "boiloff_kg": round(boiloff_kg, 4),
            "cryo_power_kw": round(cryo_power_kw, 4),
            "zbo_active": q_removed_w >= q_leak_w * 0.99,
            "insulation_health": round(self.insulation_health, 4),
        }


@dataclass
class PropellantDepot:
    """Mars cryogenic propellant depot — LOX + LCH₄ tank farm."""

    lox_tank: CryoTank = field(default_factory=lambda: CryoTank(
        label="LOX", capacity_kg=TARGET_LOX_KG,
        boiling_point_k=LOX_BP_K, latent_kj_kg=LOX_LATENT_KJ_KG,
        density_kg_m3=LOX_DENSITY,
    ))
    lch4_tank: CryoTank = field(default_factory=lambda: CryoTank(
        label="LCH4", capacity_kg=TARGET_LCH4_KG,
        boiling_point_k=LCH4_BP_K, latent_kj_kg=LCH4_LATENT_KJ_KG,
        density_kg_m3=LCH4_DENSITY,
    ))
    sol: int = 0
    total_energy_kwh: float = 0.0
    history: list = field(default_factory=list)

    def tick(
        self,
        lox_added_kg: float = 0.0,
        lch4_added_kg: float = 0.0,
        cryocooler_active: bool = True,
        ambient_temp_k: float = MARS_AVG_TEMP_K,
        available_power_kw: float = 20.0,
    ) -> dict:
        """Advance depot by one sol."""
        self.sol += 1
        power_per_tank = available_power_kw / 2.0

        lox_result = self.lox_tank.tick(
            added_kg=lox_added_kg, cryocooler_active=cryocooler_active,
            ambient_temp_k=ambient_temp_k, available_power_kw=power_per_tank,
        )
        lch4_result = self.lch4_tank.tick(
            added_kg=lch4_added_kg, cryocooler_active=cryocooler_active,
            ambient_temp_k=ambient_temp_k, available_power_kw=power_per_tank,
        )

        sol_energy_kwh = (
            (lox_result["cryo_power_kw"] + lch4_result["cryo_power_kw"]) * SOL_HOURS
        )
        self.total_energy_kwh += sol_energy_kwh

        lox_ready = self.lox_tank.current_kg >= TARGET_LOX_KG * 0.95
        lch4_ready = self.lch4_tank.current_kg >= TARGET_LCH4_KG * 0.95
        total_propellant = self.lox_tank.current_kg + self.lch4_tank.current_kg
        target_total = TARGET_LOX_KG + TARGET_LCH4_KG

        result = {
            "sol": self.sol,
            "lox": lox_result,
            "lch4": lch4_result,
            "total_propellant_kg": round(total_propellant, 2),
            "fill_pct": round(total_propellant / target_total * 100, 2),
            "energy_kwh": round(sol_energy_kwh, 2),
            "launch_ready": lox_ready and lch4_ready,
            "total_boiloff_kg": round(
                self.lox_tank.total_boiloff_kg + self.lch4_tank.total_boiloff_kg, 2
            ),
        }
        self.history.append(result)
        return result

    def days_to_ready(self, lox_rate_kg_sol: float, lch4_rate_kg_sol: float) -> float:
        """Estimate sols until launch-ready, accounting for boil-off."""
        lox_needed = max(0.0, TARGET_LOX_KG * 0.95 - self.lox_tank.current_kg)
        lch4_needed = max(0.0, TARGET_LCH4_KG * 0.95 - self.lch4_tank.current_kg)

        lox_boiloff = boiloff_rate_kg_per_sol(
            self.lox_tank.compute_heat_leak(), LOX_LATENT_KJ_KG
        )
        lch4_boiloff = boiloff_rate_kg_per_sol(
            self.lch4_tank.compute_heat_leak(), LCH4_LATENT_KJ_KG
        )

        lox_net = lox_rate_kg_sol - lox_boiloff
        lch4_net = lch4_rate_kg_sol - lch4_boiloff

        if lox_net <= 0 and lox_needed > 0:
            return float("inf")
        if lch4_net <= 0 and lch4_needed > 0:
            return float("inf")

        lox_sols = lox_needed / max(lox_net, 1e-12) if lox_needed > 0 else 0.0
        lch4_sols = lch4_needed / max(lch4_net, 1e-12) if lch4_needed > 0 else 0.0
        return max(lox_sols, lch4_sols)

    def get_status(self) -> dict:
        """Current depot status summary."""
        total = self.lox_tank.current_kg + self.lch4_tank.current_kg
        target = TARGET_LOX_KG + TARGET_LCH4_KG
        return {
            "sol": self.sol,
            "lox_kg": round(self.lox_tank.current_kg, 2),
            "lch4_kg": round(self.lch4_tank.current_kg, 2),
            "total_kg": round(total, 2),
            "fill_pct": round(total / target * 100, 2),
            "total_boiloff_kg": round(
                self.lox_tank.total_boiloff_kg + self.lch4_tank.total_boiloff_kg, 2
            ),
            "total_energy_kwh": round(self.total_energy_kwh, 2),
            "launch_ready": (
                self.lox_tank.current_kg >= TARGET_LOX_KG * 0.95
                and self.lch4_tank.current_kg >= TARGET_LCH4_KG * 0.95
            ),
        }


# ---------------------------------------------------------------------------
# Factory / runner
# ---------------------------------------------------------------------------

def make_depot(
    lox_capacity_kg: float = TARGET_LOX_KG,
    lch4_capacity_kg: float = TARGET_LCH4_KG,
) -> PropellantDepot:
    """Create a fresh propellant depot."""
    return PropellantDepot(
        lox_tank=CryoTank(
            label="LOX", capacity_kg=lox_capacity_kg,
            boiling_point_k=LOX_BP_K, latent_kj_kg=LOX_LATENT_KJ_KG,
            density_kg_m3=LOX_DENSITY,
        ),
        lch4_tank=CryoTank(
            label="LCH4", capacity_kg=lch4_capacity_kg,
            boiling_point_k=LCH4_BP_K, latent_kj_kg=LCH4_LATENT_KJ_KG,
            density_kg_m3=LCH4_DENSITY,
        ),
    )


def run_depot(
    sols: int = 365,
    lox_rate_kg_sol: float = 600.0,
    lch4_rate_kg_sol: float = 170.0,
    cryocooler_active: bool = True,
    available_power_kw: float = 20.0,
) -> list:
    """Simulate propellant depot for N sols.

    Default production rates: ~600 kg/sol LOX + ~170 kg/sol CH₄
    matches Starship-class fill in ~350 sols (just under 1 Mars year).
    """
    depot = make_depot()
    results = []
    for _ in range(sols):
        result = depot.tick(
            lox_added_kg=lox_rate_kg_sol, lch4_added_kg=lch4_rate_kg_sol,
            cryocooler_active=cryocooler_active,
            available_power_kw=available_power_kw,
        )
        results.append(result)
    return results
