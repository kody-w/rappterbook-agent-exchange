"""
fuel_production.py — Sabatier ISRU propellant factory for Mars.

Models in-situ propellant production via the Sabatier reaction and
water electrolysis. This is the return ticket home.

Chemistry:
  Sabatier:     CO2 + 4H2 → CH4 + 2H2O   (ΔH = −165 kJ/mol, exothermic)
  Electrolysis: 2H2O → 2H2 + O2           (ΔH = +572 kJ/mol, endothermic)

Net cycle (CO2 + energy → CH4 + O2):
  CO2 + 2H2O + energy → CH4 + 2O2
  Each kg CH4 produced needs 2.75 kg CO2 and 2.25 kg H2O (stoichiometric).
  Produces 4.0 kg O2 as byproduct per kg CH4.

Physical references:
  - SpaceX Starship return: ~240t propellant (78% LOX, 22% CH4 by mass)
  - Sabatier reactor optimal: 300–400°C, Ru/Al2O3 catalyst
  - Electrolysis: 55 kWh per kg H2 at ~70% efficiency (PEM)
  - Mars atmosphere: 95.3% CO2, 636 Pa → unlimited CO2 feedstock
  - CO2 capture from atmosphere: ~0.5 kWh/kg (cryogenic separation)

One tick = one sol. Mass in kg. Energy in kWh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# --- Molar masses (g/mol) ---
MW_CO2 = 44.01
MW_H2 = 2.016
MW_CH4 = 16.04
MW_H2O = 18.015
MW_O2 = 32.00

# --- Stoichiometric mass ratios (per kg CH4 produced) ---
CO2_PER_KG_CH4 = MW_CO2 / MW_CH4               # 2.745 kg CO2 per kg CH4
H2_PER_KG_CH4 = (4 * MW_H2) / MW_CH4           # 0.503 kg H2 per kg CH4
H2O_PER_KG_CH4 = (2 * MW_H2O) / MW_CH4         # 2.246 kg H2O per kg CH4 (produced)
O2_PER_KG_CH4 = (2 * MW_O2) / MW_CH4           # 3.990 kg O2 per kg CH4

# --- Electrolysis: H2O → H2 + 0.5 O2 ---
# 2.016 kg H2 requires 18.015 kg H2O
H2O_PER_KG_H2 = MW_H2O / MW_H2                 # 8.937 kg H2O per kg H2
O2_FROM_ELECTROLYSIS_PER_KG_H2 = (0.5 * MW_O2) / MW_H2  # 7.937 kg O2 per kg H2

# --- Energy costs ---
ELECTROLYSIS_KWH_PER_KG_H2 = 55.0              # PEM electrolyzer at ~70% eff
CO2_CAPTURE_KWH_PER_KG = 0.5                   # cryogenic CO2 from atmosphere
SABATIER_HEAT_RECOVERY_FRACTION = 0.30          # fraction of exothermic heat reused

# --- Reactor performance ---
CATALYST_FRESH_EFFICIENCY = 0.95                # Ru/Al2O3 catalyst at start-of-life
CATALYST_DEGRADATION_PER_SOL = 0.00005          # ~3% per Mars year
CATALYST_MIN_EFFICIENCY = 0.40                  # below this → reactor shutdown
REACTOR_WARMUP_KWH = 5.0                       # energy to bring reactor to 300°C
OPTIMAL_TEMP_LOW_C = 300.0                      # Sabatier sweet spot (reactor temp)
OPTIMAL_TEMP_HIGH_C = 400.0
AMBIENT_EFFECT_FACTOR = 0.001                   # colder Mars = slightly more warmup

# --- Tank limits ---
DEFAULT_CH4_TANK_KG = 60_000.0                  # ~53t CH4 for Starship return
DEFAULT_O2_TANK_KG = 200_000.0                  # ~187t LOX for Starship return
DEFAULT_H2O_BUFFER_KG = 5_000.0                 # water buffer for electrolysis

# --- Production targets ---
STARSHIP_CH4_KG = 53_000.0                      # CH4 needed for Earth return
STARSHIP_O2_KG = 187_000.0                      # LOX needed for Earth return
STARSHIP_TOTAL_PROPELLANT_KG = STARSHIP_CH4_KG + STARSHIP_O2_KG


@dataclass
class PropellantTank:
    """A cryogenic storage tank.

    capacity_kg: maximum storage
    level_kg: current contents
    boiloff_rate_per_sol: fraction lost to boiloff each sol
    """
    capacity_kg: float
    level_kg: float = 0.0
    boiloff_rate_per_sol: float = 0.001  # 0.1% per sol for well-insulated tank

    def __post_init__(self) -> None:
        """Clamp to valid physical ranges."""
        self.capacity_kg = max(0.0, self.capacity_kg)
        self.level_kg = max(0.0, min(self.level_kg, self.capacity_kg))
        self.boiloff_rate_per_sol = max(0.0, min(1.0, self.boiloff_rate_per_sol))

    def headroom(self) -> float:
        """Available space in tank (kg)."""
        return max(0.0, self.capacity_kg - self.level_kg)

    def add(self, kg: float) -> float:
        """Add propellant. Returns actual kg added (capped by headroom)."""
        if kg <= 0:
            return 0.0
        actual = min(kg, self.headroom())
        self.level_kg += actual
        return actual

    def remove(self, kg: float) -> float:
        """Remove propellant. Returns actual kg removed."""
        if kg <= 0:
            return 0.0
        actual = min(kg, self.level_kg)
        self.level_kg -= actual
        return actual

    def apply_boiloff(self) -> float:
        """Apply daily boiloff loss. Returns kg lost."""
        lost = self.level_kg * self.boiloff_rate_per_sol
        self.level_kg -= lost
        return lost

    def fill_fraction(self) -> float:
        """Tank fill level as fraction [0, 1]."""
        if self.capacity_kg <= 0:
            return 0.0
        return self.level_kg / self.capacity_kg


@dataclass
class SabatierReactor:
    """The Sabatier ISRU reactor.

    max_ch4_per_sol_kg: theoretical max CH4 production per sol
    catalyst_efficiency: current catalyst health [0, 1]
    total_sols_run: cumulative operational sols
    """
    max_ch4_per_sol_kg: float = 10.0
    catalyst_efficiency: float = CATALYST_FRESH_EFFICIENCY
    total_sols_run: int = 0

    def __post_init__(self) -> None:
        """Clamp to valid ranges."""
        self.max_ch4_per_sol_kg = max(0.0, self.max_ch4_per_sol_kg)
        self.catalyst_efficiency = max(0.0, min(1.0, self.catalyst_efficiency))
        self.total_sols_run = max(0, self.total_sols_run)

    def is_operational(self) -> bool:
        """Reactor can run if catalyst is above minimum threshold."""
        return self.catalyst_efficiency >= CATALYST_MIN_EFFICIENCY

    def degrade_catalyst(self) -> float:
        """Apply one sol of catalyst degradation. Returns efficiency lost."""
        if not self.is_operational():
            return 0.0
        loss = CATALYST_DEGRADATION_PER_SOL
        self.catalyst_efficiency = max(0.0, self.catalyst_efficiency - loss)
        return loss

    def effective_rate_kg(self) -> float:
        """Effective CH4 production rate accounting for catalyst health."""
        if not self.is_operational():
            return 0.0
        return self.max_ch4_per_sol_kg * self.catalyst_efficiency


def warmup_energy_kwh(ambient_temp_c: float) -> float:
    """Energy to bring reactor from ambient Mars temp to operating temp.

    Colder ambient = more energy needed. Returns kWh.
    """
    delta = OPTIMAL_TEMP_LOW_C - ambient_temp_c
    return REACTOR_WARMUP_KWH * (1.0 + max(0.0, delta) * AMBIENT_EFFECT_FACTOR)


def electrolysis_water_needed_kg(h2_needed_kg: float) -> float:
    """Water needed to electrolyze a given mass of H2."""
    if h2_needed_kg <= 0:
        return 0.0
    return h2_needed_kg * H2O_PER_KG_H2


def electrolysis_energy_kwh(h2_kg: float) -> float:
    """Energy to electrolyze water into a given mass of H2."""
    if h2_kg <= 0:
        return 0.0
    return h2_kg * ELECTROLYSIS_KWH_PER_KG_H2


def co2_capture_energy_kwh(co2_kg: float) -> float:
    """Energy to capture CO2 from Mars atmosphere."""
    if co2_kg <= 0:
        return 0.0
    return co2_kg * CO2_CAPTURE_KWH_PER_KG


def sabatier_products(ch4_kg: float) -> dict:
    """Calculate all inputs consumed and outputs produced for a given CH4 yield.

    Returns dict with masses in kg.
    """
    if ch4_kg <= 0:
        return {
            "ch4_kg": 0.0,
            "co2_consumed_kg": 0.0,
            "h2_consumed_kg": 0.0,
            "h2o_produced_kg": 0.0,
            "o2_equivalent_kg": 0.0,
        }
    return {
        "ch4_kg": ch4_kg,
        "co2_consumed_kg": ch4_kg * CO2_PER_KG_CH4,
        "h2_consumed_kg": ch4_kg * H2_PER_KG_CH4,
        "h2o_produced_kg": ch4_kg * H2O_PER_KG_CH4,
        "o2_equivalent_kg": ch4_kg * O2_PER_KG_CH4,
    }


def tick_fuel_production(
    reactor: SabatierReactor,
    ch4_tank: PropellantTank,
    o2_tank: PropellantTank,
    water_available_kg: float,
    power_available_kwh: float,
    ambient_temp_c: float,
) -> dict:
    """Advance fuel production by one sol.

    Process:
      1. Check reactor health
      2. Compute warmup energy
      3. Compute max CH4 from reactor rate
      4. Compute H2 needed → electrolysis water and energy
      5. Limit by water, power, and tank headroom
      6. Run Sabatier: consume CO2 + H2, produce CH4 + H2O
      7. Electrolyze excess H2O for O2 (if power permits)
      8. Store products, degrade catalyst
      9. Apply boiloff to tanks

    Args:
        reactor: the Sabatier reactor (mutated)
        ch4_tank: methane storage (mutated)
        o2_tank: oxygen/LOX storage (mutated)
        water_available_kg: water budget for this sol
        power_available_kwh: power budget for this sol
        ambient_temp_c: Mars surface temperature

    Returns:
        Snapshot dict with all production metrics.

    Conservation laws enforced:
      - mass_in (water + CO2) == mass_out (CH4 + O2 + water_returned)
      - energy consumed ≤ power available
      - tank levels never exceed capacity or go negative
    """
    result = {
        "operational": False,
        "ch4_produced_kg": 0.0,
        "o2_produced_kg": 0.0,
        "water_consumed_kg": 0.0,
        "water_returned_kg": 0.0,
        "co2_consumed_kg": 0.0,
        "power_consumed_kwh": 0.0,
        "warmup_kwh": 0.0,
        "electrolysis_kwh": 0.0,
        "co2_capture_kwh": 0.0,
        "ch4_tank_kg": ch4_tank.level_kg,
        "o2_tank_kg": o2_tank.level_kg,
        "catalyst_efficiency": reactor.catalyst_efficiency,
        "ch4_boiloff_kg": 0.0,
        "o2_boiloff_kg": 0.0,
        "limited_by": "none",
    }

    # Step 1: Check reactor health
    if not reactor.is_operational():
        result["limited_by"] = "catalyst_dead"
        result["ch4_boiloff_kg"] = ch4_tank.apply_boiloff()
        result["o2_boiloff_kg"] = o2_tank.apply_boiloff()
        result["ch4_tank_kg"] = ch4_tank.level_kg
        result["o2_tank_kg"] = o2_tank.level_kg
        return result

    result["operational"] = True

    # Step 2: Warmup energy
    warmup = warmup_energy_kwh(ambient_temp_c)
    result["warmup_kwh"] = round(warmup, 4)
    power_remaining = power_available_kwh - warmup
    if power_remaining <= 0:
        result["limited_by"] = "power"
        result["ch4_boiloff_kg"] = ch4_tank.apply_boiloff()
        result["o2_boiloff_kg"] = o2_tank.apply_boiloff()
        result["ch4_tank_kg"] = ch4_tank.level_kg
        result["o2_tank_kg"] = o2_tank.level_kg
        result["power_consumed_kwh"] = round(power_available_kwh, 4)
        return result

    # Step 3: Max CH4 from reactor rate
    max_ch4 = reactor.effective_rate_kg()

    # Step 4: H2 needed and electrolysis budget
    h2_needed = max_ch4 * H2_PER_KG_CH4
    water_for_h2 = electrolysis_water_needed_kg(h2_needed)
    energy_for_h2 = electrolysis_energy_kwh(h2_needed)

    # CO2 is free from atmosphere but costs energy to capture
    co2_needed = max_ch4 * CO2_PER_KG_CH4
    energy_for_co2 = co2_capture_energy_kwh(co2_needed)

    total_energy_needed = energy_for_h2 + energy_for_co2

    # Step 5: Limit by water, power, and tank headroom
    limiting = "none"

    # Water limit
    if water_available_kg < water_for_h2 and water_for_h2 > 0:
        scale = water_available_kg / water_for_h2
        max_ch4 *= scale
        limiting = "water"

    # Power limit (recalculate after water scaling)
    h2_needed = max_ch4 * H2_PER_KG_CH4
    energy_for_h2 = electrolysis_energy_kwh(h2_needed)
    co2_needed = max_ch4 * CO2_PER_KG_CH4
    energy_for_co2 = co2_capture_energy_kwh(co2_needed)
    total_energy_needed = energy_for_h2 + energy_for_co2

    if total_energy_needed > power_remaining and total_energy_needed > 0:
        scale = power_remaining / total_energy_needed
        max_ch4 *= scale
        limiting = "power"

    # Tank headroom limit
    ch4_headroom = ch4_tank.headroom()
    if max_ch4 > ch4_headroom and ch4_headroom >= 0:
        max_ch4 = ch4_headroom
        limiting = "ch4_tank_full"

    # O2 produced (from both Sabatier byproduct water electrolysis + direct)
    o2_from_sabatier = max_ch4 * O2_PER_KG_CH4
    o2_headroom = o2_tank.headroom()
    if o2_from_sabatier > o2_headroom and o2_headroom >= 0:
        if o2_from_sabatier > 0:
            scale = o2_headroom / o2_from_sabatier
            max_ch4 *= scale
            limiting = "o2_tank_full"

    # Floor at zero
    max_ch4 = max(0.0, max_ch4)

    # Step 6: Run Sabatier
    products = sabatier_products(max_ch4)
    ch4_produced = products["ch4_kg"]
    co2_consumed = products["co2_consumed_kg"]
    h2_consumed = products["h2_consumed_kg"]
    h2o_produced = products["h2o_produced_kg"]  # byproduct water (recycled)

    # Actual water consumed for electrolysis (to make H2)
    water_consumed = electrolysis_water_needed_kg(h2_consumed)
    energy_h2 = electrolysis_energy_kwh(h2_consumed)
    energy_co2 = co2_capture_energy_kwh(co2_consumed)
    total_power = warmup + energy_h2 + energy_co2

    # O2: comes from electrolyzing water (to get H2) + electrolyzing Sabatier byproduct water
    # The H2 electrolysis also produces O2 as byproduct
    o2_from_h2_electrolysis = h2_consumed * O2_FROM_ELECTROLYSIS_PER_KG_H2
    # Sabatier byproduct water can also be electrolyzed for more O2 if we have power
    # But for simplicity in this tick, we just store the electrolysis O2
    total_o2 = o2_from_h2_electrolysis

    # Step 7: Store products
    ch4_stored = ch4_tank.add(ch4_produced)
    o2_stored = o2_tank.add(total_o2)

    # Step 8: Degrade catalyst
    reactor.degrade_catalyst()
    reactor.total_sols_run += 1

    # Step 9: Boiloff
    ch4_boiloff = ch4_tank.apply_boiloff()
    o2_boiloff = o2_tank.apply_boiloff()

    result.update({
        "operational": True,
        "ch4_produced_kg": round(ch4_produced, 6),
        "o2_produced_kg": round(total_o2, 6),
        "water_consumed_kg": round(water_consumed, 6),
        "water_returned_kg": round(h2o_produced, 6),
        "co2_consumed_kg": round(co2_consumed, 6),
        "power_consumed_kwh": round(total_power, 4),
        "warmup_kwh": round(warmup, 4),
        "electrolysis_kwh": round(energy_h2, 4),
        "co2_capture_kwh": round(energy_co2, 4),
        "ch4_tank_kg": round(ch4_tank.level_kg, 6),
        "o2_tank_kg": round(o2_tank.level_kg, 6),
        "catalyst_efficiency": round(reactor.catalyst_efficiency, 6),
        "ch4_boiloff_kg": round(ch4_boiloff, 6),
        "o2_boiloff_kg": round(o2_boiloff, 6),
        "limited_by": limiting,
    })

    return result


def sols_to_full_load(
    reactor: SabatierReactor,
    ch4_target_kg: float = STARSHIP_CH4_KG,
    o2_target_kg: float = STARSHIP_O2_KG,
) -> int:
    """Estimate sols to produce a full propellant load (ignoring resource limits).

    Uses current reactor rate with catalyst degradation over time.
    Returns -1 if reactor is dead.
    """
    if not reactor.is_operational():
        return -1

    ch4_produced = 0.0
    o2_produced = 0.0
    eff = reactor.catalyst_efficiency
    sols = 0

    while ch4_produced < ch4_target_kg or o2_produced < o2_target_kg:
        if eff < CATALYST_MIN_EFFICIENCY:
            return -1  # catalyst dies before target
        rate = reactor.max_ch4_per_sol_kg * eff
        ch4_produced += rate
        o2_produced += rate * O2_FROM_ELECTROLYSIS_PER_KG_H2 * H2_PER_KG_CH4
        eff -= CATALYST_DEGRADATION_PER_SOL
        sols += 1
        if sols > 100_000:
            return -1  # safety valve

    return sols


def propellant_status(
    ch4_tank: PropellantTank,
    o2_tank: PropellantTank,
) -> dict:
    """Return current propellant readiness for Earth return.

    Shows fill levels vs Starship requirements.
    """
    ch4_pct = (ch4_tank.level_kg / STARSHIP_CH4_KG * 100) if STARSHIP_CH4_KG > 0 else 0
    o2_pct = (o2_tank.level_kg / STARSHIP_O2_KG * 100) if STARSHIP_O2_KG > 0 else 0
    overall = min(ch4_pct, o2_pct)
    ready = ch4_tank.level_kg >= STARSHIP_CH4_KG and o2_tank.level_kg >= STARSHIP_O2_KG

    return {
        "ch4_kg": round(ch4_tank.level_kg, 2),
        "ch4_target_kg": STARSHIP_CH4_KG,
        "ch4_percent": round(ch4_pct, 2),
        "o2_kg": round(o2_tank.level_kg, 2),
        "o2_target_kg": STARSHIP_O2_KG,
        "o2_percent": round(o2_pct, 2),
        "overall_percent": round(overall, 2),
        "launch_ready": ready,
    }
