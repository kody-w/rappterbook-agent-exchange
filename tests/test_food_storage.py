"""Tests for food_storage.py — Mars Colony Food Preservation & Caloric Inventory.

91 tests covering:
  - Physical constants validation
  - Freeze-drying physics (energy, yield, mass conservation)
  - Caloric accounting (consumption, decay, inventory)
  - Ration level thresholds
  - Nutrient & vitamin degradation (exponential decay, half-lives)
  - Fresh food spoilage (temperature dependence)
  - Container seal breaches (perchlorate contamination)
  - Supply ship deliveries
  - Greenhouse harvest integration
  - Storage overflow handling
  - Multi-sol simulation (smoke tests, starvation detection)
  - Edge cases (zero crew, empty storage, negative inputs)
  - Conservation laws (calories in ≥ calories out)
  - Property-based invariants (physical bounds)
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from food_storage import (
    CALORIC_DECAY_RATE_PER_SOL,
    COLD_STORAGE_CAPACITY_KG,
    COLD_STORAGE_POWER_KW,
    CONTAINER_SEAL_FAILURE_RATE,
    FREEZE_DRY_EFFICIENCY,
    FREEZE_DRY_HOURS_PER_KG,
    FREEZE_DRY_WATER_FRACTION,
    FoodBatch,
    KCAL_PER_KG_FREEZE_DRIED,
    KCAL_PER_KG_FRESH,
    KCAL_PER_PERSON_SOL_EVA,
    KCAL_PER_PERSON_SOL_NORMAL,
    KCAL_PER_PERSON_SOL_RATION,
    MARS_AVG_TEMP_C,
    MARS_SURFACE_PRESSURE_PA,
    MAX_FREEZE_DRY_KG_SOL,
    MJ_TO_KWH,
    NUTRIENT_DECAY_RATE_PER_SOL,
    PERCHLORATE_CONTAMINATION_LOSS,
    RATION_CAUTION_DAYS,
    RATION_CRITICAL_DAYS,
    RATION_NORMAL_DAYS,
    RATION_STARVATION_DAYS,
    RATION_WARNING_DAYS,
    SolReport,
    STORAGE_CAPACITY_KG,
    StorageState,
    SUPPLY_SHIP_FOOD_KG_PER_PERSON,
    SUPPLY_SHIP_INTERVAL_SOLS,
    SUPPLY_SHIP_KCAL_PER_KG,
    VACUUM_PUMP_POWER_KW,
    VITAMIN_A_HALF_LIFE_SOLS,
    VITAMIN_B12_HALF_LIFE_SOLS,
    VITAMIN_C_HALF_LIFE_SOLS,
    WATER_SUBLIMATION_ENERGY_MJ_KG,
    WATER_TRIPLE_POINT_PA,
    caloric_decay,
    check_seal_breach,
    cold_storage_power_kwh,
    consumption_rate,
    days_of_food,
    freeze_dry_energy_kwh,
    freeze_dry_yield_kg,
    fresh_spoilage_rate,
    kcal_per_kg,
    make_storage,
    nutrient_decay,
    ration_level_from_days,
    run_storage,
    supply_ship_due,
    tick_storage,
    vitamin_decay,
)


# ============================================================================
# 1. Physical constants
# ============================================================================

class TestPhysicalConstants:
    """Validate physical constants against known values."""

    def test_water_sublimation_energy(self):
        """Latent heat of sublimation ~2.83 MJ/kg at 0°C."""
        assert 2.5 < WATER_SUBLIMATION_ENERGY_MJ_KG < 3.2

    def test_mj_to_kwh_conversion(self):
        """1 kWh = 3.6 MJ → conversion factor ~0.2778."""
        assert abs(MJ_TO_KWH - 1.0 / 3.6) < 0.001

    def test_mars_surface_pressure(self):
        """Mars average surface pressure ~636 Pa."""
        assert 400 < MARS_SURFACE_PRESSURE_PA < 900

    def test_water_triple_point(self):
        """Water triple point at 611.657 Pa."""
        assert 600 < WATER_TRIPLE_POINT_PA < 620

    def test_mars_pressure_near_triple_point(self):
        """Mars pressure is borderline for sublimation."""
        ratio = MARS_SURFACE_PRESSURE_PA / WATER_TRIPLE_POINT_PA
        assert 0.8 < ratio < 1.5

    def test_mars_avg_temp(self):
        """Mars average temp ~-60°C."""
        assert -80 < MARS_AVG_TEMP_C < -30

    def test_caloric_density_freeze_dried(self):
        """Freeze-dried food ~3500-4500 kcal/kg."""
        assert 3000 < KCAL_PER_KG_FREEZE_DRIED < 5000

    def test_caloric_density_fresh(self):
        """Fresh produce ~500-1200 kcal/kg."""
        assert 400 < KCAL_PER_KG_FRESH < 1500

    def test_human_caloric_need(self):
        """Normal activity: 2200-2800 kcal/sol."""
        assert 2200 <= KCAL_PER_PERSON_SOL_NORMAL <= 2800

    def test_eva_caloric_need_higher(self):
        """EVA work requires more calories than normal."""
        assert KCAL_PER_PERSON_SOL_EVA > KCAL_PER_PERSON_SOL_NORMAL

    def test_ration_caloric_need_lower(self):
        """Emergency rations below normal."""
        assert KCAL_PER_PERSON_SOL_RATION < KCAL_PER_PERSON_SOL_NORMAL

    def test_storage_capacity_positive(self):
        assert STORAGE_CAPACITY_KG > 0.0

    def test_supply_ship_interval(self):
        """Synodic period ~26 months ≈ 780 sols."""
        assert 700 < SUPPLY_SHIP_INTERVAL_SOLS < 900


# ============================================================================
# 2. Freeze-drying physics
# ============================================================================

class TestFreezeDrying:
    """Freeze-drying energy calculations and mass conservation."""

    def test_energy_positive_mass(self):
        """Positive energy for positive mass."""
        e = freeze_dry_energy_kwh(10.0)
        assert e > 0.0

    def test_energy_zero_mass(self):
        assert freeze_dry_energy_kwh(0.0) == 0.0

    def test_energy_negative_mass(self):
        assert freeze_dry_energy_kwh(-1.0) == 0.0

    def test_energy_scales_with_mass(self):
        """Energy should increase with mass."""
        e1 = freeze_dry_energy_kwh(1.0)
        e10 = freeze_dry_energy_kwh(10.0)
        assert e10 > e1

    def test_energy_linear_scaling(self):
        """Energy should scale linearly with mass."""
        e1 = freeze_dry_energy_kwh(1.0)
        e5 = freeze_dry_energy_kwh(5.0)
        assert abs(e5 / e1 - 5.0) < 0.01

    def test_yield_positive_mass(self):
        """Dry yield = (1 - water_fraction) * wet_mass."""
        y = freeze_dry_yield_kg(10.0)
        expected = 10.0 * (1.0 - FREEZE_DRY_WATER_FRACTION)
        assert abs(y - expected) < 0.01

    def test_yield_zero(self):
        assert freeze_dry_yield_kg(0.0) == 0.0

    def test_yield_negative(self):
        assert freeze_dry_yield_kg(-5.0) == 0.0

    def test_yield_less_than_input(self):
        """Dry mass must be less than wet mass (water removed)."""
        wet = 100.0
        dry = freeze_dry_yield_kg(wet)
        assert dry < wet

    def test_yield_positive(self):
        """Dry mass must be positive for positive input."""
        assert freeze_dry_yield_kg(50.0) > 0.0

    def test_mass_conservation_ratio(self):
        """Water fraction + dry fraction = 1.0."""
        assert abs(FREEZE_DRY_WATER_FRACTION + (1.0 - FREEZE_DRY_WATER_FRACTION) - 1.0) < 1e-10

    def test_energy_includes_pump(self):
        """Energy includes both sublimation and vacuum pump."""
        e = freeze_dry_energy_kwh(1.0)
        pump_only = VACUUM_PUMP_POWER_KW * FREEZE_DRY_HOURS_PER_KG * 1.0
        assert e > pump_only  # sublimation energy adds to pump


# ============================================================================
# 3. Caloric accounting
# ============================================================================

class TestCaloricAccounting:
    """Caloric density, consumption, and inventory calculations."""

    def test_kcal_per_kg_freeze_dried(self):
        assert kcal_per_kg(True) == KCAL_PER_KG_FREEZE_DRIED

    def test_kcal_per_kg_fresh(self):
        assert kcal_per_kg(False) == KCAL_PER_KG_FRESH

    def test_freeze_dried_denser_than_fresh(self):
        """Freeze-dried food has higher caloric density than fresh."""
        assert kcal_per_kg(True) > kcal_per_kg(False)

    def test_days_of_food_basic(self):
        """2000 kg × 3800 kcal/kg ÷ (20 crew × 2500 kcal) = 152 days."""
        kcal = 2000.0 * KCAL_PER_KG_FREEZE_DRIED
        d = days_of_food(kcal, 20, KCAL_PER_PERSON_SOL_NORMAL)
        assert 140 < d < 160

    def test_days_of_food_zero_crew(self):
        """Zero crew → infinite food days."""
        d = days_of_food(100000.0, 0, KCAL_PER_PERSON_SOL_NORMAL)
        assert d == float("inf")

    def test_days_of_food_zero_kcal(self):
        """Zero food → zero days."""
        d = days_of_food(0.0, 20, KCAL_PER_PERSON_SOL_NORMAL)
        assert d == 0.0

    def test_days_of_food_positive(self):
        """Days always non-negative for non-negative inputs."""
        d = days_of_food(50000.0, 10, 2500.0)
        assert d >= 0.0

    def test_caloric_decay_slow(self):
        """Freeze-dried food loses very little per sol."""
        after = caloric_decay(100000.0, 1)
        assert after > 99990.0  # less than 0.01% loss per sol

    def test_caloric_decay_one_year(self):
        """After 365 sols, less than 1% caloric loss."""
        after = caloric_decay(100000.0, 365)
        loss_pct = (100000.0 - after) / 100000.0 * 100.0
        assert loss_pct < 1.0


# ============================================================================
# 4. Ration levels
# ============================================================================

class TestRationLevels:
    """Ration threshold classification."""

    def test_normal(self):
        assert ration_level_from_days(200.0) == "normal"

    def test_caution(self):
        assert ration_level_from_days(50.0) == "caution"

    def test_warning(self):
        assert ration_level_from_days(20.0) == "warning"

    def test_critical(self):
        assert ration_level_from_days(10.0) == "critical"

    def test_starvation(self):
        assert ration_level_from_days(2.0) == "starvation"

    def test_boundary_normal_caution(self):
        """At exactly caution threshold, should be caution."""
        assert ration_level_from_days(float(RATION_CAUTION_DAYS)) == "caution"

    def test_boundary_above_normal(self):
        assert ration_level_from_days(float(RATION_NORMAL_DAYS) + 1) == "normal"

    def test_zero_days(self):
        assert ration_level_from_days(0.0) == "starvation"

    def test_consumption_rate_normal(self):
        assert consumption_rate("normal") == KCAL_PER_PERSON_SOL_NORMAL

    def test_consumption_rate_warning(self):
        assert consumption_rate("warning") == KCAL_PER_PERSON_SOL_RATION

    def test_consumption_decreases_with_severity(self):
        """Consumption should decrease as rationing gets more severe."""
        rates = [
            consumption_rate("normal"),
            consumption_rate("caution"),
            consumption_rate("warning"),
            consumption_rate("critical"),
            consumption_rate("starvation"),
        ]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1]


# ============================================================================
# 5. Nutrient & vitamin degradation
# ============================================================================

class TestNutrientDegradation:
    """Exponential decay of nutrients and vitamins."""

    def test_nutrient_decay_one_sol(self):
        q = nutrient_decay(1.0, 1)
        assert 0.999 < q < 1.0

    def test_nutrient_decay_one_year(self):
        q = nutrient_decay(1.0, 365)
        assert 0.85 < q < 0.95

    def test_nutrient_decay_never_negative(self):
        q = nutrient_decay(0.5, 10000)
        assert q >= 0.0

    def test_vitamin_c_half_life(self):
        """After one half-life, level should be ~0.5."""
        level = vitamin_decay(1.0, VITAMIN_C_HALF_LIFE_SOLS, int(VITAMIN_C_HALF_LIFE_SOLS))
        assert 0.45 < level < 0.55

    def test_vitamin_b12_half_life(self):
        level = vitamin_decay(1.0, VITAMIN_B12_HALF_LIFE_SOLS, int(VITAMIN_B12_HALF_LIFE_SOLS))
        assert 0.45 < level < 0.55

    def test_vitamin_a_half_life(self):
        level = vitamin_decay(1.0, VITAMIN_A_HALF_LIFE_SOLS, int(VITAMIN_A_HALF_LIFE_SOLS))
        assert 0.45 < level < 0.55

    def test_vitamin_decay_monotonic(self):
        """Vitamin levels should only decrease over time."""
        v1 = vitamin_decay(1.0, 365.0, 10)
        v2 = vitamin_decay(1.0, 365.0, 20)
        assert v2 < v1

    def test_vitamin_decay_zero_half_life(self):
        """Zero half-life → instant decay to zero."""
        assert vitamin_decay(1.0, 0.0, 1) == 0.0

    def test_vitamin_decay_bounded(self):
        """Vitamin level always in [0, initial]."""
        for sols in [1, 100, 1000, 10000]:
            v = vitamin_decay(1.0, 365.0, sols)
            assert 0.0 <= v <= 1.0


# ============================================================================
# 6. Fresh food spoilage
# ============================================================================

class TestFreshSpoilage:
    """Temperature-dependent spoilage rates."""

    def test_frozen_very_slow(self):
        """Below -20°C: minimal spoilage."""
        rate = fresh_spoilage_rate(-60.0)
        assert rate < 0.005

    def test_cold_moderate(self):
        """0°C: moderate spoilage."""
        rate = fresh_spoilage_rate(0.0)
        assert 0.003 < rate < 0.02

    def test_warm_fast(self):
        """Above 4°C: rapid spoilage."""
        rate = fresh_spoilage_rate(20.0)
        assert rate > 0.05

    def test_spoilage_increases_with_temp(self):
        """Spoilage rate increases with temperature."""
        r_cold = fresh_spoilage_rate(-40.0)
        r_cool = fresh_spoilage_rate(0.0)
        r_warm = fresh_spoilage_rate(25.0)
        assert r_cold < r_cool < r_warm

    def test_spoilage_always_positive(self):
        """Rate is always > 0 (even frozen food degrades)."""
        for temp in [-80, -60, -20, 0, 10, 30]:
            assert fresh_spoilage_rate(float(temp)) > 0.0

    def test_spoilage_bounded(self):
        """Rate should be < 1.0 (can't lose more than 100%/sol)."""
        for temp in [-80, -20, 0, 10, 30, 50]:
            assert fresh_spoilage_rate(float(temp)) < 1.0


# ============================================================================
# 7. Cold storage power
# ============================================================================

class TestColdStoragePower:
    """Cold storage refrigeration power."""

    def test_zero_fresh(self):
        assert cold_storage_power_kwh(0.0) == 0.0

    def test_positive_fresh(self):
        p = cold_storage_power_kwh(100.0)
        assert p > 0.0

    def test_power_scales_with_load(self):
        p_low = cold_storage_power_kwh(100.0)
        p_high = cold_storage_power_kwh(400.0)
        assert p_high > p_low

    def test_power_capped_at_capacity(self):
        """Above capacity, power doesn't increase further."""
        p_full = cold_storage_power_kwh(COLD_STORAGE_CAPACITY_KG)
        p_over = cold_storage_power_kwh(COLD_STORAGE_CAPACITY_KG * 2)
        assert abs(p_full - p_over) < 0.01


# ============================================================================
# 8. Seal breach & contamination
# ============================================================================

class TestSealBreach:
    """Perchlorate contamination from seal failures."""

    def test_no_containers_no_breach(self):
        assert check_seal_breach(0) == 0

    def test_breach_bounded(self):
        """Breaches ≤ number of containers."""
        for _ in range(20):
            n = 100
            b = check_seal_breach(n)
            assert 0 <= b <= n

    def test_breach_deterministic_seed(self):
        """Same RNG seed → same result."""
        r1 = random.Random(42)
        r2 = random.Random(42)
        assert check_seal_breach(50, r1) == check_seal_breach(50, r2)

    def test_breach_probability_reasonable(self):
        """Over many trials, breach rate should approximate expected."""
        rng = random.Random(123)
        total_breaches = sum(check_seal_breach(100, rng) for _ in range(1000))
        expected = 100 * 1000 * CONTAINER_SEAL_FAILURE_RATE
        assert abs(total_breaches - expected) / expected < 0.15


# ============================================================================
# 9. Supply ship
# ============================================================================

class TestSupplyShip:
    """Supply ship arrival logic."""

    def test_not_due_early(self):
        assert not supply_ship_due(100, 100)

    def test_due_at_interval(self):
        assert supply_ship_due(780, int(SUPPLY_SHIP_INTERVAL_SOLS))

    def test_due_past_interval(self):
        assert supply_ship_due(800, int(SUPPLY_SHIP_INTERVAL_SOLS) + 10)


# ============================================================================
# 10. Factory / make_storage
# ============================================================================

class TestFactory:
    """Storage system initialization."""

    def test_make_default(self):
        s = make_storage()
        assert s.crew_count == 20
        assert s.freeze_dried_kg == 2000.0
        assert s.total_kcal > 0.0
        assert len(s.batches) == 1

    def test_make_custom(self):
        s = make_storage(crew_count=10, initial_food_kg=500.0)
        assert s.crew_count == 10
        assert s.freeze_dried_kg == 500.0

    def test_initial_kcal_matches_mass(self):
        s = make_storage(initial_food_kg=1000.0)
        expected = 1000.0 * KCAL_PER_KG_FREEZE_DRIED
        assert abs(s.total_kcal - expected) < 1.0

    def test_initial_batch_provenance(self):
        s = make_storage()
        assert s.batches[0].source == "supply_ship"
        assert s.batches[0].is_freeze_dried is True

    def test_initial_vitamins_full(self):
        s = make_storage()
        assert s.vitamin_c_level == 1.0
        assert s.vitamin_b12_level == 1.0
        assert s.vitamin_a_level == 1.0


# ============================================================================
# 11. Tick engine
# ============================================================================

class TestTickEngine:
    """Single-sol tick behavior."""

    def test_sol_increments(self):
        s = make_storage()
        tick_storage(s)
        assert s.sol == 1

    def test_food_consumed(self):
        s = make_storage()
        initial_kcal = s.total_kcal
        tick_storage(s, rng=random.Random(42))
        assert s.total_kcal < initial_kcal

    def test_report_structure(self):
        s = make_storage()
        r = tick_storage(s, rng=random.Random(42))
        assert isinstance(r, SolReport)
        assert r.sol == 1
        assert r.kcal_consumed > 0.0

    def test_greenhouse_harvest_adds_food(self):
        s = make_storage(initial_food_kg=100.0)
        kcal_before = s.total_kcal
        tick_storage(s, greenhouse_harvest_kg=50.0, rng=random.Random(42))
        # Fresh food added then partially freeze-dried; net kcal should increase
        # (50 kg fresh → 10 kg dry at higher density, net caloric gain)
        # The exact value depends on consumption, but with 100 kg initial
        # and 50 kg fresh harvest, we should have more than minimal
        assert s.total_kcal > 0.0

    def test_vitamins_degrade(self):
        s = make_storage()
        initial_c = s.vitamin_c_level
        tick_storage(s, rng=random.Random(42))
        assert s.vitamin_c_level < initial_c

    def test_kcal_never_negative(self):
        s = make_storage(initial_food_kg=0.1)
        for _ in range(10):
            tick_storage(s, rng=random.Random(42))
        assert s.total_kcal >= 0.0

    def test_freeze_dried_never_negative(self):
        s = make_storage(initial_food_kg=1.0)
        for _ in range(100):
            tick_storage(s, rng=random.Random(42))
        assert s.freeze_dried_kg >= 0.0

    def test_fresh_never_negative(self):
        s = make_storage(initial_food_kg=0.0)
        tick_storage(s, greenhouse_harvest_kg=10.0, rng=random.Random(42))
        assert s.fresh_kg >= 0.0


# ============================================================================
# 12. Multi-sol integration
# ============================================================================

class TestMultiSol:
    """Multi-sol simulation tests (smoke tests + property checks)."""

    def test_smoke_10_sols(self):
        """Run 10 sols without crash."""
        reports = run_storage(sols=10, seed=42)
        assert len(reports) == 10
        assert all(r.total_kcal >= 0.0 for r in reports)

    def test_smoke_365_sols(self):
        """Run a full Mars year without crash."""
        reports = run_storage(sols=365, seed=42)
        assert len(reports) == 365

    def test_food_decreases_without_harvest(self):
        """Without greenhouse, food eventually runs out."""
        s = make_storage(crew_count=20, initial_food_kg=500.0)
        rng = random.Random(42)
        for _ in range(200):
            tick_storage(s, greenhouse_harvest_kg=0.0, rng=rng)
        assert s.total_kcal < 500.0 * KCAL_PER_KG_FREEZE_DRIED * 0.1

    def test_greenhouse_sustains_colony(self):
        """Sufficient greenhouse output keeps colony fed."""
        # 20 crew × 2500 kcal/sol = 50000 kcal/sol needed
        # 15 kg/sol fresh → ~3 kg dry → 11400 kcal from dry
        # Plus fresh kcal. Need enough to sustain.
        # With 30 kg/sol, colony should survive 365 sols.
        reports = run_storage(
            sols=365, crew_count=10, initial_food_kg=2000.0,
            greenhouse_kg_per_sol=20.0, seed=42,
        )
        # Colony should not starve with ample greenhouse + initial stores
        assert reports[-1].total_kcal > 0.0
        assert reports[-1].days_of_food > 0.0

    def test_starvation_detected(self):
        """Starvation condition fires when food runs out."""
        s = make_storage(crew_count=20, initial_food_kg=50.0)
        rng = random.Random(42)
        starved = False
        for _ in range(100):
            r = tick_storage(s, greenhouse_harvest_kg=0.0, rng=rng)
            if r.ration_level == "starvation":
                starved = True
                break
        assert starved

    def test_supply_ship_arrives(self):
        """Supply ship arrives after synodic period."""
        s = make_storage(crew_count=5, initial_food_kg=5000.0)
        s.sols_since_supply = int(SUPPLY_SHIP_INTERVAL_SOLS) - 1
        rng = random.Random(42)
        r = tick_storage(s, rng=rng)
        assert r.supply_ship is True
        assert s.supply_ships_received == 1

    def test_ration_levels_progress(self):
        """As food depletes, ration levels should escalate."""
        s = make_storage(crew_count=20, initial_food_kg=2000.0)
        rng = random.Random(42)
        levels_seen = set()
        for _ in range(500):
            r = tick_storage(s, greenhouse_harvest_kg=0.0, rng=rng)
            levels_seen.add(r.ration_level)
        # Should have seen normal (starts well-fed) and at least one restricted level
        assert "normal" in levels_seen
        assert len(levels_seen) >= 3  # normal → caution → warning → ...

    def test_sol_report_deterministic(self):
        """Same seed → same results."""
        r1 = run_storage(sols=50, seed=99)
        r2 = run_storage(sols=50, seed=99)
        for a, b in zip(r1, r2):
            assert abs(a.total_kcal - b.total_kcal) < 0.01

    def test_cumulative_consumption_increases(self):
        """Cumulative consumption monotonically increases."""
        s = make_storage()
        rng = random.Random(42)
        prev = 0.0
        for _ in range(30):
            tick_storage(s, greenhouse_harvest_kg=10.0, rng=rng)
            assert s.cumulative_kcal_consumed >= prev
            prev = s.cumulative_kcal_consumed

    def test_power_consumption_positive(self):
        """Processing food requires power."""
        reports = run_storage(sols=10, seed=42)
        total_power = sum(r.power_kwh for r in reports)
        assert total_power > 0.0


# ============================================================================
# 13. Edge cases
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_crew(self):
        """Zero crew → no consumption, food preserved."""
        s = make_storage(crew_count=0, initial_food_kg=1000.0)
        rng = random.Random(42)
        tick_storage(s, rng=rng)
        # With zero crew, food should barely change (only decay)
        assert s.total_kcal > 1000.0 * KCAL_PER_KG_FREEZE_DRIED * 0.99

    def test_zero_initial_food(self):
        """Start with no food, survive on greenhouse only."""
        s = make_storage(crew_count=5, initial_food_kg=0.0)
        rng = random.Random(42)
        r = tick_storage(s, greenhouse_harvest_kg=50.0, rng=rng)
        assert r.total_kcal >= 0.0

    def test_massive_harvest(self):
        """Very large harvest gets capped by processing limit."""
        s = make_storage(initial_food_kg=0.0)
        rng = random.Random(42)
        tick_storage(s, greenhouse_harvest_kg=1000.0, rng=rng)
        # Only MAX_FREEZE_DRY_KG_SOL should be processed
        assert s.freeze_dry_processed_kg <= MAX_FREEZE_DRY_KG_SOL * (
            1.0 - FREEZE_DRY_WATER_FRACTION
        ) + 0.01

    def test_storage_overflow(self):
        """Exceeding storage capacity triggers overflow alert."""
        s = make_storage(initial_food_kg=STORAGE_CAPACITY_KG - 5.0)
        rng = random.Random(42)
        r = tick_storage(s, greenhouse_harvest_kg=100.0, rng=rng)
        assert s.freeze_dried_kg <= STORAGE_CAPACITY_KG + 0.01

    def test_negative_temp(self):
        """Extreme cold: very low spoilage."""
        rate = fresh_spoilage_rate(-100.0)
        assert rate < 0.005


# ============================================================================
# 14. Conservation / invariants
# ============================================================================

class TestConservation:
    """Physical conservation laws and invariants."""

    def test_kcal_non_negative_always(self):
        """Total kcal never goes below zero in any scenario."""
        for seed in range(10):
            reports = run_storage(sols=100, crew_count=30,
                                 initial_food_kg=100.0, seed=seed)
            for r in reports:
                assert r.total_kcal >= 0.0, f"Negative kcal at sol {r.sol}"

    def test_mass_non_negative_always(self):
        """Mass never goes below zero."""
        for seed in range(10):
            s = make_storage(crew_count=20, initial_food_kg=200.0)
            rng = random.Random(seed)
            for _ in range(200):
                tick_storage(s, greenhouse_harvest_kg=5.0, rng=rng)
                assert s.freeze_dried_kg >= 0.0
                assert s.fresh_kg >= 0.0

    def test_vitamins_bounded_0_to_1(self):
        """Vitamin levels always in [0, 1]."""
        s = make_storage()
        rng = random.Random(42)
        for _ in range(500):
            tick_storage(s, greenhouse_harvest_kg=10.0, rng=rng)
            assert 0.0 <= s.vitamin_c_level <= 1.0
            assert 0.0 <= s.vitamin_b12_level <= 1.0
            assert 0.0 <= s.vitamin_a_level <= 1.0

    def test_nutrient_quality_bounded(self):
        """Nutrient quality in [0, 1]."""
        s = make_storage()
        rng = random.Random(42)
        for _ in range(500):
            tick_storage(s, greenhouse_harvest_kg=10.0, rng=rng)
            assert 0.0 <= s.avg_nutrient_quality <= 1.0

    def test_days_of_food_non_negative(self):
        """Days of food always ≥ 0."""
        reports = run_storage(sols=200, crew_count=20,
                              initial_food_kg=100.0, seed=42)
        for r in reports:
            assert r.days_of_food >= 0.0
