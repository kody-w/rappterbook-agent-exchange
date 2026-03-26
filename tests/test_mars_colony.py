"""
Tests for src/mars_colony.py — Mars colony population model.

Property-based invariants: population never negative, resources never
negative, carrying capacity always positive, morale bounded [0,1],
conservation laws hold across ticks.

Run: python -m pytest tests/test_mars_colony.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars_colony import (
    ACCIDENT_RATE,
    BASE_DEATH_RATE,
    COLONY_BIRTH_RATE,
    EPIDEMIC_CHANCE_PER_SOL,
    EPIDEMIC_MIN_POP,
    EPIDEMIC_STRAINS,
    FOOD_KG_SOL,
    GREENHOUSE_KG_SOL_M2,
    HABITAT_M2_MIN,
    NUCLEAR_POWER_KWH,
    POWER_KWH_SOL,
    RADIATION_CONCERN,
    RADIATION_DANGER,
    RADIATION_LETHAL,
    REPRODUCTIVE_FRACTION,
    SOLAR_PANEL_KWH_M2,
    SUPPLY_SHIP_COLONISTS,
    SUPPLY_SHIP_INTERVAL,
    TERRAFORM_BASE_RATE,
    TERRAFORM_STRATEGY_MOD,
    WATER_L_SOL,
    WATER_RECYCLE_RATE,
    Colony,
    Epidemic,
    create_colony,
)


# ── Helper: minimal env snapshot matching MarsEnvironment.tick() output ──


def make_env(
    sol: int = 1,
    temperature_c: float = -60.0,
    solar_flux_wm2: float = 500.0,
    dust_opacity: float = 0.0,
    radiation_msv: float = 0.6,
    storm: str | None = None,
    flare: bool = False,
    pressure_kpa: float = 0.636,
    terraforming_progress: float = 0.0,
) -> dict:
    """Build an env dict matching MarsEnvironment.tick() output."""
    return {
        "sol": sol,
        "ls": 0.0,
        "season": "spring",
        "temperature_c": temperature_c,
        "solar_flux_wm2": solar_flux_wm2,
        "dust_opacity": dust_opacity,
        "radiation_msv": radiation_msv,
        "storm": storm,
        "flare": flare,
        "pressure_kpa": pressure_kpa,
        "terraforming_progress": terraforming_progress,
    }


def make_colony(**overrides) -> Colony:
    """Create a small test colony with sane defaults."""
    defaults = dict(
        name="TestColony",
        population=50,
        food_kg=50 * FOOD_KG_SOL * 100,
        water_l=50 * WATER_L_SOL * (1 - WATER_RECYCLE_RATE) * 100,
        power_kwh=50 * POWER_KWH_SOL * 3,
        habitat_m2=50 * HABITAT_M2_MIN * 1.2,
        greenhouse_m2=1000,
        solar_m2=1000,
        medical_level=0.5,
        morale=0.7,
        strategy="balanced",
        seed=42,
    )
    defaults.update(overrides)
    return Colony(**defaults)


# ── Epidemic tests ───────────────────────────────────────────────


class TestEpidemic:
    """Tests for the Epidemic helper class."""

    def test_construction(self) -> None:
        strain = EPIDEMIC_STRAINS[0]  # Mars Flu
        e = Epidemic(strain, duration=20, population=100)
        assert e.strain == "Mars Flu"
        assert e.remaining_sols == 20
        assert e.total_duration == 20
        assert e.infected_count == 5  # 5% of 100

    def test_min_infected(self) -> None:
        """Even tiny populations infect at least 1."""
        strain = EPIDEMIC_STRAINS[0]
        e = Epidemic(strain, duration=10, population=1)
        assert e.infected_count >= 1

    def test_infection_rate_starts_low(self) -> None:
        """At the start of an epidemic, infection rate is near 0."""
        strain = EPIDEMIC_STRAINS[1]  # Regolith Lung
        e = Epidemic(strain, duration=30, population=100)
        assert e.infection_rate() < 0.01

    def test_infection_rate_peaks_mid(self) -> None:
        """Infection rate peaks around 30% progress."""
        strain = EPIDEMIC_STRAINS[2]  # Rad Fever
        e = Epidemic(strain, duration=50, population=100)
        # Advance to ~30% progress (35 remaining of 50 = 30% done)
        for _ in range(15):
            e.tick()
        peak_rate = e.infection_rate()
        assert peak_rate > 0.1, f"Expected peak > 0.1, got {peak_rate}"

    def test_tick_decrements(self) -> None:
        strain = EPIDEMIC_STRAINS[0]
        e = Epidemic(strain, duration=5, population=50)
        assert e.tick() is True
        assert e.remaining_sols == 4
        for _ in range(3):
            e.tick()
        assert e.tick() is False  # duration=5, ticked 5 times
        assert e.remaining_sols == 0

    def test_quarantine_reduces_mortality(self) -> None:
        strain = EPIDEMIC_STRAINS[2]  # highest severity
        e = Epidemic(strain, duration=30, population=100)
        for _ in range(10):
            e.tick()
        unquarantined = e.extra_mortality()
        e.quarantined = True
        quarantined = e.extra_mortality()
        assert quarantined < unquarantined


# ── Colony construction tests ────────────────────────────────────


class TestColonyConstruction:
    def test_basic_creation(self) -> None:
        c = make_colony()
        assert c.name == "TestColony"
        assert c.population == 50
        assert c.morale == 0.7

    def test_medical_clamped(self) -> None:
        c = make_colony(medical_level=1.5)
        assert c.medical_level == 1.0
        c2 = make_colony(medical_level=-0.5)
        assert c2.medical_level == 0.0

    def test_morale_clamped(self) -> None:
        c = make_colony(morale=2.0)
        assert c.morale == 1.0
        c2 = make_colony(morale=-1.0)
        assert c2.morale == 0.0

    def test_initial_tracking(self) -> None:
        c = make_colony(population=80)
        assert c.total_births == 0
        assert c.total_deaths == 0
        assert c.sol == 0
        assert c.cumulative_radiation_msv == 0.0
        assert c.initial_population == 80

    def test_genetic_diversity_scales_with_pop(self) -> None:
        small = make_colony(population=50)
        large = make_colony(population=300)
        assert small.genetic_diversity < large.genetic_diversity
        assert large.genetic_diversity == 1.0  # 300/200 clamped to 1.0


# ── Carrying capacity tests ─────────────────────────────────────


class TestCarryingCapacity:
    def test_positive(self) -> None:
        c = make_colony()
        assert c.carrying_capacity() > 0

    def test_bigger_habitat_bigger_k(self) -> None:
        small = make_colony(habitat_m2=500)
        large = make_colony(habitat_m2=5000)
        assert large.carrying_capacity() >= small.carrying_capacity()

    def test_minimum_floor(self) -> None:
        """Even with tiny resources, K >= 2."""
        c = make_colony(
            habitat_m2=1.0,
            greenhouse_m2=1.0,
            solar_m2=1.0,
            population=1,
        )
        assert c.carrying_capacity() >= 2.0


# ── Resource consumption tests ───────────────────────────────────


class TestResourceConsumption:
    def test_zero_pop_no_consumption(self) -> None:
        c = make_colony(population=0)
        initial_food = c.food_kg
        ratios = c._consume_resources()
        assert ratios == {"food": 1.0, "water": 1.0, "power": 1.0}
        assert c.food_kg == initial_food

    def test_adequate_resources_full_ratios(self) -> None:
        c = make_colony()  # plenty of reserves
        ratios = c._consume_resources()
        assert ratios["food"] == 1.0
        assert ratios["water"] == 1.0
        assert ratios["power"] == 1.0

    def test_resources_decrease(self) -> None:
        c = make_colony()
        food_before = c.food_kg
        water_before = c.water_l
        power_before = c.power_kwh
        c._consume_resources()
        assert c.food_kg < food_before
        assert c.water_l < water_before
        assert c.power_kwh < power_before

    def test_shortage_ratios(self) -> None:
        c = make_colony(food_kg=1.0)  # very low food
        ratios = c._consume_resources()
        assert ratios["food"] < 1.0
        assert c.food_kg == 0.0  # consumed everything

    def test_resources_never_negative(self) -> None:
        c = make_colony(food_kg=0.0, water_l=0.0, power_kwh=0.0)
        c._consume_resources()
        assert c.food_kg >= 0
        assert c.water_l >= 0
        assert c.power_kwh >= 0


# ── Resource production tests ────────────────────────────────────


class TestResourceProduction:
    def test_produces_food(self) -> None:
        c = make_colony(food_kg=0.0)
        c._produce_resources(500.0, 590.0)
        assert c.food_kg > 0

    def test_produces_power(self) -> None:
        c = make_colony(power_kwh=0.0)
        c._produce_resources(500.0, 590.0)
        assert c.power_kwh > 0

    def test_produces_water(self) -> None:
        c = make_colony(water_l=0.0)
        c._produce_resources(500.0, 590.0)
        assert c.water_l > 0

    def test_low_flux_less_food(self) -> None:
        c1 = make_colony(food_kg=0.0)
        c2 = make_colony(food_kg=0.0)
        c1._produce_resources(100.0, 590.0)
        c2._produce_resources(500.0, 590.0)
        assert c2.food_kg > c1.food_kg

    def test_terraforming_boosts_food(self) -> None:
        c1 = make_colony(food_kg=0.0)
        c2 = make_colony(food_kg=0.0)
        c1._produce_resources(500.0, 590.0, terraforming_progress=0.0)
        c2._produce_resources(500.0, 590.0, terraforming_progress=0.5)
        assert c2.food_kg > c1.food_kg

    def test_nuclear_provides_minimum_power(self) -> None:
        """Even zero solar flux produces nuclear baseline."""
        c = make_colony(power_kwh=0.0, solar_m2=0.0)
        c._produce_resources(0.0, 590.0)
        assert c.power_kwh >= NUCLEAR_POWER_KWH


# ── Storm damage tests ───────────────────────────────────────────


class TestStormDamage:
    def test_no_storm_no_damage(self) -> None:
        c = make_colony()
        solar_before = c.solar_m2
        c._storm_damage(make_env(storm=None))
        assert c.solar_m2 == solar_before

    def test_global_storm_damages_solar(self) -> None:
        c = make_colony(solar_m2=1000.0)
        c._storm_damage(make_env(storm="global"))
        assert c.solar_m2 < 1000.0

    def test_global_storm_damages_greenhouse(self) -> None:
        c = make_colony(greenhouse_m2=1000.0)
        c._storm_damage(make_env(storm="global"))
        assert c.greenhouse_m2 < 1000.0

    def test_regional_less_than_global(self) -> None:
        c1 = make_colony(solar_m2=1000.0)
        c2 = make_colony(solar_m2=1000.0)
        c1._storm_damage(make_env(storm="regional"))
        c2._storm_damage(make_env(storm="global"))
        assert c1.solar_m2 > c2.solar_m2  # regional does less damage

    def test_floor_prevents_destruction(self) -> None:
        """Storm damage can't reduce panels below 50 or greenhouse below 20."""
        c = make_colony(solar_m2=51.0, greenhouse_m2=21.0)
        for _ in range(100):
            c._storm_damage(make_env(storm="global"))
        assert c.solar_m2 >= 50.0
        assert c.greenhouse_m2 >= 20.0


# ── Morale tests ─────────────────────────────────────────────────


class TestMorale:
    def test_good_conditions_raise_morale(self) -> None:
        c = make_colony(morale=0.3)
        ratios = {"food": 1.0, "water": 1.0, "power": 1.0}
        for _ in range(50):
            c._update_morale(ratios, make_env())
        assert c.morale > 0.3

    def test_storms_depress_morale(self) -> None:
        c1 = make_colony(morale=0.7)
        c2 = make_colony(morale=0.7)
        ratios = {"food": 1.0, "water": 1.0, "power": 1.0}
        c1._update_morale(ratios, make_env(storm=None))
        c2._update_morale(ratios, make_env(storm="global"))
        assert c2.morale < c1.morale

    def test_morale_bounded(self) -> None:
        c = make_colony(morale=0.5)
        for _ in range(200):
            ratios = {"food": 1.0, "water": 1.0, "power": 1.0}
            c._update_morale(ratios, make_env())
        assert 0.0 <= c.morale <= 1.0


# ── Birth / death tests ─────────────────────────────────────────


class TestBirths:
    def test_zero_pop_no_births(self) -> None:
        c = make_colony(population=0)
        births = c._compute_births({"food": 1.0, "water": 1.0, "power": 1.0})
        assert births == 0

    def test_one_person_no_births(self) -> None:
        c = make_colony(population=1)
        births = c._compute_births({"food": 1.0, "water": 1.0, "power": 1.0})
        assert births == 0

    def test_supply_ship_at_interval(self) -> None:
        """On supply ship sols, extra colonists arrive."""
        c = make_colony(population=50, strategy="balanced")
        c.sol = SUPPLY_SHIP_INTERVAL  # supply ship arrives on this sol
        births = c._compute_births({"food": 1.0, "water": 1.0, "power": 1.0})
        assert births >= SUPPLY_SHIP_COLONISTS["balanced"]


class TestDeaths:
    def test_zero_pop_no_deaths(self) -> None:
        c = make_colony(population=0)
        deaths, causes = c._compute_deaths(
            {"food": 1.0, "water": 1.0, "power": 1.0}, make_env()
        )
        assert deaths == 0

    def test_deaths_never_exceed_population(self) -> None:
        c = make_colony(population=5)
        c.cumulative_radiation_msv = 2000  # extreme radiation
        deaths, _ = c._compute_deaths(
            {"food": 0.0, "water": 0.0, "power": 0.0}, make_env(storm="global")
        )
        assert deaths <= 5

    def test_starvation_deaths(self) -> None:
        """Severe food shortage causes starvation deaths over time."""
        c = make_colony(population=100, seed=1)
        total_starved = 0
        for _ in range(100):
            _, causes = c._compute_deaths(
                {"food": 0.1, "water": 1.0, "power": 1.0}, make_env()
            )
            total_starved += causes.get("starvation", 0)
        assert total_starved > 0, "Expected some starvation deaths"


# ── Infrastructure expansion tests ───────────────────────────────


class TestInfrastructure:
    def test_crowded_expands_habitat(self) -> None:
        """When density > 0.8, habitat expands."""
        c = make_colony(
            population=100,
            habitat_m2=100 * HABITAT_M2_MIN * 0.7,  # density > 1.0
        )
        habitat_before = c.habitat_m2
        c._expand_infrastructure()
        assert c.habitat_m2 > habitat_before

    def test_low_food_expands_greenhouse(self) -> None:
        c = make_colony(food_kg=1.0)  # tiny food reserve
        gh_before = c.greenhouse_m2
        c._expand_infrastructure()
        assert c.greenhouse_m2 > gh_before

    def test_zero_pop_no_expansion(self) -> None:
        c = make_colony(population=0)
        h, g, s = c.habitat_m2, c.greenhouse_m2, c.solar_m2
        c._expand_infrastructure()
        assert c.habitat_m2 == h
        assert c.greenhouse_m2 == g
        assert c.solar_m2 == s

    def test_aggressive_expands_faster(self) -> None:
        c_bal = make_colony(
            population=100,
            habitat_m2=100 * HABITAT_M2_MIN * 0.7,
            strategy="balanced",
        )
        c_agg = make_colony(
            population=100,
            habitat_m2=100 * HABITAT_M2_MIN * 0.7,
            strategy="aggressive",
        )
        c_bal._expand_infrastructure()
        c_agg._expand_infrastructure()
        assert c_agg.habitat_m2 > c_bal.habitat_m2


# ── Terraforming tests ───────────────────────────────────────────


class TestTerraforming:
    def test_zero_pop_zero_terraform(self) -> None:
        c = make_colony(population=0)
        delta = c._compute_terraforming()
        assert delta == 0.0

    def test_positive_population_produces(self) -> None:
        c = make_colony(population=100)
        delta = c._compute_terraforming()
        assert delta > 0

    def test_aggressive_contributes_more(self) -> None:
        c_con = make_colony(population=100, strategy="conservative")
        c_agg = make_colony(population=100, strategy="aggressive")
        d_con = c_con._compute_terraforming()
        d_agg = c_agg._compute_terraforming()
        assert d_agg > d_con

    def test_cumulative_tracking(self) -> None:
        c = make_colony(population=50)
        c._compute_terraforming()
        c._compute_terraforming()
        assert c.terraforming_output > 0


# ── Genetic diversity tests ──────────────────────────────────────


class TestGeneticDiversity:
    def test_small_pop_loses_diversity(self) -> None:
        """Small populations lose diversity over 30-sol cycles."""
        c = make_colony(population=10)
        c.genetic_diversity = 0.5
        initial = c.genetic_diversity
        for sol in range(1, 91):
            c.sol = sol
            c._drift_genetic_diversity()
        assert c.genetic_diversity < initial

    def test_diversity_floor(self) -> None:
        """Diversity never goes below 0.05."""
        c = make_colony(population=2)
        c.genetic_diversity = 0.06
        for sol in range(1, 10000):
            c.sol = sol
            c._drift_genetic_diversity()
        assert c.genetic_diversity >= 0.05

    def test_immigrants_boost_diversity(self) -> None:
        c = make_colony(population=50)
        c.genetic_diversity = 0.3
        c.receive_immigrants(20)
        assert c.genetic_diversity > 0.3


# ── Full tick integration tests ──────────────────────────────────


class TestTick:
    def test_single_tick_returns_snapshot(self) -> None:
        c = make_colony()
        snap = c.tick(make_env())
        assert "sol" in snap
        assert "population" in snap
        assert "food_kg" in snap
        assert "morale" in snap
        assert snap["sol"] == 1

    def test_population_never_negative(self) -> None:
        """Property: population >= 0 always, even under catastrophic conditions."""
        c = make_colony(population=10, food_kg=0, water_l=0, power_kwh=0, seed=7)
        for sol in range(1, 101):
            snap = c.tick(make_env(
                sol=sol, radiation_msv=5.0, storm="global",
            ))
            assert snap["population"] >= 0, f"Negative pop at sol {sol}"

    def test_resources_never_negative(self) -> None:
        """Property: food, water, power >= 0 always."""
        c = make_colony(seed=99)
        for sol in range(1, 51):
            snap = c.tick(make_env(sol=sol))
            assert snap["food_kg"] >= 0
            assert snap["water_l"] >= 0
            assert snap["power_kwh"] >= 0

    def test_morale_always_bounded(self) -> None:
        """Property: morale in [0, 1] at every tick."""
        c = make_colony(seed=55)
        for sol in range(1, 201):
            snap = c.tick(make_env(sol=sol, storm="global" if sol % 3 == 0 else None))
            assert 0.0 <= snap["morale"] <= 1.0, f"Morale out of bounds at sol {sol}"

    def test_carrying_capacity_positive(self) -> None:
        """Property: K > 0 at every tick."""
        c = make_colony(seed=13)
        for sol in range(1, 51):
            snap = c.tick(make_env(sol=sol))
            assert snap["carrying_capacity"] > 0

    def test_ten_sol_smoke(self) -> None:
        """Smoke test: run 10 ticks without crash."""
        c = make_colony(seed=42)
        for sol in range(1, 11):
            snap = c.tick(make_env(sol=sol))
        assert c.sol == 10
        assert len(c.history) == 10

    def test_hundred_sol_determinism(self) -> None:
        """Same seed produces identical 100-sol trajectory."""
        def run(seed: int) -> list[int]:
            c = make_colony(seed=seed)
            pops = []
            for sol in range(1, 101):
                snap = c.tick(make_env(sol=sol))
                pops.append(snap["population"])
            return pops

        a = run(42)
        b = run(42)
        assert a == b, "Determinism violated: same seed different results"

    def test_radiation_accumulates(self) -> None:
        c = make_colony()
        for sol in range(1, 11):
            c.tick(make_env(sol=sol, radiation_msv=1.0))
        assert c.cumulative_radiation_msv > 0

    def test_history_records_all_ticks(self) -> None:
        c = make_colony()
        for sol in range(1, 21):
            c.tick(make_env(sol=sol))
        assert len(c.history) == 20

    def test_births_and_deaths_tracked(self) -> None:
        c = make_colony(seed=123)
        for sol in range(1, 201):
            c.tick(make_env(sol=sol))
        assert c.total_births > 0 or c.total_deaths > 0


# ── create_colony factory tests ──────────────────────────────────


class TestCreateColony:
    def test_conservative(self) -> None:
        c = create_colony("Ares Prime", "conservative", seed=0)
        assert c.population == 120
        assert c.strategy == "conservative"
        assert c.medical_level == 0.8

    def test_balanced(self) -> None:
        c = create_colony("Olympus Station", "balanced", seed=0)
        assert c.population == 80
        assert c.morale == 0.70

    def test_aggressive(self) -> None:
        c = create_colony("Red Frontier", "aggressive", seed=0)
        assert c.population == 60
        assert c.medical_level == 0.4

    def test_all_strategies_viable_100_sols(self) -> None:
        """All three colony archetypes survive 100 sols."""
        for strategy in ("conservative", "balanced", "aggressive"):
            c = create_colony(strategy, strategy, seed=42)
            for sol in range(1, 101):
                c.tick(make_env(sol=sol))
            assert c.population > 0, f"{strategy} colony died in 100 sols"

    def test_conservative_largest_initial(self) -> None:
        con = create_colony("c", "conservative", seed=0)
        bal = create_colony("b", "balanced", seed=0)
        agg = create_colony("a", "aggressive", seed=0)
        assert con.population > bal.population > agg.population


# ── Physical bounds property tests ───────────────────────────────


class TestPhysicalBounds:
    def test_365_sol_full_run(self) -> None:
        """Run a full Mars year. Check invariants every sol."""
        c = create_colony("Test Colony", "balanced", seed=42)
        for sol in range(1, 366):
            snap = c.tick(make_env(sol=sol))
            assert snap["population"] >= 0
            assert snap["food_kg"] >= 0
            assert snap["water_l"] >= 0
            assert snap["power_kwh"] >= 0
            assert 0.0 <= snap["morale"] <= 1.0
            assert snap["carrying_capacity"] > 0
            assert snap["genetic_diversity"] >= 0
            assert snap["genetic_diversity"] <= 1.0

    def test_death_causes_sum_to_deaths(self) -> None:
        """Sum of death causes equals total deaths each sol."""
        c = make_colony(seed=77)
        for sol in range(1, 51):
            snap = c.tick(make_env(sol=sol))
            cause_sum = sum(snap["death_causes"].values())
            assert cause_sum == snap["deaths"], (
                f"Sol {sol}: causes sum {cause_sum} != deaths {snap['deaths']}"
            )

    def test_population_accounting(self) -> None:
        """pop(T+1) = pop(T) + births - deaths (within rounding)."""
        c = make_colony(population=80, seed=42)
        prev_pop = 80
        for sol in range(1, 51):
            snap = c.tick(make_env(sol=sol))
            expected = prev_pop + snap["births"] - snap["deaths"]
            expected = max(0, expected)
            assert snap["population"] == expected, (
                f"Sol {sol}: {prev_pop}+{snap['births']}-{snap['deaths']}"
                f" = {expected} != {snap['population']}"
            )
            prev_pop = snap["population"]


# -- Habitability index tests ----------------------------------------


class TestHabitabilityIndex:
    """Tests for Colony.habitability_index() -- composite viability metric."""

    def test_zero_population_returns_zero(self) -> None:
        c = make_colony(population=0)
        assert c.habitability_index() == 0.0

    def test_healthy_colony_positive(self) -> None:
        c = make_colony()
        assert c.habitability_index() > 0.0

    def test_bounded_zero_one(self) -> None:
        c = make_colony()
        h = c.habitability_index()
        assert 0.0 <= h <= 1.0

    def test_high_food_high_morale_above_half(self) -> None:
        c = make_colony(morale=0.95, food_kg=50 * FOOD_KG_SOL * 200)
        h = c.habitability_index()
        assert h > 0.5

    def test_starvation_tanks_index(self) -> None:
        c = make_colony(food_kg=0.0)
        h = c.habitability_index()
        assert h == 0.0

    def test_zero_morale_tanks_index(self) -> None:
        c = make_colony(morale=0.0)
        h = c.habitability_index()
        assert h == 0.0

    def test_lethal_radiation_tanks_index(self) -> None:
        c = make_colony()
        c.cumulative_radiation_msv = RADIATION_LETHAL
        h = c.habitability_index()
        assert h == 0.0

    def test_medical_breakthroughs_boost(self) -> None:
        c1 = make_colony()
        c2 = make_colony()
        c2.medical_breakthroughs = 4
        assert c2.habitability_index() >= c1.habitability_index()

    def test_genetic_diversity_matters(self) -> None:
        c1 = make_colony()
        c2 = make_colony()
        c2.genetic_diversity = 0.1
        assert c2.habitability_index() < c1.habitability_index()

    def test_monotonic_with_morale(self) -> None:
        """Higher morale -> higher or equal habitability."""
        prev = 0.0
        for m in [0.1, 0.3, 0.5, 0.7, 0.9]:
            c = make_colony(morale=m)
            h = c.habitability_index()
            assert h >= prev, f"morale {m}: {h} < {prev}"
            prev = h

    def test_snapshot_includes_habitability(self) -> None:
        """The tick snapshot should contain the habitability_index field."""
        c = make_colony()
        snap = c.tick(make_env())
        assert "habitability_index" in snap
        assert 0.0 <= snap["habitability_index"] <= 1.0

    def test_full_year_habitability_bounded(self) -> None:
        """Habitability stays in [0,1] across a full Mars year."""
        c = create_colony("Test Colony", "balanced", seed=42)
        for sol in range(1, 366):
            snap = c.tick(make_env(sol=sol))
            assert 0.0 <= snap["habitability_index"] <= 1.0

    def test_deterministic(self) -> None:
        """Same inputs -> same habitability_index."""
        c1 = make_colony(seed=42)
        c2 = make_colony(seed=42)
        assert c1.habitability_index() == c2.habitability_index()

    def test_radiation_degrades_over_time(self) -> None:
        """Accumulating radiation should lower habitability over time."""
        c = make_colony()
        h0 = c.habitability_index()
        c.cumulative_radiation_msv = 500
        h1 = c.habitability_index()
        assert h1 < h0

