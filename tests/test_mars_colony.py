"""
Unit tests for src/mars_colony.py — Epidemic, Colony, create_colony.

The population model is the heart of Mars Barn. 673 lines, zero dedicated
unit tests until now. Community voted: ship code, not governance.

Run: python -m pytest tests/test_mars_colony.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars_colony import (
    Colony,
    Epidemic,
    create_colony,
    FOOD_KG_SOL,
    WATER_L_SOL,
    POWER_KWH_SOL,
    HABITAT_M2_MIN,
    GREENHOUSE_KG_SOL_M2,
    SOLAR_PANEL_KWH_M2,
    NUCLEAR_POWER_KWH,
    WATER_RECYCLE_RATE,
    BASE_DEATH_RATE,
    ACCIDENT_RATE,
    RADIATION_DANGER,
    EPIDEMIC_STRAINS,
    EPIDEMIC_MIN_POP,
    COLONY_BIRTH_RATE,
    REPRODUCTIVE_FRACTION,
    SUPPLY_SHIP_INTERVAL,
    SUPPLY_SHIP_COLONISTS,
    TERRAFORM_BASE_RATE,
    TERRAFORM_STRATEGY_MOD,
)
from src.mars_env import MarsEnvironment


def _make_env_snap(**overrides) -> dict:
    """Minimal environment snapshot for colony.tick()."""
    base = {
        "sol": 1,
        "ls": 90.0,
        "season": "summer",
        "temperature_c": -30.0,
        "solar_flux_wm2": 400.0,
        "dust_opacity": 0.0,
        "radiation_msv": 0.67,
        "storm": None,
        "flare": False,
        "pressure_kpa": 0.636,
        "terraforming_progress": 0.0,
        "terraform_phase": None,
    }
    base.update(overrides)
    return base


# ─── Epidemic class ───


class TestEpidemic:
    """Epidemic is the disease engine. Zero direct tests until now."""

    def test_creation(self) -> None:
        strain = EPIDEMIC_STRAINS[0]  # Mars Flu
        ep = Epidemic(strain, 20, 100)
        assert ep.strain == "Mars Flu"
        assert ep.severity == 0.3
        assert ep.remaining_sols == 20
        assert ep.total_duration == 20
        assert ep.infected_count == 5  # 5% of 100
        assert ep.quarantined is False

    def test_infected_count_minimum(self) -> None:
        """Even tiny populations get at least 1 infected."""
        strain = EPIDEMIC_STRAINS[0]
        ep = Epidemic(strain, 10, 1)
        assert ep.infected_count >= 1

    def test_tick_decrements(self) -> None:
        strain = EPIDEMIC_STRAINS[1]  # Regolith Lung
        ep = Epidemic(strain, 5, 50)
        assert ep.tick() is True
        assert ep.remaining_sols == 4

    def test_tick_returns_false_at_end(self) -> None:
        strain = EPIDEMIC_STRAINS[0]
        ep = Epidemic(strain, 1, 50)
        assert ep.tick() is False
        assert ep.remaining_sols == 0

    def test_infection_rate_curve(self) -> None:
        """Infection rises then falls (SIR-like)."""
        strain = EPIDEMIC_STRAINS[2]  # Rad Fever, severity=0.8
        ep = Epidemic(strain, 100, 200)
        rates = []
        for _ in range(100):
            rates.append(ep.infection_rate())
            ep.tick()
        # Peak should be in the first third, then decline
        peak_idx = rates.index(max(rates))
        assert peak_idx < 50, f"Peak at {peak_idx}, expected <50"
        # Rate should always be non-negative
        assert all(r >= 0 for r in rates)

    def test_infection_rate_bounded(self) -> None:
        """Infection rate never exceeds severity."""
        for strain in EPIDEMIC_STRAINS:
            ep = Epidemic(strain, 30, 100)
            for _ in range(30):
                rate = ep.infection_rate()
                assert 0.0 <= rate <= strain["severity"] + 1e-9
                ep.tick()

    def test_extra_mortality_positive(self) -> None:
        """Epidemic always adds some death risk."""
        strain = EPIDEMIC_STRAINS[1]
        ep = Epidemic(strain, 20, 100)
        # Advance to peak infection
        for _ in range(5):
            ep.tick()
        mort = ep.extra_mortality()
        assert mort > 0

    def test_quarantine_reduces_mortality(self) -> None:
        """Quarantine cuts epidemic mortality by 60%."""
        strain = EPIDEMIC_STRAINS[2]
        ep1 = Epidemic(strain, 20, 100)
        ep2 = Epidemic(strain, 20, 100)
        ep2.quarantined = True
        # Advance both to same point
        for _ in range(5):
            ep1.tick()
            ep2.tick()
        assert ep2.extra_mortality() < ep1.extra_mortality()
        # Quarantine factor is 0.4
        ratio = ep2.extra_mortality() / ep1.extra_mortality()
        assert abs(ratio - 0.4) < 1e-9


# ─── Colony factory ───


class TestCreateColony:
    """create_colony produces viable colonies for all strategies."""

    def test_all_strategies_valid(self) -> None:
        for strategy in ["conservative", "balanced", "aggressive"]:
            c = create_colony(f"test-{strategy}", strategy, 42)
            assert c.population > 0
            assert c.food_kg > 0
            assert c.water_l > 0
            assert c.power_kwh > 0
            assert c.habitat_m2 > 0
            assert c.greenhouse_m2 > 0
            assert c.solar_m2 > 0

    def test_conservative_has_most_reserves(self) -> None:
        con = create_colony("con", "conservative", 1)
        agg = create_colony("agg", "aggressive", 1)
        # Conservative has more food per capita
        con_days = con.food_kg / (con.population * FOOD_KG_SOL)
        agg_days = agg.food_kg / (agg.population * FOOD_KG_SOL)
        assert con_days > agg_days

    def test_aggressive_has_least_population(self) -> None:
        con = create_colony("con", "conservative", 1)
        bal = create_colony("bal", "balanced", 1)
        agg = create_colony("agg", "aggressive", 1)
        assert con.population > bal.population > agg.population

    def test_morale_set_by_strategy(self) -> None:
        con = create_colony("con", "conservative", 1)
        agg = create_colony("agg", "aggressive", 1)
        assert con.morale > agg.morale


# ─── Carrying capacity ───


class TestCarryingCapacity:
    """K = min(habitat, food, water, power). The binding constraint wins."""

    def test_positive(self) -> None:
        c = create_colony("test", "balanced", 42)
        assert c.carrying_capacity() > 0

    def test_minimum_floor(self) -> None:
        """K is at least 2.0 (the floor)."""
        c = Colony(
            name="tiny", population=1, food_kg=0, water_l=0,
            power_kwh=0, habitat_m2=1, greenhouse_m2=0, solar_m2=0,
            seed=1,
        )
        assert c.carrying_capacity() >= 2.0

    def test_habitat_is_bottleneck(self) -> None:
        """Tiny habitat should constrain K regardless of other resources."""
        c = create_colony("test", "balanced", 42)
        c.habitat_m2 = HABITAT_M2_MIN * 5  # K_habitat = 5
        c.greenhouse_m2 = 99999  # huge food
        c.solar_m2 = 99999  # huge power
        k = c.carrying_capacity()
        assert k < 50  # habitat constrains to ~5 + tech bonus


# ─── Resource consumption ───


class TestConsumeResources:
    """Resources are consumed proportionally. Shortage ratios in [0, 1]."""

    def test_full_ratios_when_abundant(self) -> None:
        c = create_colony("test", "balanced", 42)
        ratios = c._consume_resources()
        assert ratios["food"] == 1.0
        assert ratios["water"] == 1.0
        assert ratios["power"] == 1.0

    def test_zero_pop_returns_full(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.population = 0
        ratios = c._consume_resources()
        assert ratios["food"] == 1.0

    def test_starvation_ratio(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.food_kg = 1.0  # nearly empty
        ratios = c._consume_resources()
        assert 0.0 <= ratios["food"] < 1.0

    def test_resources_decrease_after_consumption(self) -> None:
        c = create_colony("test", "balanced", 42)
        food_before = c.food_kg
        c._consume_resources()
        assert c.food_kg < food_before


# ─── Births ───


class TestBirths:
    """IVF-assisted colony program. Births require population >= 2."""

    def test_zero_pop_no_births(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.population = 0
        ratios = {"food": 1.0, "water": 1.0, "power": 1.0}
        assert c._compute_births(ratios) == 0

    def test_one_pop_no_births(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.population = 1
        ratios = {"food": 1.0, "water": 1.0, "power": 1.0}
        assert c._compute_births(ratios) == 0

    def test_supply_ship_arrives(self) -> None:
        """At sol = SUPPLY_SHIP_INTERVAL, extra colonists arrive."""
        c = create_colony("test", "balanced", 42)
        c.sol = SUPPLY_SHIP_INTERVAL  # next tick will trigger ship
        ratios = {"food": 1.0, "water": 1.0, "power": 1.0}
        births = c._compute_births(ratios)
        ship_size = SUPPLY_SHIP_COLONISTS["balanced"]
        assert births >= ship_size  # at minimum the ship colonists

    def test_births_nonnegative(self) -> None:
        """Births can never be negative."""
        c = create_colony("test", "conservative", 42)
        for _ in range(50):
            ratios = {"food": 1.0, "water": 1.0, "power": 1.0}
            b = c._compute_births(ratios)
            assert b >= 0
            c.sol += 1


# ─── Deaths ───


class TestDeaths:
    """Deaths with cause attribution. First lethal cause wins."""

    def test_zero_pop_no_deaths(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.population = 0
        env = _make_env_snap()
        deaths, causes = c._compute_deaths(
            {"food": 1.0, "water": 1.0, "power": 1.0}, env
        )
        assert deaths == 0
        assert sum(causes.values()) == 0

    def test_deaths_bounded_by_population(self) -> None:
        """Can't die more than the population."""
        c = create_colony("test", "balanced", 42)
        c.population = 5
        env = _make_env_snap(storm="global")
        deaths, _ = c._compute_deaths(
            {"food": 0.0, "water": 0.0, "power": 0.0}, env
        )
        assert deaths <= 5

    def test_cause_sum_equals_total(self) -> None:
        """Sum of all causes = total deaths."""
        c = create_colony("test", "balanced", 42)
        env = _make_env_snap()
        deaths, causes = c._compute_deaths(
            {"food": 0.3, "water": 0.3, "power": 0.2}, env
        )
        assert sum(causes.values()) == deaths

    def test_starvation_causes_deaths(self) -> None:
        """No food → starvation deaths."""
        c = create_colony("test", "balanced", 42)
        env = _make_env_snap()
        _, causes = c._compute_deaths(
            {"food": 0.0, "water": 1.0, "power": 1.0}, env
        )
        # Over many colonists, starvation should appear
        # (probabilistic, but 0% food is severe)
        # Run multiple times for statistical confidence
        total_starvation = causes.get("starvation", 0)
        for _ in range(20):
            c2 = create_colony("test", "balanced", c.rng.randint(0, 99999))
            _, c2_causes = c2._compute_deaths(
                {"food": 0.0, "water": 1.0, "power": 1.0}, env
            )
            total_starvation += c2_causes.get("starvation", 0)
        assert total_starvation > 0, "Expected starvation deaths with 0% food"


# ─── Storm damage ───


class TestStormDamage:
    """Storms degrade solar panels and greenhouses."""

    def test_no_storm_no_damage(self) -> None:
        c = create_colony("test", "balanced", 42)
        solar_before = c.solar_m2
        c._storm_damage(_make_env_snap(storm=None))
        assert c.solar_m2 == solar_before

    def test_global_storm_damages_solar(self) -> None:
        c = create_colony("test", "balanced", 42)
        solar_before = c.solar_m2
        c._storm_damage(_make_env_snap(storm="global"))
        assert c.solar_m2 < solar_before

    def test_global_storm_damages_greenhouse(self) -> None:
        c = create_colony("test", "balanced", 42)
        gh_before = c.greenhouse_m2
        c._storm_damage(_make_env_snap(storm="global"))
        assert c.greenhouse_m2 < gh_before

    def test_regional_less_damage_than_global(self) -> None:
        c1 = create_colony("g", "balanced", 42)
        c2 = create_colony("r", "balanced", 42)
        c1._storm_damage(_make_env_snap(storm="global"))
        c2._storm_damage(_make_env_snap(storm="regional"))
        assert c1.solar_m2 < c2.solar_m2

    def test_floor_prevents_destruction(self) -> None:
        """Solar and greenhouse have minimum floors."""
        c = create_colony("test", "balanced", 42)
        for _ in range(5000):
            c._storm_damage(_make_env_snap(storm="global"))
        assert c.solar_m2 >= 50.0
        assert c.greenhouse_m2 >= 20.0


# ─── Morale ───


class TestMorale:
    """Morale drifts based on conditions. Always in [0, 1]."""

    def test_bounded(self) -> None:
        c = create_colony("test", "balanced", 42)
        env = _make_env_snap()
        for _ in range(500):
            c._update_morale({"food": 0.0, "water": 0.0, "power": 0.0}, env)
        assert 0.0 <= c.morale <= 1.0

    def test_good_conditions_raise_morale(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.morale = 0.3
        for _ in range(100):
            c._update_morale(
                {"food": 1.0, "water": 1.0, "power": 1.0},
                _make_env_snap(),
            )
        assert c.morale > 0.3

    def test_storms_lower_morale(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.morale = 0.8
        for _ in range(50):
            c._update_morale(
                {"food": 1.0, "water": 1.0, "power": 1.0},
                _make_env_snap(storm="global"),
            )
        c2 = create_colony("test2", "balanced", 42)
        c2.morale = 0.8
        for _ in range(50):
            c2._update_morale(
                {"food": 1.0, "water": 1.0, "power": 1.0},
                _make_env_snap(storm=None),
            )
        assert c.morale < c2.morale


# ─── Terraforming ───


class TestTerraforming:
    """Industrial greenhouse gas output for terraforming."""

    def test_zero_pop_no_output(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.population = 0
        delta = c._compute_terraforming()
        assert delta == 0.0

    def test_positive_output(self) -> None:
        c = create_colony("test", "balanced", 42)
        delta = c._compute_terraforming()
        assert delta > 0.0

    def test_aggressive_terraforms_faster(self) -> None:
        """Aggressive strategy has 1.5x modifier vs conservative's 0.7x."""
        con = create_colony("con", "conservative", 42)
        agg = create_colony("agg", "aggressive", 42)
        # Equalize everything so only strategy modifier differs
        for c in [con, agg]:
            c.population = 100
            c.power_kwh = 10000
            c.greenhouse_m2 = 1000
        d_con = con._compute_terraforming()
        d_agg = agg._compute_terraforming()
        assert d_agg > d_con

    def test_cumulative_tracking(self) -> None:
        c = create_colony("test", "balanced", 42)
        c._compute_terraforming()
        c._compute_terraforming()
        assert c.terraforming_output > 0


# ─── Genetic diversity ───


class TestGeneticDiversity:
    """Wright-Fisher drift. Small populations lose diversity."""

    def test_initial_diversity(self) -> None:
        c = create_colony("test", "balanced", 42)
        assert 0.0 < c.genetic_diversity <= 1.0

    def test_diversity_decreases_over_time(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.population = 20  # small pop = fast drift
        initial = c.genetic_diversity
        for sol in range(1, 301):
            c.sol = sol
            c._drift_genetic_diversity()
        assert c.genetic_diversity < initial

    def test_diversity_floor(self) -> None:
        """Diversity never drops below 0.05."""
        c = create_colony("test", "balanced", 42)
        c.population = 5
        for sol in range(1, 10000):
            c.sol = sol
            c._drift_genetic_diversity()
        assert c.genetic_diversity >= 0.05

    def test_immigrants_boost_diversity(self) -> None:
        c = create_colony("test", "balanced", 42)
        c.genetic_diversity = 0.3
        c.receive_immigrants(50)
        assert c.genetic_diversity > 0.3

    def test_zero_immigrants_no_change(self) -> None:
        c = create_colony("test", "balanced", 42)
        d_before = c.genetic_diversity
        c.receive_immigrants(0)
        assert c.genetic_diversity == d_before


# ─── Full tick integration ───


class TestColonyTick:
    """Colony.tick() — one sol, all subsystems fire."""

    def test_snapshot_keys(self) -> None:
        c = create_colony("test", "balanced", 42)
        env = _make_env_snap()
        snap = c.tick(env)
        required = {
            "sol", "population", "food_kg", "water_l", "power_kwh",
            "morale", "births", "deaths", "death_causes", "habitat_m2",
            "greenhouse_m2", "solar_m2", "cumulative_radiation_msv",
            "carrying_capacity", "genetic_diversity", "net_migration",
            "terraforming_contribution", "tech",
        }
        assert set(snap.keys()) == required

    def test_sol_increments(self) -> None:
        c = create_colony("test", "balanced", 42)
        for i in range(1, 6):
            snap = c.tick(_make_env_snap(sol=i))
            assert snap["sol"] == i

    def test_population_nonnegative(self) -> None:
        """Population can never go negative."""
        c = create_colony("test", "balanced", 42)
        env = MarsEnvironment(seed=42)
        for _ in range(500):
            snap = c.tick(env.tick())
            assert snap["population"] >= 0

    def test_morale_bounded_in_tick(self) -> None:
        c = create_colony("test", "balanced", 42)
        env = MarsEnvironment(seed=42)
        for _ in range(200):
            snap = c.tick(env.tick())
            assert 0.0 <= snap["morale"] <= 1.0

    def test_resources_nonnegative(self) -> None:
        c = create_colony("test", "aggressive", 42)
        env = MarsEnvironment(seed=42)
        for _ in range(200):
            snap = c.tick(env.tick())
            assert snap["food_kg"] >= 0
            assert snap["water_l"] >= 0
            assert snap["power_kwh"] >= 0

    def test_radiation_accumulates(self) -> None:
        c = create_colony("test", "balanced", 42)
        env = MarsEnvironment(seed=42)
        for _ in range(50):
            c.tick(env.tick())
        assert c.cumulative_radiation_msv > 0

    def test_history_grows(self) -> None:
        c = create_colony("test", "balanced", 42)
        for i in range(10):
            c.tick(_make_env_snap(sol=i + 1))
        assert len(c.history) == 10

    def test_deterministic(self) -> None:
        """Same seed = same outcome."""
        c1 = create_colony("a", "balanced", 42)
        c2 = create_colony("a", "balanced", 42)
        env1 = MarsEnvironment(seed=99)
        env2 = MarsEnvironment(seed=99)
        for _ in range(100):
            s1 = c1.tick(env1.tick())
            s2 = c2.tick(env2.tick())
            assert s1["population"] == s2["population"]
            assert s1["food_kg"] == s2["food_kg"]


# ─── Conservation law ───


class TestConservationLaw:
    """births - deaths = population change (no migration in solo colony)."""

    def test_population_accounting(self) -> None:
        c = create_colony("test", "balanced", 42)
        initial = c.population
        env = MarsEnvironment(seed=42)
        for _ in range(200):
            c.tick(env.tick())
        expected = initial + c.total_births - c.total_deaths
        assert c.population == max(0, expected)

    def test_death_causes_sum_to_total(self) -> None:
        """Cumulative death causes must equal total deaths."""
        c = create_colony("test", "balanced", 42)
        env = MarsEnvironment(seed=42)
        for _ in range(365):
            c.tick(env.tick())
        assert sum(c.cumulative_death_causes.values()) == c.total_deaths
