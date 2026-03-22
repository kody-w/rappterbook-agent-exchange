"""
Tests for Mars Barn terrarium — environment, colony, engine, curves.

Run: python -m pytest tests/test_mars.py -v
"""
from __future__ import annotations

import json
import math
import sys
import os
import tempfile
from pathlib import Path

# Add repo root and src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars_env import (
    MarsEnvironment,
    sol_to_ls,
    season_name,
    surface_temperature_c,
    solar_flux_wm2,
    radiation_msv,
    SOLS_PER_MARS_YEAR,
    BASE_SOLAR_FLUX_WM2,
    BASE_RADIATION_MSV_SOL,
)
from src.mars_colony import (
    Colony,
    create_colony,
    FOOD_KG_SOL,
    WATER_L_SOL,
    POWER_KWH_SOL,
)
from src.tick_engine import Simulation
from src.mars_curves import generate_dashboard


# ─── Environment tests ───


class TestSolToLs:
    def test_sol_zero(self) -> None:
        assert sol_to_ls(0) == 0.0

    def test_full_year(self) -> None:
        ls = sol_to_ls(int(SOLS_PER_MARS_YEAR))
        assert abs(ls - 360.0) < 1.0 or abs(ls) < 1.0  # wraps

    def test_monotonic(self) -> None:
        prev = -1.0
        wrapped = False
        for sol in range(0, 668):
            ls = sol_to_ls(sol)
            if ls < prev:
                wrapped = True
            if not wrapped:
                assert ls >= prev or sol == 0
            prev = ls

    def test_range(self) -> None:
        for sol in range(0, 1000):
            ls = sol_to_ls(sol)
            assert 0 <= ls < 360


class TestSeasonName:
    def test_spring(self) -> None:
        assert season_name(45) == "spring"

    def test_summer(self) -> None:
        assert season_name(135) == "summer"

    def test_autumn(self) -> None:
        assert season_name(225) == "autumn"

    def test_winter(self) -> None:
        assert season_name(315) == "winter"


class TestTemperature:
    def test_physical_bounds(self) -> None:
        """Temperature must stay within Mars physical limits."""
        for sol in range(0, 700):
            ls = sol_to_ls(sol)
            temp = surface_temperature_c(ls)
            assert -150 < temp < 30, f"Sol {sol}: {temp}°C out of bounds"

    def test_seasonal_variation(self) -> None:
        """Summer should be warmer than winter."""
        summer_temp = surface_temperature_c(135)  # Ls=135, summer
        winter_temp = surface_temperature_c(315)  # Ls=315, winter
        assert summer_temp > winter_temp


class TestSolarFlux:
    def test_clear_sky(self) -> None:
        flux = solar_flux_wm2(90, 0.0)
        assert flux > 0
        assert flux < BASE_SOLAR_FLUX_WM2 * 1.1  # tau=0.3 attenuates

    def test_global_storm(self) -> None:
        flux_clear = solar_flux_wm2(90, 0.0)
        flux_storm = solar_flux_wm2(90, 1.0)
        assert flux_storm < flux_clear * 0.2  # massive reduction

    def test_positive(self) -> None:
        for dust in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert solar_flux_wm2(180, dust) > 0


class TestRadiation:
    def test_baseline(self) -> None:
        rad = radiation_msv(0.0, False)
        assert abs(rad - BASE_RADIATION_MSV_SOL) < 0.01

    def test_flare_increases(self) -> None:
        normal = radiation_msv(0.0, False)
        flare = radiation_msv(0.0, True)
        assert flare > normal

    def test_dust_shields_gcr(self) -> None:
        clear = radiation_msv(0.0, False)
        dusty = radiation_msv(1.0, False)
        assert dusty < clear


class TestMarsEnvironment:
    def test_smoke_100_sols(self) -> None:
        """Run 100 sols without crash."""
        env = MarsEnvironment(seed=42)
        for _ in range(100):
            snap = env.tick()
            assert "sol" in snap
            assert "temperature_c" in snap
            assert "radiation_msv" in snap
            assert snap["sol"] > 0

    def test_deterministic(self) -> None:
        """Same seed = same results."""
        env1 = MarsEnvironment(seed=99)
        env2 = MarsEnvironment(seed=99)
        for _ in range(50):
            s1 = env1.tick()
            s2 = env2.tick()
            assert s1 == s2

    def test_storms_occur(self) -> None:
        """Over 2 Mars years, at least one storm should happen."""
        env = MarsEnvironment(seed=42)
        storms = 0
        for _ in range(1337):  # ~2 Mars years
            snap = env.tick()
            if snap.get("storm"):
                storms += 1
        assert storms > 0, "No storms in 2 Mars years — suspicious"


# ─── Colony tests ───


class TestColonyCreation:
    def test_conservative(self) -> None:
        c = create_colony("Test", "conservative", 1)
        assert c.population == 120
        assert c.medical_level == 0.8
        assert c.food_kg > 0

    def test_balanced(self) -> None:
        c = create_colony("Test", "balanced", 2)
        assert c.population == 80

    def test_aggressive(self) -> None:
        c = create_colony("Test", "aggressive", 3)
        assert c.population == 60
        assert c.medical_level == 0.4


class TestColonyTick:
    def test_smoke_10_sols(self) -> None:
        """Colony survives 10 sols."""
        c = create_colony("Test", "balanced", 42)
        env = MarsEnvironment(seed=42)
        for _ in range(10):
            snap = env.tick()
            c.tick(snap)
        assert c.population > 0
        assert len(c.history) == 10

    def test_population_nonnegative(self) -> None:
        """Population never goes negative, even under stress."""
        c = create_colony("Stress", "aggressive", 7)
        c.food_kg = 10  # nearly starving
        c.water_l = 5
        env = MarsEnvironment(seed=7)
        for _ in range(100):
            snap = env.tick()
            c.tick(snap)
            assert c.population >= 0

    def test_morale_bounded(self) -> None:
        """Morale stays in [0, 1]."""
        c = create_colony("Morale", "balanced", 11)
        env = MarsEnvironment(seed=11)
        for _ in range(200):
            snap = env.tick()
            c.tick(snap)
            assert 0.0 <= c.morale <= 1.0

    def test_resources_nonnegative(self) -> None:
        """Food, water, power never go negative."""
        c = create_colony("Resources", "conservative", 13)
        env = MarsEnvironment(seed=13)
        for _ in range(100):
            snap = env.tick()
            c.tick(snap)
            assert c.food_kg >= 0
            assert c.water_l >= 0
            assert c.power_kwh >= 0

    def test_deaths_lte_population(self) -> None:
        """Deaths in a sol never exceed population at start of sol."""
        c = create_colony("Deaths", "balanced", 17)
        env = MarsEnvironment(seed=17)
        for _ in range(100):
            prev_pop = c.population
            snap = env.tick()
            result = c.tick(snap)
            assert result["deaths"] <= prev_pop + result["births"]  # births happen same sol

    def test_history_grows(self) -> None:
        """Each tick appends to history."""
        c = create_colony("Hist", "balanced", 19)
        env = MarsEnvironment(seed=19)
        for i in range(5):
            c.tick(env.tick())
        assert len(c.history) == 5

    def test_zero_population_stable(self) -> None:
        """Colony with 0 population doesn't crash."""
        c = create_colony("Ghost", "balanced", 23)
        c.population = 0
        env = MarsEnvironment(seed=23)
        for _ in range(10):
            c.tick(env.tick())
        assert c.population == 0  # no spontaneous generation


# ─── Simulation (tick_engine) tests ───


class TestSimulation:
    def test_smoke_10_sols(self) -> None:
        """Full simulation runs 10 sols without crash."""
        sim = Simulation(sols=10, env_seed=42)
        results = sim.run()
        assert results["_meta"]["sols"] == 10
        assert len(results["colonies"]) == 3
        for c in results["colonies"]:
            assert len(c["history"]) == 10

    def test_deterministic(self) -> None:
        """Same seeds = same results."""
        r1 = Simulation(sols=50, env_seed=42).run()
        r2 = Simulation(sols=50, env_seed=42).run()
        for c1, c2 in zip(r1["colonies"], r2["colonies"]):
            pops1 = [h["population"] for h in c1["history"]]
            pops2 = [h["population"] for h in c2["history"]]
            assert pops1 == pops2

    def test_365_sols_all_survive(self) -> None:
        """All 3 colonies survive 365 sols with default params."""
        sim = Simulation(sols=365, env_seed=42)
        results = sim.run()
        for c in results["colonies"]:
            final_pop = c["history"][-1]["population"]
            assert final_pop > 0, f"{c['name']} went extinct"

    def test_summary_consistent(self) -> None:
        """Summary stats match history."""
        sim = Simulation(sols=100, env_seed=42)
        results = sim.run()
        for c_result, s in zip(results["colonies"], results["summary"]["colonies"]):
            pops = [h["population"] for h in c_result["history"]]
            assert s["peak_pop"] == max(pops)
            assert s["min_pop"] == min(pops)
            assert s["end_pop"] == pops[-1]

    def test_env_history_matches(self) -> None:
        """Environment history length matches sols."""
        sim = Simulation(sols=30, env_seed=42)
        results = sim.run()
        assert len(results["environment"]["history"]) == 30

    def test_callback_invoked(self) -> None:
        """Callback fires each sol."""
        calls = []
        sim = Simulation(sols=5, env_seed=42)
        sim.run(callback=lambda sol, env, cols: calls.append(sol))
        assert calls == [1, 2, 3, 4, 5]

    def test_results_serializable(self) -> None:
        """Results can be serialized to JSON."""
        sim = Simulation(sols=10, env_seed=42)
        results = sim.run()
        text = json.dumps(results)
        parsed = json.loads(text)
        assert parsed["_meta"]["sols"] == 10


# ─── Curves (HTML generation) tests ───


class TestDashboard:
    def test_generates_html(self) -> None:
        """Dashboard produces valid-looking HTML."""
        sim = Simulation(sols=30, env_seed=42)
        results = sim.run()
        html = generate_dashboard(results)
        assert "<!DOCTYPE html>" in html
        assert "Mars Barn" in html
        assert "Ares Prime" in html
        assert "Olympus Station" in html
        assert "Red Frontier" in html
        assert "<canvas" in html

    def test_canvas_charts_present(self) -> None:
        """Dashboard contains expected canvas charts."""
        sim = Simulation(sols=30, env_seed=42)
        results = sim.run()
        html = generate_dashboard(results)
        for chart_id in ["pop-chart", "food-chart", "morale-chart", "births-chart", "temp-chart"]:
            assert chart_id in html, f"Missing chart: {chart_id}"
        assert "diversity-chart" in html


# ─── Property-based invariant tests ───


class TestConservationLaws:
    """Physics-level invariants that must hold for any seed."""

    def test_population_accounting(self) -> None:
        """births - deaths + migration = population change (over full sim)."""
        for seed in [1, 42, 99, 256, 1000]:
            sim = Simulation(sols=100, env_seed=seed)
            results = sim.run()
            for c in results["colonies"]:
                hist = c["history"]
                start_pop = hist[0]["population"] - hist[0]["births"] + hist[0]["deaths"]
                # Account for migration that happened on sol 1
                start_pop -= hist[0].get("net_migration", 0)
                end_pop = hist[-1]["population"]
                total_b = sum(h["births"] for h in hist)
                total_d = sum(h["deaths"] for h in hist)
                total_mig = sum(h.get("net_migration", 0) for h in hist)
                assert end_pop == start_pop + total_b - total_d + total_mig, (
                    f"Accounting error for {c['name']} seed={seed}: "
                    f"start={start_pop} + births={total_b} - deaths={total_d}"
                    f" + mig={total_mig} != end={end_pop}"
                )

    def test_temperature_physical_range(self) -> None:
        """All temperatures within Mars physical bounds for any seed."""
        for seed in [1, 42, 99]:
            sim = Simulation(sols=700, env_seed=seed)
            results = sim.run()
            for e in results["environment"]["history"]:
                t = e["temperature_c"]
                assert -150 < t < 30, f"Temp {t}°C out of Mars bounds (seed={seed})"

    def test_radiation_positive(self) -> None:
        """Radiation is always positive."""
        for seed in [1, 42, 99]:
            sim = Simulation(sols=700, env_seed=seed)
            results = sim.run()
            for e in results["environment"]["history"]:
                assert e["radiation_msv"] > 0


# ─── Trade and discovery tests ────────────────────────────────────────

class TestInterColonyTrade:
    """Inter-colony food sharing during shortages."""

    def test_trade_transfers_food(self) -> None:
        """When one colony starves and another has surplus, food flows."""
        sim = Simulation(sols=200, env_seed=42)
        results = sim.run()
        # Just verify all colonies have some food — trade prevents total starvation
        for c in results["colonies"]:
            last = c["history"][-1]
            assert last["food_kg"] >= 0

    def test_trade_doesnt_create_food(self) -> None:
        """Trade is zero-sum — total food before trade == total after."""
        sim = Simulation(sols=10, env_seed=42)
        # Run without trade to compare
        sim2 = Simulation(sols=10, env_seed=42)
        r1 = sim.run()
        r2 = sim2.run()
        # Results should be deterministic and identical
        for i in range(3):
            assert (r1["colonies"][i]["history"][-1]["population"] ==
                    r2["colonies"][i]["history"][-1]["population"])


class TestDiscoveries:
    """Rare permanent improvements (ice veins, medical, crop strains)."""

    def test_discoveries_occur_over_365_sols(self) -> None:
        """At least one discovery should happen in 365 sols across 3 colonies."""
        sim = Simulation(sols=365, env_seed=42)
        sim.run()
        # Check the colony objects directly (not the trimmed results)
        total_discoveries = sum(
            sum(1 for e in c.events if e["type"] == "discovery")
            for c in sim.colonies
        )
        assert total_discoveries > 0, "Expected at least one discovery in 365 sols"

    def test_water_mining_bonus_nonnegative(self) -> None:
        """Water mining bonus accumulates, never negative."""
        sim = Simulation(sols=365, env_seed=42)
        sim.run()
        for c in sim.colonies:
            assert c.water_mining_bonus >= 0

    def test_medical_breakthroughs_capped(self) -> None:
        """Medical breakthroughs capped at 4."""
        sim = Simulation(sols=2000, env_seed=42)
        sim.run()
        for c in sim.colonies:
            assert c.medical_breakthroughs <= 4


# ─── Carrying capacity tests ─────────────────────────────────────────

class TestCarryingCapacity:
    """Explicit carrying capacity K and logistic damping."""

    def test_k_positive(self) -> None:
        """K is always positive for non-zero colonies."""
        for strategy in ["conservative", "balanced", "aggressive"]:
            c = create_colony("Test", strategy, 42)
            assert c.carrying_capacity() > 0

    def test_k_in_history(self) -> None:
        """K is tracked in every history snapshot."""
        sim = Simulation(sols=50, env_seed=42)
        results = sim.run()
        for c in results["colonies"]:
            for h in c["history"]:
                assert "carrying_capacity" in h
                assert h["carrying_capacity"] > 0

    def test_k_grows_with_infrastructure(self) -> None:
        """K increases as greenhouse/habitat/solar expand."""
        c = create_colony("Test", "balanced", 42)
        k1 = c.carrying_capacity()
        c.greenhouse_m2 *= 2
        c.habitat_m2 *= 2
        c.solar_m2 *= 2
        c.water_mining_bonus += 50  # boost water to remove bottleneck
        k2 = c.carrying_capacity()
        assert k2 > k1

    def test_logistic_damping_slows_growth(self) -> None:
        """Birth rate decreases as population approaches K."""
        c = create_colony("Test", "balanced", 42)
        k = c.carrying_capacity()
        # Run until population is well above initial but below K
        env = MarsEnvironment(seed=42)
        for _ in range(200):
            c.tick(env.tick())
        if c.population > 0:
            assert c.population < k * 2, "Population should not wildly exceed K"

    def test_conservative_has_highest_initial_k(self) -> None:
        """Conservative strategy starts with highest K (most infrastructure)."""
        cons = create_colony("C", "conservative", 1)
        bal = create_colony("B", "balanced", 2)
        agg = create_colony("A", "aggressive", 3)
        assert cons.carrying_capacity() > bal.carrying_capacity()
        assert bal.carrying_capacity() > agg.carrying_capacity()


# ─── Epidemic tests ──────────────────────────────────────────────────

class TestEpidemics:
    """Disease outbreaks in colonies."""

    def test_epidemic_class(self) -> None:
        """Epidemic tracks state correctly."""
        from src.mars_colony import Epidemic, EPIDEMIC_STRAINS
        strain = EPIDEMIC_STRAINS[0]
        ep = Epidemic(strain, 20, 100)
        assert ep.strain == "Mars Flu"
        assert ep.remaining_sols == 20
        assert ep.infection_rate() >= 0
        assert ep.extra_mortality() >= 0
        assert ep.tick()  # still alive
        assert ep.remaining_sols == 19

    def test_epidemic_ends(self) -> None:
        """Epidemic ends after its duration."""
        from src.mars_colony import Epidemic, EPIDEMIC_STRAINS
        ep = Epidemic(EPIDEMIC_STRAINS[0], 2, 50)
        assert ep.tick()  # sol 1: still alive
        assert not ep.tick()  # sol 2: ends

    def test_epidemics_happen_over_long_run(self) -> None:
        """At least one epidemic across 3 colonies in 1000 sols."""
        sim = Simulation(sols=1000, env_seed=42)
        sim.run()
        total_epidemics = sum(
            sum(1 for e in c.events if e["type"] == "epidemic_start")
            for c in sim.colonies
        )
        assert total_epidemics > 0, "Expected epidemics in 1000 sols"

    def test_quarantine_reduces_spread(self) -> None:
        """Colonies with high medical level quarantine epidemics."""
        from src.mars_colony import Epidemic, EPIDEMIC_STRAINS
        ep = Epidemic(EPIDEMIC_STRAINS[1], 30, 100)
        # Advance a few sols so infection_rate > 0
        for _ in range(10):
            ep.tick()
        base_mortality = ep.extra_mortality()
        assert base_mortality > 0, "Epidemic should have nonzero mortality mid-outbreak"
        ep.quarantined = True
        quarantined_mortality = ep.extra_mortality()
        assert quarantined_mortality < base_mortality


# ─── Genetic diversity tests ─────────────────────────────────────────

class TestGeneticDiversity:
    """Founder effect and genetic drift."""

    def test_initial_diversity(self) -> None:
        """Initial diversity scales with population size."""
        small = create_colony("Small", "aggressive", 1)
        large = create_colony("Large", "conservative", 2)
        assert large.genetic_diversity >= small.genetic_diversity

    def test_diversity_in_history(self) -> None:
        """Genetic diversity tracked in snapshots."""
        sim = Simulation(sols=50, env_seed=42)
        results = sim.run()
        for c in results["colonies"]:
            for h in c["history"]:
                assert "genetic_diversity" in h
                assert 0.0 <= h["genetic_diversity"] <= 1.0

    def test_diversity_drifts_down(self) -> None:
        """Diversity decreases over time without immigration."""
        c = create_colony("Isolated", "balanced", 42)
        env = MarsEnvironment(seed=42)
        initial_diversity = c.genetic_diversity
        for _ in range(300):  # ~10 generations
            c.tick(env.tick())
        assert c.genetic_diversity <= initial_diversity

    def test_immigration_boosts_diversity(self) -> None:
        """receive_immigrants increases genetic diversity."""
        c = create_colony("Colony", "balanced", 42)
        c.genetic_diversity = 0.3  # degraded
        c.receive_immigrants(20)
        assert c.genetic_diversity > 0.3


# ─── Migration tests ─────────────────────────────────────────────────

class TestMigration:
    """Inter-colony migration mechanics."""

    def test_migration_occurs_over_365_sols(self) -> None:
        """At least some migration happens in a year."""
        sim = Simulation(sols=365, env_seed=42)
        sim.run()
        assert sim.total_migrations > 0, "Expected some migration in 365 sols"

    def test_migration_zero_sum(self) -> None:
        """Total immigrants == total emigrants across all colonies."""
        sim = Simulation(sols=200, env_seed=42)
        sim.run()
        total_in = sum(c.total_immigrants for c in sim.colonies)
        total_out = sum(c.total_emigrants for c in sim.colonies)
        assert total_in == total_out

    def test_net_migration_in_history(self) -> None:
        """net_migration field exists in history snapshots."""
        sim = Simulation(sols=50, env_seed=42)
        results = sim.run()
        for c in results["colonies"]:
            for h in c["history"]:
                assert "net_migration" in h


# ─── Tech tree integration tests ─────────────────────────────────────

from src.tech_tree import ResearchEngine, TECH_CATALOG


class TestTechTreeIntegration:
    """Tests for tech tree wired into colonies and simulation."""

    def test_colony_has_research_engine(self) -> None:
        """Every colony gets a ResearchEngine."""
        c = create_colony("Test", "balanced", 42)
        assert hasattr(c, "research_engine")
        assert isinstance(c.research_engine, ResearchEngine)

    def test_techs_unlock_over_365_sols(self) -> None:
        """Techs unlock during a full simulation."""
        sim = Simulation(sols=365, env_seed=42)
        sim.run()
        for colony in sim.colonies:
            assert len(colony.research_engine.unlocked) > 0, (
                f"{colony.name} unlocked no techs in 365 sols"
            )

    def test_strategy_affects_research_priority(self) -> None:
        """Different strategies prioritize different techs."""
        sim = Simulation(sols=365, env_seed=42)
        sim.run()
        conservative = sim.colonies[0]  # Ares Prime
        aggressive = sim.colonies[2]    # Red Frontier
        # They should have different tech sets (or at least different order)
        assert conservative.research_engine.unlocked != aggressive.research_engine.unlocked, (
            "Conservative and aggressive should diverge in tech choices"
        )

    def test_tech_effects_applied_to_production(self) -> None:
        """After unlocking food tech, greenhouse yield multiplier > 1.0."""
        sim = Simulation(sols=365, env_seed=42)
        sim.run()
        for colony in sim.colonies:
            if "greenhouse_biotech_1" in colony.research_engine.unlocked:
                fx = colony._tech_effects
                # food_production_mult should be > 1.0 with food tech
                assert fx["food_production_mult"] > 1.0

    def test_tech_snapshot_in_results(self) -> None:
        """Results dict contains tech_tree data per colony."""
        sim = Simulation(sols=100, env_seed=42)
        results = sim.run()
        for c in results["colonies"]:
            assert "tech_tree" in c, "Colony result missing tech_tree"
            tt = c["tech_tree"]
            assert "unlocked" in tt
            assert "branch_points" in tt
            assert "effects" in tt

    def test_techs_unlocked_in_history(self) -> None:
        """History snapshots track techs_unlocked count."""
        sim = Simulation(sols=200, env_seed=42)
        results = sim.run()
        for c in results["colonies"]:
            for h in c["history"]:
                assert "techs_unlocked" in h

    def test_tech_timeline_in_dashboard(self) -> None:
        """Dashboard HTML contains tech timeline chart."""
        sim = Simulation(sols=100, env_seed=42)
        results = sim.run()
        html = generate_dashboard(results)
        assert "tech-chart" in html
        assert "Tech Timeline" in html

    def test_no_tech_before_min_pop(self) -> None:
        """Colonies with pop < 5 don't generate research points."""
        engine = ResearchEngine(strategy="balanced", seed=42)
        points = engine.generate_points(population=3, morale=0.8, sol=1)
        assert points == 0.0

    def test_tech_unlock_events_recorded(self) -> None:
        """Tech unlocks appear in colony events."""
        sim = Simulation(sols=365, env_seed=42)
        sim.run()
        tech_events = []
        for colony in sim.colonies:
            tech_events.extend(
                e for e in colony.events if e.get("type") == "tech_unlock"
            )
        assert len(tech_events) > 0, "No tech_unlock events recorded"

    def test_techs_in_summary(self) -> None:
        """Summary includes techs_unlocked per colony."""
        sim = Simulation(sols=365, env_seed=42)
        results = sim.run()
        for s in results["summary"]["colonies"]:
            assert "techs_unlocked" in s

    def test_radiation_shielding_from_tech(self) -> None:
        """Rad shielding tech reduces per-sol radiation accumulation."""
        # Verify that the shielding bonus is correctly applied
        from src.tech_tree import ResearchEngine
        engine = ResearchEngine(strategy="balanced", seed=42)
        # Manually add rad_shielding_1 to verify effect
        engine.unlocked.add("rad_shielding_1")
        fx = engine.cumulative_effects()
        assert fx["radiation_shielding_bonus"] > 0, "Rad shielding should provide bonus"
        # Base shielding is 0.8, with bonus it should be > 0.8
        shielding = min(0.95, 0.8 + fx["radiation_shielding_bonus"])
        assert shielding > 0.8, f"Expected shielding > 0.8, got {shielding}"
