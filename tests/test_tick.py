"""
Unit tests for src/tick.py — The Abyss evolution engine.

531 lines of organism simulation with zero dedicated tests.
Pure functions, emergent behavior, physical invariants.

Run: python -m pytest tests/test_tick.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.tick import (
    clamp,
    dist,
    genome_dist,
    epoch_name,
    make_organism,
    make_grid,
    regrow,
    add_food_cluster,
    cluster_species,
    step,
    new_world,
    run_tick,
    WORLD_W,
    WORLD_H,
    GRID_W,
    GRID_H,
    GENE_COUNT,
    INITIAL_POP,
    MAX_POP,
    EPOCHS,
    PHEROMONE_DECAY,
)


# ─── Pure helper functions ───


class TestClamp:
    def test_within_range(self) -> None:
        assert clamp(0.5) == 0.5

    def test_below_min(self) -> None:
        assert clamp(-1.0) == 0.0

    def test_above_max(self) -> None:
        assert clamp(2.0) == 1.0

    def test_custom_bounds(self) -> None:
        assert clamp(15.0, 10.0, 20.0) == 15.0
        assert clamp(5.0, 10.0, 20.0) == 10.0
        assert clamp(25.0, 10.0, 20.0) == 20.0

    def test_edge_values(self) -> None:
        assert clamp(0.0) == 0.0
        assert clamp(1.0) == 1.0


class TestDist:
    def test_same_point(self) -> None:
        a = {"x": 100.0, "y": 200.0}
        assert dist(a, a) == 0.0

    def test_known_distance(self) -> None:
        a = {"x": 0.0, "y": 0.0}
        b = {"x": 3.0, "y": 4.0}
        assert abs(dist(a, b) - 5.0) < 1e-9

    def test_symmetry(self) -> None:
        a = {"x": 10.0, "y": 20.0}
        b = {"x": 30.0, "y": 50.0}
        assert abs(dist(a, b) - dist(b, a)) < 1e-9

    def test_nonnegative(self) -> None:
        rng = random.Random(42)
        for _ in range(100):
            a = {"x": rng.uniform(0, WORLD_W), "y": rng.uniform(0, WORLD_H)}
            b = {"x": rng.uniform(0, WORLD_W), "y": rng.uniform(0, WORLD_H)}
            assert dist(a, b) >= 0.0


class TestGenomeDist:
    def test_identical(self) -> None:
        g = [0.5] * GENE_COUNT
        assert genome_dist(g, g) == 0.0

    def test_known_value(self) -> None:
        g1 = [0.0] * GENE_COUNT
        g2 = [1.0] * GENE_COUNT
        expected = math.sqrt(GENE_COUNT)
        assert abs(genome_dist(g1, g2) - expected) < 1e-9

    def test_symmetry(self) -> None:
        rng = random.Random(42)
        g1 = [rng.random() for _ in range(GENE_COUNT)]
        g2 = [rng.random() for _ in range(GENE_COUNT)]
        assert abs(genome_dist(g1, g2) - genome_dist(g2, g1)) < 1e-9

    def test_triangle_inequality(self) -> None:
        rng = random.Random(42)
        g1 = [rng.random() for _ in range(GENE_COUNT)]
        g2 = [rng.random() for _ in range(GENE_COUNT)]
        g3 = [rng.random() for _ in range(GENE_COUNT)]
        assert genome_dist(g1, g3) <= genome_dist(g1, g2) + genome_dist(g2, g3) + 1e-9


class TestEpochName:
    def test_primordial(self) -> None:
        assert epoch_name(0) == "Primordial Soup"

    def test_first_sparks(self) -> None:
        assert epoch_name(50) == "First Sparks"

    def test_deep_time(self) -> None:
        assert epoch_name(5000) == "Deep Time"
        assert epoch_name(99999) == "Deep Time"

    def test_all_epochs_reachable(self) -> None:
        """Every epoch in the table can be reached."""
        reached = set()
        for threshold, name in EPOCHS:
            reached.add(epoch_name(threshold))
        assert len(reached) == len(EPOCHS)

    def test_monotonic_progression(self) -> None:
        """Epochs never go backward as tick increases."""
        prev_idx = 0
        for tick in range(0, 6000, 10):
            name = epoch_name(tick)
            idx = next(i for i, (_, n) in enumerate(EPOCHS) if n == name)
            assert idx >= prev_idx
            prev_idx = idx


# ─── Organism factory ───


class TestMakeOrganism:
    def test_has_required_keys(self) -> None:
        org = make_organism(0)
        required = {
            "id", "name", "x", "y", "vx", "vy", "energy",
            "genome", "age", "generation", "species_id",
            "born_tick", "parent_id",
        }
        assert set(org.keys()) == required

    def test_genome_length(self) -> None:
        org = make_organism(0)
        assert len(org["genome"]) == GENE_COUNT

    def test_genome_values_bounded(self) -> None:
        """All genes should be in [0, 1] when created from defaults."""
        for _ in range(50):
            org = make_organism(0)
            for gene in org["genome"]:
                assert 0.0 <= gene <= 1.0

    def test_custom_genome(self) -> None:
        g = [0.5] * GENE_COUNT
        org = make_organism(0, genome=g)
        assert org["genome"] == [0.5] * GENE_COUNT

    def test_position_in_world(self) -> None:
        for _ in range(100):
            org = make_organism(0)
            assert 0 <= org["x"] <= WORLD_W
            assert 0 <= org["y"] <= WORLD_H

    def test_custom_position(self) -> None:
        org = make_organism(0, x=100.0, y=200.0)
        assert org["x"] == 100.0
        assert org["y"] == 200.0

    def test_generation_tracking(self) -> None:
        org = make_organism(0, generation=5, parent_id="parent-123")
        assert org["generation"] == 5
        assert org["parent_id"] == "parent-123"

    def test_energy_positive(self) -> None:
        for _ in range(100):
            org = make_organism(0)
            assert org["energy"] > 0


# ─── Nutrient grid ───


class TestNutrientGrid:
    def test_grid_size(self) -> None:
        grid = make_grid()
        assert len(grid) == GRID_W * GRID_H

    def test_grid_values_bounded(self) -> None:
        grid = make_grid()
        for val in grid:
            assert 0.0 <= val <= 1.0

    def test_regrow_increases_nutrients(self) -> None:
        grid = [0.1] * (GRID_W * GRID_H)
        total_before = sum(grid)
        regrow(grid, 100)
        total_after = sum(grid)
        assert total_after > total_before

    def test_regrow_bounded(self) -> None:
        """Nutrients stay in [0, 1] after regrow."""
        grid = [0.99] * (GRID_W * GRID_H)
        for tick in range(100):
            regrow(grid, tick)
        for val in grid:
            assert 0.0 <= val <= 1.0

    def test_food_cluster_adds_nutrients(self) -> None:
        grid = [0.1] * (GRID_W * GRID_H)
        total_before = sum(grid)
        add_food_cluster(grid)
        total_after = sum(grid)
        assert total_after > total_before

    def test_food_cluster_bounded(self) -> None:
        grid = [0.8] * (GRID_W * GRID_H)
        add_food_cluster(grid)
        for val in grid:
            assert 0.0 <= val <= 1.0


# ─── Species clustering ───


class TestSpeciesClustering:
    def test_empty_organisms(self) -> None:
        assert cluster_species([], 0) == {}

    def test_single_organism(self) -> None:
        org = make_organism(0, genome=[0.5] * GENE_COUNT)
        species = cluster_species([org], 0)
        assert len(species) == 1
        assert org["id"] in list(species.values())[0]

    def test_identical_genomes_same_species(self) -> None:
        g = [0.5] * GENE_COUNT
        orgs = [make_organism(0, genome=list(g)) for _ in range(10)]
        species = cluster_species(orgs, 0)
        assert len(species) == 1

    def test_divergent_genomes_different_species(self) -> None:
        """Very different genomes should be in different species."""
        orgs = [
            make_organism(0, genome=[0.0] * GENE_COUNT),
            make_organism(0, genome=[1.0] * GENE_COUNT),
        ]
        species = cluster_species(orgs, 0)
        assert len(species) == 2

    def test_all_organisms_assigned(self) -> None:
        """Every organism gets a species_id."""
        orgs = [make_organism(0) for _ in range(20)]
        cluster_species(orgs, 0)
        for org in orgs:
            assert org["species_id"].startswith("sp-")


# ─── Simulation step ───


class TestStep:
    """step() is the core evolution function. Test its invariants."""

    def test_returns_list(self) -> None:
        orgs = [make_organism(0) for _ in range(10)]
        grid = make_grid()
        result = step(orgs, grid, 1, [])
        assert isinstance(result, list)

    def test_organisms_have_keys(self) -> None:
        orgs = [make_organism(0) for _ in range(5)]
        grid = make_grid()
        result = step(orgs, grid, 1, [])
        for org in result:
            assert "id" in org
            assert "energy" in org
            assert "genome" in org

    def test_population_bounded(self) -> None:
        """Population can't exceed MAX_POP."""
        orgs = [make_organism(0) for _ in range(50)]
        grid = make_grid()
        for tick in range(50):
            orgs = step(orgs, grid, tick, [])
            regrow(grid, tick)
        assert len(orgs) <= MAX_POP

    def test_energy_consumed(self) -> None:
        """Organisms lose energy via metabolism."""
        org = make_organism(0, genome=[0.5] * GENE_COUNT)
        initial_energy = org["energy"]
        grid = [0.0] * (GRID_W * GRID_H)  # no food
        result = step([org], grid, 1, [])
        if result:  # might have starved
            assert result[0]["energy"] < initial_energy

    def test_starvation_kills(self) -> None:
        """Organisms with no energy die."""
        org = make_organism(0)
        org["energy"] = 0.01  # nearly dead
        grid = [0.0] * (GRID_W * GRID_H)
        events: list[dict] = []
        result = step([org], grid, 1, events)
        # Should have starved
        death_events = [e for e in events if e["type"] == "death"]
        assert len(result) == 0 or len(death_events) > 0

    def test_boundary_wrap(self) -> None:
        """Organisms wrap around world boundaries."""
        org = make_organism(0, x=WORLD_W - 1, y=WORLD_H - 1)
        org["vx"] = 5.0
        org["vy"] = 5.0
        grid = make_grid()
        result = step([org], grid, 1, [])
        if result:
            assert 0 <= result[0]["x"] < WORLD_W
            assert 0 <= result[0]["y"] < WORLD_H

    def test_age_increments(self) -> None:
        org = make_organism(0)
        assert org["age"] == 0
        grid = make_grid()
        result = step([org], grid, 1, [])
        if result:
            assert result[0]["age"] == 1

    def test_events_recorded(self) -> None:
        """Birth and death events should be recorded."""
        orgs = [make_organism(0) for _ in range(30)]
        # Give some lots of energy to trigger reproduction
        for o in orgs[:10]:
            o["energy"] = 200.0
        grid = make_grid()
        events: list[dict] = []
        step(orgs, grid, 1, events)
        event_types = {e["type"] for e in events}
        assert len(events) > 0  # something should happen with 30 organisms


# ─── World creation ───


class TestNewWorld:
    def test_has_required_keys(self) -> None:
        world = new_world()
        required = {
            "_meta", "tick", "organisms", "nutrients",
            "species", "history", "events", "pheromones",
        }
        assert set(world.keys()) == required

    def test_initial_tick_zero(self) -> None:
        world = new_world()
        assert world["tick"] == 0

    def test_initial_population(self) -> None:
        world = new_world()
        assert len(world["organisms"]) == INITIAL_POP

    def test_grid_size(self) -> None:
        world = new_world()
        assert len(world["nutrients"]) == GRID_W * GRID_H

    def test_pheromone_grid_size(self) -> None:
        world = new_world()
        assert len(world["pheromones"]) == GRID_W * GRID_H
        assert all(p == 0.0 for p in world["pheromones"])

    def test_history_initialized(self) -> None:
        world = new_world()
        assert world["history"]["population"] == [INITIAL_POP]
        assert len(world["history"]["species_count"]) == 1

    def test_genesis_event(self) -> None:
        world = new_world()
        assert any(e["type"] == "genesis" for e in world["events"])


# ─── run_tick integration ───


class TestRunTick:
    def test_tick_increments(self) -> None:
        world = new_world()
        world = run_tick(world)
        assert world["tick"] == 1

    def test_smoke_10_ticks(self) -> None:
        """10 ticks without crash — the minimum bar."""
        world = new_world()
        for _ in range(10):
            world = run_tick(world)
        assert world["tick"] == 10
        assert len(world["organisms"]) > 0

    def test_population_tracked(self) -> None:
        world = new_world()
        for _ in range(5):
            world = run_tick(world)
        assert len(world["history"]["population"]) == 6  # initial + 5 ticks

    def test_epoch_updates(self) -> None:
        world = new_world()
        assert world["_meta"]["epoch"] == "Primordial Soup"
        # Fast forward
        world["tick"] = 49
        world = run_tick(world)
        assert world["_meta"]["epoch"] == "First Sparks"

    def test_extinction_rescue(self) -> None:
        """If population drops below 5, new organisms are seeded."""
        world = new_world()
        world["organisms"] = world["organisms"][:2]  # nearly extinct
        world = run_tick(world)
        assert len(world["organisms"]) >= 5

    def test_nutrients_stay_bounded(self) -> None:
        world = new_world()
        for _ in range(20):
            world = run_tick(world)
        for val in world["nutrients"]:
            assert 0.0 <= val <= 1.0

    def test_pheromone_decay(self) -> None:
        """Pheromones should decay over time."""
        world = new_world()
        # Inject a strong pheromone signal
        world["pheromones"][0] = 1.0
        world = run_tick(world)
        # Should have decayed
        assert world["pheromones"][0] < 1.0

    def test_deterministic_with_seed(self) -> None:
        """Same random seed = same evolution."""
        random.seed(42)
        w1 = new_world()
        for _ in range(5):
            w1 = run_tick(w1)

        random.seed(42)
        w2 = new_world()
        for _ in range(5):
            w2 = run_tick(w2)

        assert len(w1["organisms"]) == len(w2["organisms"])
        assert w1["tick"] == w2["tick"]

    def test_history_capped(self) -> None:
        """History doesn't grow unbounded."""
        world = new_world()
        world["history"]["population"] = list(range(700))
        world = run_tick(world)
        assert len(world["history"]["population"]) <= 601  # HISTORY_CAP + 1


# ─── Physical invariants ───


class TestInvariants:
    """Properties that must hold regardless of simulation state."""

    def test_organisms_in_world_bounds(self) -> None:
        """After any tick, all organisms are within world bounds."""
        world = new_world()
        for _ in range(20):
            world = run_tick(world)
        for org in world["organisms"]:
            assert 0 <= org["x"] < WORLD_W + 10  # small margin for float
            assert 0 <= org["y"] < WORLD_H + 10

    def test_energy_survivors_positive(self) -> None:
        """All surviving organisms have positive energy."""
        world = new_world()
        for _ in range(20):
            world = run_tick(world)
        for org in world["organisms"]:
            assert org["energy"] > 0

    def test_genome_length_preserved(self) -> None:
        """Genomes maintain GENE_COUNT length through reproduction."""
        world = new_world()
        for _ in range(30):
            world = run_tick(world)
        for org in world["organisms"]:
            assert len(org["genome"]) == GENE_COUNT

    def test_species_count_nonnegative(self) -> None:
        world = new_world()
        for _ in range(10):
            world = run_tick(world)
        assert len(world["species"]) >= 0

    def test_50_tick_survival(self) -> None:
        """The ecosystem survives 50 ticks without collapsing to zero."""
        world = new_world()
        for _ in range(50):
            world = run_tick(world)
        # Extinction rescue should keep pop above 0
        assert len(world["organisms"]) > 0
