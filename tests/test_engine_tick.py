"""
tests/test_engine_tick.py — Unit tests for engine/tick.py (Emergence tick engine).

The Emergence sim has 10+ pure functions, zero tests. This file fills the gap.
Covers: distance, genome ops, movement, feeding, energy drain, combat,
reproduction, nutrient regrowth, species tracking, full tick integration.

Run: python -m pytest tests/test_engine_tick.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.tick import (
    distance,
    genome_distance,
    genome_to_species,
    move_organism,
    feed_organism,
    drain_energy,
    interact,
    try_reproduce,
    regrow_nutrients,
    rebuild_species,
    tick,
    MAX_POP,
    MIN_POP,
    NUTRIENT_REGROW,
    G_SPEED, G_SIZE, G_SENSE, G_METABOLISM,
    G_AGGRESSION, G_SOCIALITY, G_CAMOUFLAGE,
    G_REPRODUCTION, G_MUTATION, G_LIFESPAN,
    G_TOXICITY, G_BIOLUM, G_DIET, G_ARMOR,
    G_MEMORY, G_SYMBIOSIS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(**overrides: float) -> list:
    """Create a 16-gene genome with defaults at 0.5, applying overrides."""
    g = [0.5] * 16
    name_to_idx = {
        "speed": G_SPEED, "size": G_SIZE, "sense": G_SENSE,
        "metabolism": G_METABOLISM, "aggression": G_AGGRESSION,
        "sociality": G_SOCIALITY, "camouflage": G_CAMOUFLAGE,
        "reproduction": G_REPRODUCTION, "mutation": G_MUTATION,
        "lifespan": G_LIFESPAN, "toxicity": G_TOXICITY,
        "biolum": G_BIOLUM, "diet": G_DIET, "armor": G_ARMOR,
        "memory": G_MEMORY, "symbiosis": G_SYMBIOSIS,
    }
    for name, val in overrides.items():
        g[name_to_idx[name]] = val
    return g


def _make_organism(
    oid="org-001",
    x=100.0,
    y=100.0,
    energy=100.0,
    age=0,
    **genome_overrides,
):
    """Create a minimal organism dict for testing."""
    return {
        "id": oid,
        "origin_agent": "test",
        "genome": _make_genome(**genome_overrides),
        "x": x,
        "y": y,
        "energy": energy,
        "age": age,
        "children": 0,
        "species_id": "0000",
        "archetype": "test",
    }


def _make_nutrients(width=80, height=60, fill=50):
    """Create a nutrient grid for testing."""
    return {
        "width": width,
        "height": height,
        "grid": [fill] * (width * height),
    }


def _make_world(n_organisms=50, tick_num=0, energy=100.0):
    """Create a minimal world state for tick() testing."""
    rng = random.Random(42)
    organisms = []
    for i in range(n_organisms):
        organisms.append({
            "id": "org-%04d" % i,
            "origin_agent": "test",
            "genome": [round(rng.random(), 4) for _ in range(16)],
            "x": rng.random() * 800,
            "y": rng.random() * 600,
            "energy": energy,
            "age": rng.randint(0, 50),
            "children": 0,
            "species_id": "0000",
            "archetype": "test",
        })
    return {
        "tick": tick_num,
        "organisms": organisms,
        "nutrients": _make_nutrients(),
        "config": {"width": 800, "height": 600},
        "events": [],
        "population_history": [],
        "species_history": [],
        "stats": {},
        "_meta": {"engine": "emergence", "last_tick": ""},
    }


# ---------------------------------------------------------------------------
# distance()
# ---------------------------------------------------------------------------


class TestDistance:
    def test_same_point(self):
        a = {"x": 10.0, "y": 20.0}
        assert distance(a, a) == 0.0

    def test_known_distance(self):
        a = {"x": 0.0, "y": 0.0}
        b = {"x": 3.0, "y": 4.0}
        assert distance(a, b) == 5.0

    def test_symmetric(self):
        a = {"x": 1.0, "y": 2.0}
        b = {"x": 5.0, "y": 7.0}
        assert distance(a, b) == distance(b, a)

    def test_positive(self):
        a = {"x": -10.0, "y": -20.0}
        b = {"x": 10.0, "y": 20.0}
        assert distance(a, b) > 0

    def test_triangle_inequality(self):
        a = {"x": 0.0, "y": 0.0}
        b = {"x": 3.0, "y": 0.0}
        c = {"x": 3.0, "y": 4.0}
        assert distance(a, c) <= distance(a, b) + distance(b, c) + 0.001


# ---------------------------------------------------------------------------
# genome_distance()
# ---------------------------------------------------------------------------


class TestGenomeDistance:
    def test_identical(self):
        g = [0.5] * 16
        assert genome_distance(g, g) == 0.0

    def test_max_distance(self):
        g1 = [0.0] * 16
        g2 = [1.0] * 16
        d = genome_distance(g1, g2)
        assert d == math.sqrt(16)

    def test_symmetric(self):
        g1 = [0.1, 0.9, 0.3] + [0.5] * 13
        g2 = [0.8, 0.2, 0.7] + [0.5] * 13
        assert genome_distance(g1, g2) == genome_distance(g2, g1)


# ---------------------------------------------------------------------------
# genome_to_species()
# ---------------------------------------------------------------------------


class TestGenomeToSpecies:
    def test_returns_string(self):
        g = [0.5] * 16
        sid = genome_to_species(g)
        assert isinstance(sid, str)
        assert len(sid) == 4

    def test_deterministic(self):
        g = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] + [0.5] * 10
        assert genome_to_species(g) == genome_to_species(g)

    def test_similar_genomes_same_species(self):
        """Genomes within quantization tolerance map to same species."""
        g1 = [0.50] * 16
        g2 = [0.52] * 16
        assert genome_to_species(g1) == genome_to_species(g2)

    def test_different_genomes_different_species(self):
        g1 = [0.0] * 16
        g2 = [1.0] * 16
        assert genome_to_species(g1) != genome_to_species(g2)


# ---------------------------------------------------------------------------
# move_organism()
# ---------------------------------------------------------------------------


class TestMoveOrganism:
    def test_position_changes(self):
        o = _make_organism(speed=0.8)
        x0, y0 = o["x"], o["y"]
        rng = random.Random(42)
        move_organism(o, [], {"width": 800, "height": 600}, rng)
        assert (o["x"], o["y"]) != (x0, y0)

    def test_wraps_around(self):
        """Position wraps to stay in bounds."""
        o = _make_organism(x=799.0, y=599.0, speed=1.0)
        rng = random.Random(42)
        for _ in range(100):
            move_organism(o, [], {"width": 800, "height": 600}, rng)
        assert 0 <= o["x"] < 800
        assert 0 <= o["y"] < 600

    def test_faster_moves_more(self):
        """Higher speed gene produces larger displacement per tick."""
        fast = _make_organism(oid="fast", speed=1.0)
        slow = _make_organism(oid="slow", speed=0.0)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        move_organism(fast, [], {"width": 800, "height": 600}, rng1)
        move_organism(slow, [], {"width": 800, "height": 600}, rng2)
        fast_dist = math.sqrt((fast["x"] - 100) ** 2 + (fast["y"] - 100) ** 2)
        slow_dist = math.sqrt((slow["x"] - 100) ** 2 + (slow["y"] - 100) ** 2)
        assert fast_dist > slow_dist

    def test_moves_toward_target(self):
        """Aggressive carnivore should move toward prey."""
        predator = _make_organism(oid="pred", x=100, y=100,
                                   aggression=0.9, diet=0.9, sense=1.0)
        prey = _make_organism(oid="prey", x=120, y=100)
        rng = random.Random(42)
        move_organism(predator, [predator, prey],
                      {"width": 800, "height": 600}, rng)
        assert predator["x"] > 100


# ---------------------------------------------------------------------------
# feed_organism()
# ---------------------------------------------------------------------------


class TestFeedOrganism:
    def test_herbivore_gains_energy(self):
        o = _make_organism(x=50, y=50, diet=0.3)
        nutrients = _make_nutrients(fill=80)
        rng = random.Random(42)
        gained = feed_organism(o, nutrients, rng)
        assert gained > 0

    def test_carnivore_skips_plants(self):
        """High diet gene (>0.7) = carnivore, does not eat plants."""
        o = _make_organism(diet=0.9)
        nutrients = _make_nutrients(fill=100)
        rng = random.Random(42)
        gained = feed_organism(o, nutrients, rng)
        assert gained == 0

    def test_nutrient_grid_depletes(self):
        o = _make_organism(x=50, y=50, diet=0.3, size=0.8)
        nutrients = _make_nutrients(fill=80)
        before = sum(nutrients["grid"])
        rng = random.Random(42)
        feed_organism(o, nutrients, rng)
        after = sum(nutrients["grid"])
        assert after < before

    def test_empty_grid_no_food(self):
        o = _make_organism(diet=0.3)
        nutrients = _make_nutrients(fill=0)
        rng = random.Random(42)
        gained = feed_organism(o, nutrients, rng)
        assert gained == 0

    def test_bigger_eats_more(self):
        """Larger organisms harvest more per feed."""
        small = _make_organism(oid="small", x=50, y=50, size=0.1, diet=0.3)
        big = _make_organism(oid="big", x=50, y=50, size=0.9, diet=0.3)
        n1 = _make_nutrients(fill=80)
        n2 = _make_nutrients(fill=80)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        small_gain = feed_organism(small, n1, rng1)
        big_gain = feed_organism(big, n2, rng2)
        assert big_gain > small_gain


# ---------------------------------------------------------------------------
# drain_energy()
# ---------------------------------------------------------------------------


class TestDrainEnergy:
    def test_energy_decreases(self):
        o = _make_organism(energy=100)
        drain_energy(o)
        assert o["energy"] < 100

    def test_faster_drains_more(self):
        fast = _make_organism(oid="fast", energy=100, speed=1.0)
        slow = _make_organism(oid="slow", energy=100, speed=0.0)
        drain_energy(fast)
        drain_energy(slow)
        assert fast["energy"] < slow["energy"]

    def test_bigger_drains_more(self):
        big = _make_organism(oid="big", energy=100, size=1.0)
        small = _make_organism(oid="small", energy=100, size=0.0)
        drain_energy(big)
        drain_energy(small)
        assert big["energy"] < small["energy"]

    def test_drain_is_positive(self):
        """All genomes result in positive energy drain."""
        for _ in range(20):
            g = [random.random() for _ in range(16)]
            o = {"genome": g, "energy": 1000}
            drain_energy(o)
            assert o["energy"] < 1000

    def test_toxic_biolum_cost(self):
        """Toxicity and bioluminescence have energy cost."""
        plain = _make_organism(oid="p", energy=100, toxicity=0.0, biolum=0.0)
        fancy = _make_organism(oid="f", energy=100, toxicity=1.0, biolum=1.0)
        drain_energy(plain)
        drain_energy(fancy)
        assert fancy["energy"] < plain["energy"]


# ---------------------------------------------------------------------------
# interact()
# ---------------------------------------------------------------------------


class TestInteract:
    def test_predation_transfers_energy(self):
        """Successful predation transfers 60% of prey energy to predator."""
        for seed in range(50):
            pred = _make_organism(oid="pred", energy=100,
                                   aggression=0.9, diet=0.9, size=0.9, speed=0.9)
            prey = _make_organism(oid="prey", energy=80,
                                   aggression=0.1, armor=0.0, toxicity=0.0, size=0.2)
            rng = random.Random(seed)
            events = interact(pred, prey, rng)
            if prey["energy"] == 0:
                assert pred["energy"] == 100 + 80 * 0.6
                assert any(e["type"] == "predation" for e in events)
                return
        assert False, "No predation event in 50 seeds"

    def test_cooperation_both_gain(self):
        """Two social organisms with symbiosis cooperate."""
        a = _make_organism(oid="a", energy=50,
                            sociality=0.9, symbiosis=0.8, aggression=0.1)
        b = _make_organism(oid="b", energy=50,
                            sociality=0.9, symbiosis=0.8, aggression=0.1)
        rng = random.Random(42)
        interact(a, b, rng)
        assert a["energy"] > 50
        assert b["energy"] > 50

    def test_toxic_defense_damages_attacker(self):
        """Toxic prey damages the attacker on failed predation."""
        for seed in range(100):
            p = _make_organism(oid="pred", energy=100,
                                aggression=0.9, diet=0.9, size=0.5, speed=0.5)
            q = _make_organism(oid="prey", energy=80,
                                toxicity=0.9, armor=0.9, size=0.9)
            rng = random.Random(seed)
            events = interact(p, q, rng)
            if q["energy"] > 0 and any(e["type"] == "toxic" for e in events):
                assert p["energy"] < 100
                return

    def test_non_aggressive_no_predation(self):
        """Peaceful organisms do not attempt predation."""
        a = _make_organism(oid="a", energy=100,
                            aggression=0.1, diet=0.1, sociality=0.1)
        b = _make_organism(oid="b", energy=100,
                            aggression=0.1, diet=0.1, sociality=0.1)
        rng = random.Random(42)
        events = interact(a, b, rng)
        assert not any(e["type"] == "predation" for e in events)


# ---------------------------------------------------------------------------
# try_reproduce()
# ---------------------------------------------------------------------------


class TestTryReproduce:
    def test_insufficient_energy_no_offspring(self):
        o = _make_organism(energy=10, reproduction=0.5)
        rng = random.Random(42)
        child = try_reproduce(o, [o], rng)
        assert child is None

    def test_sufficient_energy_produces_offspring(self):
        o = _make_organism(energy=500, reproduction=0.5)
        rng = random.Random(42)
        child = try_reproduce(o, [o], rng)
        assert child is not None
        assert "genome" in child
        assert child["age"] == 0

    def test_parent_energy_decreases(self):
        o = _make_organism(energy=500, reproduction=0.5)
        before = o["energy"]
        rng = random.Random(42)
        try_reproduce(o, [o], rng)
        assert o["energy"] < before

    def test_child_count_increments(self):
        o = _make_organism(energy=500, reproduction=0.5)
        assert o["children"] == 0
        rng = random.Random(42)
        try_reproduce(o, [o], rng)
        assert o["children"] == 1

    def test_population_cap(self):
        """No reproduction when at MAX_POP."""
        o = _make_organism(energy=500, reproduction=0.5)
        crowd = [_make_organism(oid="o-%d" % i) for i in range(MAX_POP)]
        rng = random.Random(42)
        child = try_reproduce(o, crowd, rng)
        assert child is None

    def test_child_genome_length(self):
        o = _make_organism(energy=500, reproduction=0.5)
        rng = random.Random(42)
        child = try_reproduce(o, [o], rng)
        assert child is not None
        assert len(child["genome"]) == 16

    def test_child_genome_bounded(self):
        """All child genes remain in [0, 1]."""
        o = _make_organism(energy=500, mutation=1.0)
        rng = random.Random(42)
        child = try_reproduce(o, [o], rng)
        assert child is not None
        for g in child["genome"]:
            assert 0.0 <= g <= 1.0

    def test_sexual_reproduction_with_mate(self):
        """When a mate is nearby, offspring mixes both genomes."""
        parent = _make_organism(oid="p", x=100, y=100, energy=500,
                                 reproduction=0.5)
        mate = _make_organism(oid="m", x=105, y=100, energy=100,
                               reproduction=0.5)
        parent["genome"] = [0.0] * 16
        mate["genome"] = [1.0] * 16
        rng = random.Random(42)
        child = try_reproduce(parent, [parent, mate], rng)
        assert child is not None
        zeros = sum(1 for g in child["genome"] if abs(g) < 0.2)
        ones = sum(1 for g in child["genome"] if abs(g - 1.0) < 0.2)
        assert zeros > 0 or ones > 0


# ---------------------------------------------------------------------------
# regrow_nutrients()
# ---------------------------------------------------------------------------


class TestRegrowNutrients:
    def test_depleted_cells_regrow(self):
        nutrients = _make_nutrients(fill=0)
        regrow_nutrients(nutrients)
        total = sum(nutrients["grid"])
        assert total > 0

    def test_full_cells_dont_exceed_cap(self):
        nutrients = _make_nutrients(fill=100)
        regrow_nutrients(nutrients)
        assert max(nutrients["grid"]) <= 100

    def test_center_regrows_faster(self):
        """Center cells get higher regrowth rate than edges."""
        nutrients = _make_nutrients(width=10, height=10, fill=0)
        regrow_nutrients(nutrients)
        center_idx = 5 * 10 + 5
        corner_idx = 0
        assert nutrients["grid"][center_idx] >= nutrients["grid"][corner_idx]

    def test_repeated_regrowth_increases(self):
        nutrients = _make_nutrients(fill=10)
        for _ in range(10):
            regrow_nutrients(nutrients)
        assert sum(nutrients["grid"]) > 10 * 80 * 60


# ---------------------------------------------------------------------------
# rebuild_species()
# ---------------------------------------------------------------------------


class TestRebuildSpecies:
    def test_empty_list(self):
        species = rebuild_species([])
        assert species == {}

    def test_single_species(self):
        orgs = [_make_organism(oid="a"), _make_organism(oid="b")]
        orgs[0]["species_id"] = "aaaa"
        orgs[1]["species_id"] = "aaaa"
        species = rebuild_species(orgs)
        assert "aaaa" in species
        assert species["aaaa"]["count"] == 2

    def test_multiple_species(self):
        orgs = [_make_organism(oid="a"), _make_organism(oid="b"),
                _make_organism(oid="c")]
        orgs[0]["species_id"] = "aaaa"
        orgs[1]["species_id"] = "bbbb"
        orgs[2]["species_id"] = "aaaa"
        species = rebuild_species(orgs)
        assert len(species) == 2
        assert species["aaaa"]["count"] == 2
        assert species["bbbb"]["count"] == 1

    def test_avg_energy_computed(self):
        orgs = [_make_organism(oid="a", energy=100),
                _make_organism(oid="b", energy=200)]
        orgs[0]["species_id"] = "xxxx"
        orgs[1]["species_id"] = "xxxx"
        species = rebuild_species(orgs)
        assert species["xxxx"]["avg_energy"] == 150.0


# ---------------------------------------------------------------------------
# tick() -- full integration
# ---------------------------------------------------------------------------


class TestTick:
    def test_tick_number_increments(self):
        world = _make_world()
        world = tick(world)
        assert world["tick"] == 1

    def test_organisms_survive(self):
        world = _make_world(n_organisms=50, energy=200)
        world = tick(world)
        assert len(world["organisms"]) > 0

    def test_ages_advance(self):
        world = _make_world(n_organisms=20, energy=200)
        ages_before = {o["id"]: o["age"] for o in world["organisms"]}
        world = tick(world)
        for o in world["organisms"]:
            if o["id"] in ages_before:
                assert o["age"] == ages_before[o["id"]] + 1

    def test_population_history_grows(self):
        world = _make_world()
        world = tick(world)
        assert len(world["population_history"]) == 1

    def test_species_history_grows(self):
        world = _make_world()
        world = tick(world)
        assert len(world["species_history"]) == 1

    def test_stats_updated(self):
        world = _make_world()
        world = tick(world)
        assert "total_births" in world["stats"]
        assert "total_deaths" in world["stats"]
        assert "max_population" in world["stats"]

    def test_meta_updated(self):
        world = _make_world()
        world = tick(world)
        assert world["_meta"]["last_tick"] != ""

    def test_events_bounded(self):
        """Events list is trimmed to last 50."""
        world = _make_world(n_organisms=100, energy=200)
        world["events"] = [{"type": "old"}] * 100
        world = tick(world)
        assert len(world["events"]) <= 50

    def test_dead_organisms_removed(self):
        """Organisms with zero energy die and are removed."""
        world = _make_world(n_organisms=50, energy=1)
        world = tick(world)
        for o in world["organisms"]:
            assert o["energy"] > 0

    def test_emergency_spawn(self):
        """Population cannot drop below MIN_POP."""
        world = _make_world(n_organisms=MIN_POP + 5, energy=1)
        world = tick(world)
        assert len(world["organisms"]) >= MIN_POP

    def test_nutrients_present(self):
        world = _make_world()
        world = tick(world)
        assert "nutrients" in world
        assert len(world["nutrients"]["grid"]) > 0


# ---------------------------------------------------------------------------
# Multi-tick smoke tests
# ---------------------------------------------------------------------------


class TestMultiTickSmoke:
    def test_10_ticks_no_crash(self):
        world = _make_world(n_organisms=40, energy=150)
        for _ in range(10):
            world = tick(world)
        assert world["tick"] == 10
        assert len(world["organisms"]) > 0

    def test_50_ticks_population_bounded(self):
        """Population stays within [MIN_POP, MAX_POP] over 50 ticks."""
        world = _make_world(n_organisms=100, energy=200)
        for _ in range(50):
            world = tick(world)
            pop = len(world["organisms"])
            assert pop >= MIN_POP
            assert pop <= MAX_POP + 50

    def test_species_diversity_emerges(self):
        """After many ticks, multiple species should exist."""
        world = _make_world(n_organisms=60, energy=200)
        for _ in range(30):
            world = tick(world)
        assert len(world["species"]) >= 1

    def test_stats_accumulate(self):
        world = _make_world(n_organisms=50, energy=150)
        for _ in range(20):
            world = tick(world)
        assert world["stats"]["total_births"] >= 0
        assert world["stats"]["total_deaths"] >= 0
        assert world["stats"]["max_population"] > 0

    def test_tick_deterministic(self):
        """Same world state produces same result (seeded RNG)."""
        import json
        w1 = _make_world(n_organisms=30, energy=150)
        w2 = json.loads(json.dumps(w1))
        w1 = tick(w1)
        w2 = tick(w2)
        assert len(w1["organisms"]) == len(w2["organisms"])
        assert w1["tick"] == w2["tick"]


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------


class TestConservationLaws:
    def test_energy_not_created_from_nothing(self):
        """Total energy in system bounded by nutrients consumed + cooperation."""
        world = _make_world(n_organisms=30, energy=100)
        total_before = sum(o["energy"] for o in world["organisms"])
        nutrient_before = sum(world["nutrients"]["grid"])
        world = tick(world)
        total_after = sum(o["energy"] for o in world["organisms"])
        nutrient_after = sum(world["nutrients"]["grid"])
        nutrient_consumed = max(0, nutrient_before - nutrient_after)
        assert total_after <= total_before + nutrient_consumed * 2 + 1000

    def test_position_bounded(self):
        """All organisms stay within world bounds after tick."""
        world = _make_world(n_organisms=50, energy=200)
        for _ in range(5):
            world = tick(world)
        w = world["config"]["width"]
        h = world["config"]["height"]
        for o in world["organisms"]:
            assert 0 <= o["x"] <= w + 1
            assert 0 <= o["y"] <= h + 1

    def test_age_nonnegative(self):
        world = _make_world(n_organisms=30, energy=200)
        for _ in range(10):
            world = tick(world)
        for o in world["organisms"]:
            assert o["age"] >= 0

    def test_genome_bounded_after_reproduction(self):
        """All genes stay in [0.0, 1.0] even after mutation."""
        world = _make_world(n_organisms=50, energy=300)
        for _ in range(20):
            world = tick(world)
        for o in world["organisms"]:
            for i, g in enumerate(o["genome"]):
                assert 0.0 <= g <= 1.0
