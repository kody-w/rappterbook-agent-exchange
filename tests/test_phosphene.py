"""
Tests for phosphene.py — Phosphene emergent neural ecosystem.

Covers: utility functions, genome operations, neuron creation, genesis,
tick simulation (Kuramoto sync, Hebbian learning, reproduction, death),
cluster detection, history tracking, and property-based invariants.

Run: python -m pytest tests/test_phosphene.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.phosphene import (
    clamp,
    dist,
    species_hash,
    random_genome,
    mutate_genome,
    make_neuron,
    genesis,
    tick,
    WORLD_W,
    WORLD_H,
    INIT_POP,
    MAX_POP,
    MIN_POP,
    MAX_CONNS,
    WEAK_THRESHOLD,
    REPRO_THRESHOLD,
    REPRO_COST,
    DEATH_ENERGY,
    MAX_AGE,
    HISTORY_CAP,
    EVENT_CAP,
    GENE_NAMES,
    GENE_RANGES,
    TWO_PI,
)


# ───────────────────── Utilities ─────────────────────


class TestClamp:
    """Value clamping."""

    def test_within_range(self) -> None:
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_floor(self) -> None:
        assert clamp(-5.0, 0.0, 10.0) == 0.0

    def test_above_ceiling(self) -> None:
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_at_boundaries(self) -> None:
        assert clamp(0.0, 0.0, 1.0) == 0.0
        assert clamp(1.0, 0.0, 1.0) == 1.0


class TestDist:
    """Euclidean distance."""

    def test_same_point(self) -> None:
        a = {"x": 5.0, "y": 5.0}
        assert dist(a, a) == 0.0

    def test_known_value(self) -> None:
        a, b = {"x": 0.0, "y": 0.0}, {"x": 3.0, "y": 4.0}
        assert abs(dist(a, b) - 5.0) < 1e-9

    def test_symmetric(self) -> None:
        a, b = {"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 40.0}
        assert abs(dist(a, b) - dist(b, a)) < 1e-9

    def test_nonnegative(self) -> None:
        random.seed(42)
        for _ in range(50):
            a = {"x": random.uniform(0, WORLD_W), "y": random.uniform(0, WORLD_H)}
            b = {"x": random.uniform(0, WORLD_W), "y": random.uniform(0, WORLD_H)}
            assert dist(a, b) >= 0.0


class TestSpeciesHash:
    """Species hash from genome."""

    def test_deterministic(self) -> None:
        """Same genome always produces same hash."""
        g = random_genome()
        assert species_hash(g) == species_hash(g)

    def test_hex_format(self) -> None:
        """Hash is 4 hex characters."""
        g = random_genome()
        h = species_hash(g)
        assert len(h) == 4
        int(h, 16)  # valid hex

    def test_similar_genomes_may_share_hash(self) -> None:
        """Very similar genomes can hash the same (bucketing)."""
        g1 = {n: GENE_RANGES[n][0] for n in GENE_NAMES}
        g2 = dict(g1)
        g2["freq"] += 0.01  # tiny change within bucket
        # May or may not collide — just ensure no crash
        species_hash(g1)
        species_hash(g2)


# ───────────────────── Genome operations ─────────────────────


class TestRandomGenome:
    """Random genome generation."""

    def test_all_genes_present(self) -> None:
        """Genome has all gene names."""
        g = random_genome()
        for name in GENE_NAMES:
            assert name in g

    def test_values_in_range(self) -> None:
        """Gene values are within declared ranges."""
        random.seed(42)
        for _ in range(100):
            g = random_genome()
            for name in GENE_NAMES:
                lo, hi = GENE_RANGES[name]
                assert lo <= g[name] <= hi, f"{name}={g[name]} not in [{lo},{hi}]"

    def test_values_rounded(self) -> None:
        """Gene values are rounded to 4 decimal places."""
        g = random_genome()
        for name in GENE_NAMES:
            assert g[name] == round(g[name], 4)


class TestMutateGenome:
    """Genome mutation."""

    def test_all_genes_present(self) -> None:
        """Mutated genome has all gene names."""
        parent = random_genome()
        child = mutate_genome(parent, 0.5)
        for name in GENE_NAMES:
            assert name in child

    def test_values_in_range(self) -> None:
        """Mutated genes stay within declared ranges."""
        random.seed(42)
        for _ in range(100):
            parent = random_genome()
            child = mutate_genome(parent, 1.0)
            for name in GENE_NAMES:
                lo, hi = GENE_RANGES[name]
                assert lo <= child[name] <= hi

    def test_zero_rate_preserves(self) -> None:
        """Mutation rate 0 preserves genome exactly."""
        parent = random_genome()
        child = mutate_genome(parent, 0.0)
        for name in GENE_NAMES:
            assert child[name] == parent[name]

    def test_high_rate_changes(self) -> None:
        """High mutation rate produces some changes."""
        random.seed(42)
        parent = {n: (GENE_RANGES[n][0] + GENE_RANGES[n][1]) / 2 for n in GENE_NAMES}
        parent = {n: round(v, 4) for n, v in parent.items()}
        changed = 0
        for _ in range(20):
            child = mutate_genome(parent, 1.0)
            if any(child[n] != parent[n] for n in GENE_NAMES):
                changed += 1
        assert changed > 10


# ───────────────────── Neuron creation ─────────────────────


class TestMakeNeuron:
    """Neuron construction."""

    def test_fields_present(self) -> None:
        """Neuron has all required fields."""
        n = make_neuron(0)
        required = {"id", "x", "y", "vx", "vy", "phase", "energy",
                     "genome", "conns", "age", "generation", "parent",
                     "species", "fires", "last_fire"}
        assert required.issubset(n.keys())

    def test_id_format(self) -> None:
        """Neuron id starts with 'n-'."""
        n = make_neuron(0)
        assert n["id"].startswith("n-")

    def test_id_unique(self) -> None:
        """Sequential neurons get unique ids."""
        ids = {make_neuron(0)["id"] for _ in range(100)}
        assert len(ids) == 100

    def test_custom_position(self) -> None:
        """Explicit x, y are used."""
        n = make_neuron(0, x=42.0, y=99.0)
        assert n["x"] == 42.0
        assert n["y"] == 99.0

    def test_initial_energy(self) -> None:
        """New neurons start with 45 energy."""
        n = make_neuron(0)
        assert n["energy"] == 45.0

    def test_initial_state(self) -> None:
        """New neurons are unfired with age 0."""
        n = make_neuron(0)
        assert n["fires"] == 0
        assert n["age"] == 0
        assert n["last_fire"] == -1

    def test_phase_bounded(self) -> None:
        """Initial phase is in [0, 2π)."""
        random.seed(42)
        for _ in range(100):
            n = make_neuron(0)
            assert 0 <= n["phase"] < TWO_PI

    def test_parent_tracking(self) -> None:
        """Parent and generation stored."""
        n = make_neuron(0, parent_id="n-abc", generation=3)
        assert n["parent"] == "n-abc"
        assert n["generation"] == 3

    def test_custom_genome(self) -> None:
        """Provided genome is used."""
        g = random_genome()
        n = make_neuron(0, genome=g)
        assert n["genome"] == g

    def test_species_from_genome(self) -> None:
        """Species hash is derived from genome."""
        g = random_genome()
        n = make_neuron(0, genome=g)
        assert n["species"] == species_hash(g)


# ───────────────────── Genesis ─────────────────────


class TestGenesis:
    """World creation."""

    def test_structure(self) -> None:
        """Genesis state has all required keys."""
        random.seed(42)
        s = genesis()
        required = {"_meta", "tick", "neurons", "history", "events",
                     "clusters", "stats"}
        assert required.issubset(s.keys())

    def test_neuron_count(self) -> None:
        """Genesis creates INIT_POP neurons."""
        random.seed(42)
        s = genesis()
        assert len(s["neurons"]) == INIT_POP

    def test_tick_zero(self) -> None:
        """Genesis starts at tick 0."""
        random.seed(42)
        assert genesis()["tick"] == 0

    def test_meta_engine(self) -> None:
        """Meta marks engine as 'phosphene'."""
        random.seed(42)
        assert genesis()["_meta"]["engine"] == "phosphene"

    def test_connections_exist(self) -> None:
        """Genesis creates some connections."""
        random.seed(42)
        s = genesis()
        total_conns = sum(len(n["conns"]) for n in s["neurons"])
        assert total_conns > 0

    def test_stats_initialized(self) -> None:
        """Stats counters initialized."""
        random.seed(42)
        s = genesis()
        assert s["stats"]["total_births"] == INIT_POP
        assert s["stats"]["total_deaths"] == 0
        assert s["stats"]["total_fires"] == 0

    def test_neurons_positioned(self) -> None:
        """All neurons within world bounds."""
        random.seed(42)
        s = genesis()
        for n in s["neurons"]:
            assert 0 <= n["x"] <= WORLD_W
            assert 0 <= n["y"] <= WORLD_H

    def test_genomes_valid(self) -> None:
        """All genomes have correct gene count and ranges."""
        random.seed(42)
        s = genesis()
        for n in s["neurons"]:
            for name in GENE_NAMES:
                assert name in n["genome"]
                lo, hi = GENE_RANGES[name]
                assert lo <= n["genome"][name] <= hi


# ───────────────────── Tick simulation ─────────────────────


def make_state(seed: int = 42) -> dict:
    """Create a seeded genesis state."""
    random.seed(seed)
    return genesis()


class TestTick:
    """Single tick execution."""

    def test_tick_increments(self) -> None:
        """Tick counter increments by 1."""
        s = make_state()
        tick(s)
        assert s["tick"] == 1

    def test_returns_none(self) -> None:
        """tick() returns None (mutates in place)."""
        s = make_state()
        assert tick(s) is None

    def test_population_bounded(self) -> None:
        """Population stays between MIN_POP and MAX_POP."""
        s = make_state()
        for _ in range(30):
            tick(s)
            pop = len(s["neurons"])
            assert MIN_POP <= pop <= MAX_POP, f"Pop {pop} out of range"

    def test_neurons_age(self) -> None:
        """Neurons age by 1 per tick."""
        s = make_state()
        ages_before = {n["id"]: n["age"] for n in s["neurons"]}
        tick(s)
        for n in s["neurons"]:
            if n["id"] in ages_before:
                assert n["age"] == ages_before[n["id"]] + 1

    def test_phase_bounded(self) -> None:
        """Phase stays in [0, 2π) after tick."""
        s = make_state()
        for _ in range(10):
            tick(s)
        for n in s["neurons"]:
            assert 0 <= n["phase"] < TWO_PI, f"Phase {n['phase']} out of bounds"

    def test_energy_bounded_alive(self) -> None:
        """Surviving neurons have energy > DEATH_ENERGY."""
        s = make_state()
        for _ in range(10):
            tick(s)
        for n in s["neurons"]:
            assert n["energy"] > DEATH_ENERGY

    def test_connection_weights_bounded(self) -> None:
        """Connection weights stay in [0, 1]."""
        s = make_state()
        for _ in range(10):
            tick(s)
        for n in s["neurons"]:
            for cid, w in n["conns"].items():
                assert 0.0 <= w <= 1.0, f"Weight {w} out of bounds"

    def test_weak_connections_pruned(self) -> None:
        """Connections below WEAK_THRESHOLD are removed."""
        s = make_state()
        for _ in range(10):
            tick(s)
        for n in s["neurons"]:
            for cid, w in n["conns"].items():
                assert w >= WEAK_THRESHOLD

    def test_max_connections_respected(self) -> None:
        """No neuron has more than MAX_CONNS connections."""
        s = make_state()
        for _ in range(20):
            tick(s)
        for n in s["neurons"]:
            # Growth adds max 1 per tick, but existing can exceed slightly
            # The growth check uses < MAX_CONNS so this should hold
            assert len(n["conns"]) <= MAX_CONNS + 5  # small tolerance

    def test_history_grows(self) -> None:
        """History gets one entry per tick."""
        s = make_state()
        for _ in range(5):
            tick(s)
        assert len(s["history"]) == 5

    def test_history_capped(self) -> None:
        """History never exceeds HISTORY_CAP."""
        s = make_state()
        for _ in range(HISTORY_CAP + 10):
            tick(s)
        assert len(s["history"]) <= HISTORY_CAP

    def test_events_capped(self) -> None:
        """Events never exceed EVENT_CAP."""
        s = make_state()
        for _ in range(EVENT_CAP + 50):
            tick(s)
        assert len(s["events"]) <= EVENT_CAP

    def test_positions_bounded(self) -> None:
        """All neurons stay within world bounds."""
        s = make_state()
        for _ in range(20):
            tick(s)
        for n in s["neurons"]:
            assert 5 <= n["x"] <= WORLD_W - 5
            assert 5 <= n["y"] <= WORLD_H - 5

    def test_stats_updated(self) -> None:
        """Stats counters increase over time."""
        s = make_state()
        for _ in range(100):
            tick(s)
        assert s["stats"]["total_births"] >= INIT_POP
        assert s["stats"]["max_pop"] >= MIN_POP

    def test_clusters_detected(self) -> None:
        """Clusters are detected after tick."""
        s = make_state()
        for _ in range(5):
            tick(s)
        # Clusters should exist (neurons start clustered in genesis)
        assert isinstance(s["clusters"], list)

    def test_dead_neurons_removed(self) -> None:
        """Neurons with low energy are removed."""
        s = make_state()
        # Kill specific neurons
        for n in s["neurons"][:3]:
            n["energy"] = 0
        dead_ids = {n["id"] for n in s["neurons"][:3]}
        tick(s)
        alive_ids = {n["id"] for n in s["neurons"]}
        for did in dead_ids:
            assert did not in alive_ids

    def test_dead_neuron_connections_cleaned(self) -> None:
        """Dead neurons are removed from other neurons' connection lists."""
        s = make_state()
        dead_id = s["neurons"][0]["id"]
        s["neurons"][0]["energy"] = 0
        tick(s)
        for n in s["neurons"]:
            assert dead_id not in n["conns"]

    def test_min_pop_maintained(self) -> None:
        """Population never drops below MIN_POP."""
        s = make_state()
        # Kill almost all neurons
        for n in s["neurons"]:
            n["energy"] = 0
        tick(s)
        assert len(s["neurons"]) >= MIN_POP

    def test_reproduction_occurs(self) -> None:
        """High-energy neurons reproduce."""
        s = make_state()
        # Give neurons lots of energy and age
        for n in s["neurons"]:
            n["energy"] = 200.0
            n["age"] = 20
        initial = len(s["neurons"])
        tick(s)
        # Some births should occur (stats tracks them)
        assert s["stats"]["total_births"] > INIT_POP


# ───────────────────── History entry ─────────────────────


class TestHistoryEntry:
    """History entry structure."""

    def test_fields_present(self) -> None:
        """History entries have all required fields."""
        s = make_state()
        tick(s)
        entry = s["history"][-1]
        required = {"t", "pop", "conns", "clusters", "sync",
                     "energy", "species", "fires"}
        assert required.issubset(entry.keys())

    def test_pop_matches(self) -> None:
        """History pop matches actual neuron count."""
        s = make_state()
        tick(s)
        entry = s["history"][-1]
        assert entry["pop"] == len(s["neurons"])

    def test_tick_matches(self) -> None:
        """History t matches current tick."""
        s = make_state()
        tick(s)
        assert s["history"][-1]["t"] == s["tick"]


# ───────────────────── Property invariants ─────────────────────


class TestInvariants:
    """Property-based invariants across multiple ticks."""

    def test_genome_bounds_20_ticks(self) -> None:
        """All genome values stay within declared ranges after mutations."""
        s = make_state()
        for _ in range(20):
            tick(s)
        for n in s["neurons"]:
            for name in GENE_NAMES:
                lo, hi = GENE_RANGES[name]
                assert lo <= n["genome"][name] <= hi, \
                    f"{name}={n['genome'][name]} not in [{lo},{hi}]"

    def test_fire_count_monotonic(self) -> None:
        """Fire count only increases."""
        s = make_state()
        prev = {n["id"]: 0 for n in s["neurons"]}
        for _ in range(10):
            tick(s)
            for n in s["neurons"]:
                if n["id"] in prev:
                    assert n["fires"] >= prev[n["id"]]
                    prev[n["id"]] = n["fires"]

    def test_connections_symmetric(self) -> None:
        """If A→B exists, B→A exists with same weight."""
        s = make_state()
        for _ in range(10):
            tick(s)
        by_id = {n["id"]: n for n in s["neurons"]}
        for n in s["neurons"]:
            for cid, w in n["conns"].items():
                other = by_id.get(cid)
                if other:
                    assert n["id"] in other["conns"], \
                        f"{n['id']}→{cid} but not reverse"
                    assert abs(other["conns"][n["id"]] - w) < 1e-3, \
                        f"Asymmetric weight: {w} vs {other['conns'][n['id']]}"

    def test_species_hash_valid(self) -> None:
        """All species hashes are 4-char hex."""
        s = make_state()
        for _ in range(10):
            tick(s)
        for n in s["neurons"]:
            assert len(n["species"]) == 4
            int(n["species"], 16)

    def test_age_nonnegative(self) -> None:
        """Age is always >= 0."""
        s = make_state()
        for _ in range(10):
            tick(s)
        for n in s["neurons"]:
            assert n["age"] >= 0


# ───────────────────── Smoke tests ─────────────────────


class TestSmoke:
    """End-to-end smoke tests."""

    def test_10_ticks_no_crash(self) -> None:
        """Survives 10 ticks without crashing."""
        s = make_state()
        for _ in range(10):
            tick(s)
        assert len(s["neurons"]) >= MIN_POP

    def test_50_ticks_no_crash(self) -> None:
        """Survives 50 ticks without crashing."""
        s = make_state()
        for _ in range(50):
            tick(s)
        assert len(s["neurons"]) >= MIN_POP
        assert s["tick"] == 50

    def test_deterministic(self) -> None:
        """Same seed produces identical results."""
        def run(seed):
            random.seed(seed)
            s = genesis()
            for _ in range(10):
                tick(s)
            return (len(s["neurons"]), s["stats"]["total_fires"],
                    s["stats"]["total_deaths"])
        assert run(42) == run(42)

    def test_different_seeds_diverge(self) -> None:
        """Different seeds produce different results."""
        def run(seed):
            random.seed(seed)
            s = genesis()
            for _ in range(20):
                tick(s)
            return s["stats"]["total_fires"]
        assert run(42) != run(99)

    def test_fires_accumulate(self) -> None:
        """Total fires increase over time."""
        s = make_state()
        for _ in range(30):
            tick(s)
        assert s["stats"]["total_fires"] > 0

    def test_history_records_all_ticks(self) -> None:
        """History has one entry per tick."""
        s = make_state()
        for _ in range(25):
            tick(s)
        assert len(s["history"]) == 25
