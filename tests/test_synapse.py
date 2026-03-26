"""
test_synapse.py — Unit tests for The Synapse neural evolution engine.

70 tests covering utility functions, neuron/synapse construction, genesis,
simulation steps, evolution (birth/death/pruning), clustering, dream state,
snapshot, IO, and physical invariants.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import synapse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def deterministic():
    """Fix random seed for reproducible tests."""
    random.seed(42)
    yield


@pytest.fixture
def world():
    """Return a freshly generated world."""
    return synapse.genesis()


@pytest.fixture
def tmp_state(tmp_path):
    """Patch STATE_DIR and STATE_PATH to temp directory."""
    old_dir = synapse.STATE_DIR
    old_path = synapse.STATE_PATH
    synapse.STATE_DIR = tmp_path
    synapse.STATE_PATH = tmp_path / "synapse_state.json"
    yield tmp_path
    synapse.STATE_DIR = old_dir
    synapse.STATE_PATH = old_path


# ===========================================================================
# Utility Functions
# ===========================================================================

class TestClamp:

    def test_within_range(self):
        assert synapse.clamp(0.5) == 0.5

    def test_below_min(self):
        assert synapse.clamp(-1.0) == 0.0

    def test_above_max(self):
        assert synapse.clamp(2.0) == 1.0

    def test_custom_bounds(self):
        assert synapse.clamp(5, 0, 10) == 5
        assert synapse.clamp(-5, 0, 10) == 0
        assert synapse.clamp(15, 0, 10) == 10

    def test_at_boundaries(self):
        assert synapse.clamp(0.0) == 0.0
        assert synapse.clamp(1.0) == 1.0


class TestDist:

    def test_same_point(self):
        a = {"x": 10, "y": 20}
        assert synapse.dist(a, a) == 0.0

    def test_known_distance(self):
        a = {"x": 0, "y": 0}
        b = {"x": 3, "y": 4}
        assert synapse.dist(a, b) == pytest.approx(5.0)

    def test_symmetric(self):
        a = {"x": 10, "y": 20}
        b = {"x": 30, "y": 40}
        assert synapse.dist(a, b) == pytest.approx(synapse.dist(b, a))

    def test_non_negative(self):
        for _ in range(50):
            a = {"x": random.uniform(-100, 100), "y": random.uniform(-100, 100)}
            b = {"x": random.uniform(-100, 100), "y": random.uniform(-100, 100)}
            assert synapse.dist(a, b) >= 0.0


class TestEpochName:

    def test_tick_zero(self):
        assert synapse.epoch_name(0) == "Silent Void"

    def test_first_spark(self):
        assert synapse.epoch_name(20) == "First Spark"

    def test_transcendence(self):
        assert synapse.epoch_name(6000) == "Transcendence"

    def test_large_tick(self):
        assert synapse.epoch_name(999999) == "Transcendence"

    def test_between_epochs(self):
        assert synapse.epoch_name(150) == "Kindling"


class TestGeneVal:

    def test_min(self):
        genome = [0.0] * synapse.GENE_COUNT
        assert synapse.gene_val(genome, 0, 0.2, 0.9) == pytest.approx(0.2)

    def test_max(self):
        genome = [1.0] * synapse.GENE_COUNT
        assert synapse.gene_val(genome, 0, 0.2, 0.9) == pytest.approx(0.9)

    def test_midpoint(self):
        genome = [0.5] * synapse.GENE_COUNT
        assert synapse.gene_val(genome, 0, 0.0, 1.0) == pytest.approx(0.5)

    def test_all_gene_indices(self):
        """Every gene index maps correctly."""
        genome = [i / (synapse.GENE_COUNT - 1) for i in range(synapse.GENE_COUNT)]
        for i in range(synapse.GENE_COUNT):
            val = synapse.gene_val(genome, i, 0.0, 1.0)
            assert 0.0 <= val <= 1.0


class TestUid:

    def test_format(self):
        nid = synapse.uid("n")
        assert nid.startswith("n-")
        assert len(nid) == 8  # "n-" + 6 hex chars

    def test_unique(self):
        ids = {synapse.uid("n") for _ in range(100)}
        assert len(ids) == 100

    def test_custom_prefix(self):
        assert synapse.uid("syn").startswith("syn-")


# ===========================================================================
# Neuron & Synapse Construction
# ===========================================================================

class TestMakeNeuron:

    def test_required_fields(self):
        n = synapse.make_neuron(0)
        required = {"id", "x", "y", "genome", "potential", "energy", "fired",
                     "fire_count", "last_fire", "age", "generation", "born_tick",
                     "parent_id", "cluster_id"}
        assert required.issubset(set(n.keys()))

    def test_genome_length(self):
        n = synapse.make_neuron(0)
        assert len(n["genome"]) == synapse.GENE_COUNT

    def test_initial_energy(self):
        n = synapse.make_neuron(0)
        assert n["energy"] == 100.0

    def test_initial_state(self):
        n = synapse.make_neuron(0)
        assert n["fired"] is False
        assert n["fire_count"] == 0
        assert n["age"] == 0

    def test_custom_position(self):
        n = synapse.make_neuron(0, x=42.0, y=99.0)
        assert n["x"] == 42.0
        assert n["y"] == 99.0

    def test_parent_tracking(self):
        n = synapse.make_neuron(5, parent_id="p-abc", generation=3)
        assert n["parent_id"] == "p-abc"
        assert n["generation"] == 3

    def test_genome_values_rounded(self):
        n = synapse.make_neuron(0)
        for g in n["genome"]:
            assert len(str(g).split(".")[-1]) <= 4


class TestMakeSynapse:

    def test_required_fields(self):
        s = synapse.make_synapse("a", "b")
        assert s["src"] == "a"
        assert s["dst"] == "b"
        assert "weight" in s
        assert "age" in s

    def test_default_weight(self):
        s = synapse.make_synapse("a", "b")
        assert s["weight"] == 0.5

    def test_custom_weight_clamped(self):
        s = synapse.make_synapse("a", "b", weight=5.0)
        assert s["weight"] <= 2.0

    def test_negative_weight_clamped(self):
        s = synapse.make_synapse("a", "b", weight=-1.0)
        assert s["weight"] >= 0.01

    def test_initial_state(self):
        s = synapse.make_synapse("a", "b")
        assert s["age"] == 0
        assert s["signal"] == 0.0


# ===========================================================================
# Genesis
# ===========================================================================

class TestGenesis:

    def test_structure(self, world):
        """Genesis returns all expected top-level keys."""
        required = {"tick", "step", "epoch", "neurons", "synapses",
                     "signals", "history", "events", "stats", "dream", "updated_at"}
        assert required.issubset(set(world.keys()))

    def test_neuron_count(self, world):
        assert len(world["neurons"]) == synapse.INITIAL_NEURONS

    def test_tick_zero(self, world):
        assert world["tick"] == 0
        assert world["step"] == 0

    def test_epoch_silent_void(self, world):
        assert world["epoch"] == "Silent Void"

    def test_synapses_created(self, world):
        """Genesis creates at least some synapses."""
        assert len(world["synapses"]) > 0

    def test_neurons_have_clusters(self, world):
        """Every neuron is assigned a cluster_id."""
        for n in world["neurons"]:
            assert n["cluster_id"] != ""

    def test_stats_initialized(self, world):
        stats = world["stats"]
        assert stats["total_fires"] == 0
        assert stats["total_births"] == 0
        assert stats["total_deaths"] == 0
        assert stats["peak_neurons"] == synapse.INITIAL_NEURONS


# ===========================================================================
# Simulation Step
# ===========================================================================

class TestSimStep:

    def test_returns_events(self, world):
        events = synapse.sim_step(world)
        assert isinstance(events, list)

    def test_step_increments(self, world):
        old_step = world["step"]
        synapse.sim_step(world)
        assert world["step"] == old_step + 1

    def test_potential_non_negative(self, world):
        """After a step, all potentials are non-negative."""
        for _ in range(10):
            synapse.sim_step(world)
        for n in world["neurons"]:
            assert n["potential"] >= 0.0, f"Neuron {n['id']} has negative potential"

    def test_neurons_age(self, world):
        """Neurons age by 1 each step."""
        ages_before = {n["id"]: n["age"] for n in world["neurons"]}
        synapse.sim_step(world)
        for n in world["neurons"]:
            assert n["age"] == ages_before[n["id"]] + 1

    def test_synapses_age(self, world):
        """Synapses age by 1 each step."""
        ages_before = [s["age"] for s in world["synapses"]]
        synapse.sim_step(world)
        for i, s in enumerate(world["synapses"]):
            assert s["age"] == ages_before[i] + 1

    def test_fire_count_monotonic(self, world):
        """fire_count only increases."""
        counts = {n["id"]: n["fire_count"] for n in world["neurons"]}
        for _ in range(20):
            synapse.sim_step(world)
        for n in world["neurons"]:
            assert n["fire_count"] >= counts.get(n["id"], 0)

    def test_cascade_events_have_size(self, world):
        """If a cascade event occurs, it has a size field."""
        all_events = []
        for _ in range(50):
            all_events.extend(synapse.sim_step(world))
        cascades = [e for e in all_events if e["type"] == "cascade"]
        for c in cascades:
            assert "size" in c
            assert c["size"] > 5

    def test_synapse_weight_bounded(self, world):
        """Synapse weights stay in [0.01, 2.0] after many steps."""
        for _ in range(30):
            synapse.sim_step(world)
        for s in world["synapses"]:
            assert 0.0 <= s["weight"] <= 2.0 + 1e-9


# ===========================================================================
# Evolution
# ===========================================================================

class TestEvolve:

    def test_returns_events(self, world):
        events = synapse.evolve(world)
        assert isinstance(events, list)

    def test_dead_neurons_removed(self, world):
        """Neurons with zero energy are removed by evolve."""
        world["neurons"][0]["energy"] = 0
        dead_id = world["neurons"][0]["id"]
        synapse.evolve(world)
        alive_ids = {n["id"] for n in world["neurons"]}
        assert dead_id not in alive_ids

    def test_old_neurons_die(self, world):
        """Neurons past max age are removed."""
        world["neurons"][0]["age"] = 99999
        dead_id = world["neurons"][0]["id"]
        synapse.evolve(world)
        alive_ids = {n["id"] for n in world["neurons"]}
        assert dead_id not in alive_ids

    def test_births_from_high_energy(self, world):
        """High-energy neurons can reproduce."""
        for n in world["neurons"]:
            n["energy"] = 200.0
        initial_births = world["stats"]["total_births"]
        synapse.evolve(world)
        assert world["stats"]["total_births"] > initial_births

    def test_min_neurons_maintained(self):
        """Population never drops below MIN_NEURONS."""
        w = synapse.genesis()
        # Kill all neurons
        for n in w["neurons"]:
            n["energy"] = 0
        synapse.evolve(w)
        assert len(w["neurons"]) >= synapse.MIN_NEURONS

    def test_max_neurons_respected(self, world):
        """Population never exceeds MAX_NEURONS."""
        # Give everyone max energy to trigger reproduction
        for n in world["neurons"]:
            n["energy"] = 999.0
        for _ in range(10):
            synapse.evolve(world)
        assert len(world["neurons"]) <= synapse.MAX_NEURONS

    def test_death_events_generated(self, world):
        """Dead neurons generate death events."""
        world["neurons"][0]["energy"] = 0
        events = synapse.evolve(world)
        death_events = [e for e in events if e["type"] == "death"]
        assert len(death_events) >= 1

    def test_birth_events_generated(self, world):
        """Births generate birth events."""
        for n in world["neurons"]:
            n["energy"] = 200.0
        events = synapse.evolve(world)
        birth_events = [e for e in events if e["type"] == "birth"]
        assert len(birth_events) >= 1

    def test_dead_neuron_synapses_cleaned(self, world):
        """Synapses referencing dead neurons are removed."""
        dead_id = world["neurons"][0]["id"]
        world["neurons"][0]["energy"] = 0
        synapse.evolve(world)
        for s in world["synapses"]:
            assert s["src"] != dead_id
            assert s["dst"] != dead_id

    def test_stats_tracking(self, world):
        """total_ticks increments after evolve."""
        before = world["stats"]["total_ticks"]
        synapse.evolve(world)
        assert world["stats"]["total_ticks"] == before + 1


# ===========================================================================
# Dream State
# ===========================================================================

class TestDreamState:

    def test_dream_fields(self, world):
        """Dream state has all required fields."""
        for _ in range(10):
            synapse.sim_step(world)
        synapse.evolve(world)
        dream = world["dream"]
        assert "pattern" in dream
        assert "intensity" in dream
        assert "coherence" in dream
        assert "dominant_freq" in dream

    def test_intensity_bounded(self, world):
        for _ in range(20):
            synapse.sim_step(world)
        synapse.evolve(world)
        assert 0.0 <= world["dream"]["intensity"] <= 1.0

    def test_coherence_bounded(self, world):
        for _ in range(20):
            synapse.sim_step(world)
        synapse.evolve(world)
        assert 0.0 <= world["dream"]["coherence"] <= 1.0

    def test_pattern_from_known_list(self, world):
        patterns = ["void", "flicker", "pulse", "wave", "spiral", "bloom", "storm", "dream"]
        for _ in range(20):
            synapse.sim_step(world)
        synapse.evolve(world)
        assert world["dream"]["pattern"] in patterns


# ===========================================================================
# Snapshot
# ===========================================================================

class TestSnapshot:

    def test_fields(self, world):
        snap = synapse.snapshot(world)
        required = {"tick", "neurons", "synapses", "clusters", "active", "signals", "dream"}
        assert required.issubset(set(snap.keys()))

    def test_counts_match(self, world):
        snap = synapse.snapshot(world)
        assert snap["neurons"] == len(world["neurons"])
        assert snap["synapses"] == len(world["synapses"])
        assert snap["signals"] == len(world["signals"])


# ===========================================================================
# IO
# ===========================================================================

class TestIO:

    def test_save_creates_file(self, world, tmp_state):
        synapse.save_world(world)
        assert synapse.STATE_PATH.exists()

    def test_save_load_roundtrip(self, world, tmp_state):
        synapse.save_world(world)
        loaded = synapse.load_world()
        assert loaded["tick"] == world["tick"]
        assert len(loaded["neurons"]) == len(world["neurons"])

    def test_load_missing_returns_none(self, tmp_state):
        assert synapse.load_world() is None

    def test_history_capped_on_save(self, world, tmp_state):
        world["history"] = list(range(1000))
        synapse.save_world(world)
        loaded = synapse.load_world()
        assert len(loaded["history"]) <= synapse.HISTORY_CAP

    def test_events_capped_on_save(self, world, tmp_state):
        world["events"] = list(range(500))
        synapse.save_world(world)
        loaded = synapse.load_world()
        assert len(loaded["events"]) <= synapse.EVENT_CAP


# ===========================================================================
# Physical Invariants
# ===========================================================================

class TestPhysicalInvariants:

    def test_neuron_positions_in_bounds(self, world):
        """Neurons stay within world bounds after evolution."""
        for _ in range(5):
            for _ in range(20):
                synapse.sim_step(world)
            synapse.evolve(world)
        for n in world["neurons"]:
            assert 0 <= n["x"] <= synapse.WORLD_W
            assert 0 <= n["y"] <= synapse.WORLD_H

    def test_genome_values_bounded(self, world):
        """All genome values stay in [0, 1] after evolution."""
        for _ in range(5):
            for _ in range(20):
                synapse.sim_step(world)
            synapse.evolve(world)
        for n in world["neurons"]:
            for i, g in enumerate(n["genome"]):
                assert 0.0 <= g <= 1.0, f"Neuron {n['id']} gene {i} = {g}"

    def test_energy_tracking(self, world):
        """Neurons gain energy from firing, lose from metabolism."""
        for _ in range(30):
            synapse.sim_step(world)
        # At least some neurons should have different energy than 100
        energies = [n["energy"] for n in world["neurons"]]
        assert not all(e == 100.0 for e in energies)


# ===========================================================================
# Smoke — Full Pipeline
# ===========================================================================

class TestSmoke:

    def test_full_tick_no_crash(self):
        """Run 3 full ticks (steps + evolve) without crash."""
        w = synapse.genesis()
        for t in range(3):
            w["tick"] += 1
            w["epoch"] = synapse.epoch_name(w["tick"])
            for _ in range(synapse.STEPS_PER_TICK):
                synapse.sim_step(w)
            synapse.evolve(w)
            w["history"].append(synapse.snapshot(w))
        assert w["tick"] == 3
        assert len(w["neurons"]) >= synapse.MIN_NEURONS
        assert len(w["history"]) == 3

    def test_deterministic_with_seed(self):
        """Same random seed produces identical simulation."""
        results = []
        for _ in range(2):
            random.seed(777)
            w = synapse.genesis()
            for _ in range(20):
                synapse.sim_step(w)
            results.append(w["stats"]["total_fires"])
        assert results[0] == results[1]

    def test_10_tick_smoke(self):
        """Run 10 ticks — the synapse lives and evolves."""
        w = synapse.genesis()
        for t in range(10):
            w["tick"] += 1
            for _ in range(40):  # Fewer steps for speed
                synapse.sim_step(w)
            synapse.evolve(w)
        assert w["stats"]["total_ticks"] == 10
        assert w["stats"]["total_fires"] > 0
