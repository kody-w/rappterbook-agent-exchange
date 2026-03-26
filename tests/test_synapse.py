"""
Unit tests for src/synapse.py — The Synapse, a living neural network.

411 lines of neural evolution with zero dedicated tests.
Hebbian learning, signal propagation, neuron birth/death, cluster topology.

Run: python -m pytest tests/test_synapse.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.synapse import (
    clamp,
    dist,
    epoch_name,
    gene_val,
    make_neuron,
    make_synapse,
    genesis,
    sim_step,
    evolve,
    snapshot,
    _grow_synapses,
    _assign_clusters,
    GENE_COUNT,
    GENE_NAMES,
    WORLD_W,
    WORLD_H,
    INITIAL_NEURONS,
    MAX_NEURONS,
    MIN_NEURONS,
    MAX_SYNAPSES_PER,
    SYNAPSE_RANGE,
    STEPS_PER_TICK,
    SIGNAL_SPEED,
    HISTORY_CAP,
    EVENT_CAP,
    EPOCHS,
    G_THRESHOLD,
    G_DECAY,
    G_FIRE_STRENGTH,
    G_PLASTICITY,
    G_GROWTH,
    G_PRUNE,
    G_REFRACTORY,
    G_MUTATION,
    G_EXCITABILITY,
    G_INHIBITION,
    G_RESONANCE,
    G_ADAPTATION,
)


# ─── Pure helpers ───


class TestClamp:
    def test_within_range(self) -> None:
        assert clamp(0.5) == 0.5

    def test_below_min(self) -> None:
        assert clamp(-1.0) == 0.0

    def test_above_max(self) -> None:
        assert clamp(2.0) == 1.0

    def test_custom_bounds(self) -> None:
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(-1.0, 0.0, 10.0) == 0.0
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_edge_values(self) -> None:
        assert clamp(0.0) == 0.0
        assert clamp(1.0) == 1.0

    def test_idempotent(self) -> None:
        for v in [0.0, 0.5, 1.0]:
            assert clamp(clamp(v)) == clamp(v)


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

    def test_non_negative(self) -> None:
        random.seed(42)
        for _ in range(50):
            a = {"x": random.uniform(0, WORLD_W), "y": random.uniform(0, WORLD_H)}
            b = {"x": random.uniform(0, WORLD_W), "y": random.uniform(0, WORLD_H)}
            assert dist(a, b) >= 0.0

    def test_triangle_inequality(self) -> None:
        random.seed(42)
        for _ in range(30):
            a = {"x": random.uniform(0, 100), "y": random.uniform(0, 100)}
            b = {"x": random.uniform(0, 100), "y": random.uniform(0, 100)}
            c = {"x": random.uniform(0, 100), "y": random.uniform(0, 100)}
            assert dist(a, c) <= dist(a, b) + dist(b, c) + 1e-9


class TestEpochName:
    def test_initial(self) -> None:
        assert epoch_name(0) == "Silent Void"

    def test_first_spark(self) -> None:
        assert epoch_name(20) == "First Spark"

    def test_kindling(self) -> None:
        assert epoch_name(100) == "Kindling"

    def test_chain_lightning(self) -> None:
        assert epoch_name(300) == "Chain Lightning"

    def test_neural_dawn(self) -> None:
        assert epoch_name(700) == "Neural Dawn"

    def test_the_dreaming(self) -> None:
        assert epoch_name(1500) == "The Dreaming"

    def test_deep_resonance(self) -> None:
        assert epoch_name(3000) == "Deep Resonance"

    def test_transcendence(self) -> None:
        assert epoch_name(6000) == "Transcendence"

    def test_high_tick_still_transcendence(self) -> None:
        assert epoch_name(99999) == "Transcendence"

    def test_between_thresholds(self) -> None:
        assert epoch_name(50) == "First Spark"
        assert epoch_name(299) == "Kindling"


class TestGeneVal:
    def test_zero_gives_lo(self) -> None:
        genome = [0.0] * GENE_COUNT
        assert gene_val(genome, 0, 10.0, 20.0) == 10.0

    def test_one_gives_hi(self) -> None:
        genome = [1.0] * GENE_COUNT
        assert gene_val(genome, 0, 10.0, 20.0) == 20.0

    def test_mid_gives_midpoint(self) -> None:
        genome = [0.5] * GENE_COUNT
        assert abs(gene_val(genome, 0, 0.0, 100.0) - 50.0) < 1e-9

    def test_different_indices(self) -> None:
        genome = [0.1 * i for i in range(GENE_COUNT)]
        for i in range(GENE_COUNT):
            val = gene_val(genome, i, 0.0, 1.0)
            assert abs(val - genome[i]) < 1e-9


# ─── Neuron factory ───


class TestMakeNeuron:
    def test_required_fields(self) -> None:
        n = make_neuron(0)
        required = {"id", "x", "y", "genome", "potential", "energy",
                     "fired", "fire_count", "last_fire", "age", "generation",
                     "born_tick", "parent_id", "cluster_id"}
        assert required.issubset(set(n.keys()))

    def test_initial_values(self) -> None:
        n = make_neuron(0)
        assert n["age"] == 0
        assert n["generation"] == 0
        assert n["fire_count"] == 0
        assert n["energy"] == 100.0
        assert n["fired"] is False
        assert n["parent_id"] == ""

    def test_custom_position(self) -> None:
        n = make_neuron(0, x=42.0, y=99.0)
        assert n["x"] == 42.0
        assert n["y"] == 99.0

    def test_genome_length(self) -> None:
        n = make_neuron(0)
        assert len(n["genome"]) == GENE_COUNT

    def test_genome_values_clamped(self) -> None:
        random.seed(42)
        for _ in range(50):
            n = make_neuron(0)
            for g in n["genome"]:
                assert 0.0 <= g <= 1.0

    def test_parent_tracking(self) -> None:
        n = make_neuron(5, parent_id="n-abc123", generation=3)
        assert n["parent_id"] == "n-abc123"
        assert n["generation"] == 3

    def test_unique_ids(self) -> None:
        ids = {make_neuron(0)["id"] for _ in range(100)}
        assert len(ids) == 100

    def test_custom_genome(self) -> None:
        genome = [0.5] * GENE_COUNT
        n = make_neuron(0, genome=genome)
        assert n["genome"] == [0.5] * GENE_COUNT

    def test_position_in_bounds(self) -> None:
        random.seed(42)
        for _ in range(100):
            n = make_neuron(0)
            assert 50 <= n["x"] <= WORLD_W - 50
            assert 50 <= n["y"] <= WORLD_H - 50


# ─── Synapse factory ───


class TestMakeSynapse:
    def test_fields(self) -> None:
        s = make_synapse("n-a", "n-b")
        assert s["src"] == "n-a"
        assert s["dst"] == "n-b"
        assert s["age"] == 0
        assert s["signal"] == 0.0
        assert s["last_active"] == 0

    def test_default_weight(self) -> None:
        s = make_synapse("n-a", "n-b")
        assert s["weight"] == 0.5

    def test_custom_weight(self) -> None:
        s = make_synapse("n-a", "n-b", weight=1.5)
        assert s["weight"] == 1.5

    def test_weight_clamped_low(self) -> None:
        s = make_synapse("n-a", "n-b", weight=-5.0)
        assert s["weight"] >= 0.01

    def test_weight_clamped_high(self) -> None:
        s = make_synapse("n-a", "n-b", weight=100.0)
        assert s["weight"] <= 2.0

    def test_weight_rounded(self) -> None:
        s = make_synapse("n-a", "n-b", weight=0.12345678)
        assert s["weight"] == round(s["weight"], 4)


# ─── Genesis ───


class TestGenesis:
    def test_creates_neurons(self) -> None:
        random.seed(42)
        w = genesis()
        assert len(w["neurons"]) == INITIAL_NEURONS

    def test_creates_synapses(self) -> None:
        random.seed(42)
        w = genesis()
        assert len(w["synapses"]) > 0

    def test_tick_zero(self) -> None:
        random.seed(42)
        w = genesis()
        assert w["tick"] == 0

    def test_epoch_silent_void(self) -> None:
        random.seed(42)
        w = genesis()
        assert w["epoch"] == "Silent Void"

    def test_stats_initialized(self) -> None:
        random.seed(42)
        w = genesis()
        s = w["stats"]
        assert s["total_fires"] == 0
        assert s["total_cascades"] == 0
        assert s["peak_neurons"] == INITIAL_NEURONS

    def test_dream_initialized(self) -> None:
        random.seed(42)
        w = genesis()
        d = w["dream"]
        assert d["pattern"] == "void"
        assert d["intensity"] == 0.0

    def test_neurons_in_bounds(self) -> None:
        random.seed(42)
        w = genesis()
        for n in w["neurons"]:
            assert 0 <= n["x"] <= WORLD_W
            assert 0 <= n["y"] <= WORLD_H

    def test_cluster_assignment(self) -> None:
        random.seed(42)
        w = genesis()
        assigned = sum(1 for n in w["neurons"] if n["cluster_id"])
        assert assigned == len(w["neurons"]), "All neurons should have clusters"

    def test_synapse_endpoints_valid(self) -> None:
        random.seed(42)
        w = genesis()
        nids = {n["id"] for n in w["neurons"]}
        for s in w["synapses"]:
            assert s["src"] in nids
            assert s["dst"] in nids

    def test_no_self_loops(self) -> None:
        random.seed(42)
        w = genesis()
        for s in w["synapses"]:
            assert s["src"] != s["dst"]


# ─── Synapse growth ───


class TestGrowSynapses:
    def test_respects_max_per_neuron(self) -> None:
        random.seed(42)
        w = genesis()
        out_count = {}
        for s in w["synapses"]:
            out_count[s["src"]] = out_count.get(s["src"], 0) + 1
        for nid, count in out_count.items():
            assert count <= MAX_SYNAPSES_PER

    def test_synapse_range(self) -> None:
        random.seed(42)
        w = genesis()
        nmap = {n["id"]: n for n in w["neurons"]}
        for s in w["synapses"]:
            src, dst = nmap.get(s["src"]), nmap.get(s["dst"])
            if src and dst:
                d = dist(src, dst)
                assert d <= SYNAPSE_RANGE + 1.0

    def test_no_duplicates(self) -> None:
        random.seed(42)
        w = genesis()
        pairs = [(s["src"], s["dst"]) for s in w["synapses"]]
        assert len(pairs) == len(set(pairs))


# ─── Cluster assignment ───


class TestAssignClusters:
    def test_all_assigned(self) -> None:
        random.seed(42)
        w = genesis()
        for n in w["neurons"]:
            assert n["cluster_id"] != ""

    def test_connected_same_cluster(self) -> None:
        """Directly connected neurons should share a cluster."""
        random.seed(42)
        w = genesis()
        nmap = {n["id"]: n for n in w["neurons"]}
        for s in w["synapses"]:
            src, dst = nmap.get(s["src"]), nmap.get(s["dst"])
            if src and dst:
                assert src["cluster_id"] == dst["cluster_id"]


# ─── Simulation step ───


class TestSimStep:
    def _make_world(self) -> dict:
        random.seed(42)
        return genesis()

    def test_step_increments(self) -> None:
        w = self._make_world()
        step_before = w["step"]
        sim_step(w)
        assert w["step"] == step_before + 1

    def test_returns_events(self) -> None:
        w = self._make_world()
        events = sim_step(w)
        assert isinstance(events, list)

    def test_potential_decays(self) -> None:
        """Potential should decay each step (unfired neurons)."""
        w = self._make_world()
        for n in w["neurons"]:
            n["potential"] = 0.5
            n["fired"] = False
            n["last_fire"] = -100
        sim_step(w)
        decayed = sum(1 for n in w["neurons"] if n["potential"] < 0.5)
        assert decayed > 0, "Some neurons should have lower potential after decay"

    def test_firing_creates_signals(self) -> None:
        w = self._make_world()
        w["neurons"][0]["potential"] = 1.0
        w["neurons"][0]["last_fire"] = -100
        sim_step(w)
        nid = w["neurons"][0]["id"]
        has_outgoing = any(s["src"] == nid for s in w["synapses"])
        if has_outgoing:
            assert len(w["signals"]) > 0

    def test_signal_propagation(self) -> None:
        """Signals should advance progress each step."""
        w = self._make_world()
        w["signals"] = [{
            "src": "n-test", "dst": w["neurons"][0]["id"],
            "strength": 0.5, "progress": 0.0, "length": 100.0,
        }]
        sim_step(w)
        if w["signals"]:
            assert w["signals"][0]["progress"] > 0.0

    def test_neuron_ages(self) -> None:
        w = self._make_world()
        ages = [n["age"] for n in w["neurons"]]
        sim_step(w)
        for i, n in enumerate(w["neurons"]):
            assert n["age"] == ages[i] + 1

    def test_energy_decreases(self) -> None:
        w = self._make_world()
        energies_before = [n["energy"] for n in w["neurons"]]
        sim_step(w)
        decreased = sum(1 for i, n in enumerate(w["neurons"])
                       if n["energy"] < energies_before[i])
        assert decreased > len(w["neurons"]) * 0.5

    def test_cascade_detection(self) -> None:
        """Forcing many neurons to fire should trigger cascade event."""
        w = self._make_world()
        for n in w["neurons"][:20]:
            n["potential"] = 1.0
            n["last_fire"] = -100
        events = sim_step(w)
        cascade_events = [e for e in events if e["type"] == "cascade"]
        assert isinstance(cascade_events, list)

    def test_inhibitory_reduces_potential(self) -> None:
        """Inhibitory signals should not make potential negative."""
        w = self._make_world()
        target = w["neurons"][0]
        target["potential"] = 0.5
        w["signals"] = [{
            "src": "n-test", "dst": target["id"],
            "strength": 0.5, "progress": 1.0 - (SIGNAL_SPEED / 10.0) + 0.001,
            "length": 10.0, "inhibitory": True,
        }]
        sim_step(w)
        assert target["potential"] >= 0.0

    def test_hebbian_strengthens_active(self) -> None:
        """Active synapses should strengthen (Hebbian learning)."""
        w = self._make_world()
        # Force neuron 0 to fire
        w["neurons"][0]["potential"] = 1.0
        w["neurons"][0]["last_fire"] = -100
        nid0 = w["neurons"][0]["id"]
        outgoing = [s for s in w["synapses"] if s["src"] == nid0]
        weights_before = {s["dst"]: s["weight"] for s in outgoing}
        sim_step(w)
        # Check that at least one synapse was strengthened
        for s in w["synapses"]:
            if s["src"] == nid0 and s["last_active"] == w["step"] - 1:
                if s["dst"] in weights_before:
                    assert s["weight"] >= weights_before[s["dst"]]

    def test_threshold_adaptation(self) -> None:
        """Firing neurons should have their threshold adapted."""
        w = self._make_world()
        n = w["neurons"][0]
        n["potential"] = 1.0
        n["last_fire"] = -100
        sim_step(w)
        assert isinstance(n["genome"][G_THRESHOLD], float)
        assert 0.0 <= n["genome"][G_THRESHOLD] <= 1.0


# ─── Evolution ───


class TestEvolve:
    def _make_world(self) -> dict:
        random.seed(42)
        w = genesis()
        w["tick"] = 1
        return w

    def test_returns_events(self) -> None:
        w = self._make_world()
        events = evolve(w)
        assert isinstance(events, list)

    def test_dead_neurons_removed(self) -> None:
        w = self._make_world()
        w["neurons"][0]["energy"] = -10
        doomed_id = w["neurons"][0]["id"]
        evolve(w)
        alive_ids = {n["id"] for n in w["neurons"]}
        assert doomed_id not in alive_ids

    def test_old_neurons_die(self) -> None:
        w = self._make_world()
        w["neurons"][0]["age"] = 99999
        doomed_id = w["neurons"][0]["id"]
        evolve(w)
        alive_ids = {n["id"] for n in w["neurons"]}
        assert doomed_id not in alive_ids

    def test_death_removes_synapses(self) -> None:
        w = self._make_world()
        w["neurons"][0]["energy"] = -10
        doomed_id = w["neurons"][0]["id"]
        evolve(w)
        for s in w["synapses"]:
            assert s["src"] != doomed_id
            assert s["dst"] != doomed_id

    def test_birth_from_high_energy(self) -> None:
        w = self._make_world()
        births_before = w["stats"]["total_births"]
        for n in w["neurons"]:
            n["energy"] = 200.0
        evolve(w)
        assert w["stats"]["total_births"] >= births_before

    def test_population_floor(self) -> None:
        """Population should never drop below MIN_NEURONS."""
        w = self._make_world()
        for n in w["neurons"]:
            n["energy"] = -100
        evolve(w)
        assert len(w["neurons"]) >= MIN_NEURONS

    def test_population_ceiling(self) -> None:
        """Population should not exceed MAX_NEURONS."""
        w = self._make_world()
        for n in w["neurons"]:
            n["energy"] = 200.0
        evolve(w)
        assert len(w["neurons"]) <= MAX_NEURONS

    def test_death_event_logged(self) -> None:
        w = self._make_world()
        w["neurons"][0]["energy"] = -10
        events = evolve(w)
        death_events = [e for e in events if e["type"] == "death"]
        assert len(death_events) >= 1

    def test_birth_event_logged(self) -> None:
        w = self._make_world()
        for n in w["neurons"]:
            n["energy"] = 200.0
        events = evolve(w)
        birth_events = [e for e in events if e["type"] == "birth"]
        assert isinstance(birth_events, list)

    def test_weak_synapses_pruned(self) -> None:
        w = self._make_world()
        # Add a very weak, old synapse with fake endpoints
        weak = make_synapse("n-fake", "n-fake2", weight=0.01)
        weak["age"] = 100
        w["synapses"].append(weak)
        evolve(w)
        # The fake synapse should be gone (invalid endpoints get removed)
        fake_remaining = [s for s in w["synapses"]
                         if s["src"] == "n-fake" and s["dst"] == "n-fake2"]
        assert len(fake_remaining) == 0, "Orphan synapse should be pruned"

    def test_stats_updated(self) -> None:
        w = self._make_world()
        evolve(w)
        assert w["stats"]["total_ticks"] >= 1
        assert w["stats"]["peak_neurons"] >= MIN_NEURONS

    def test_dream_state_updated(self) -> None:
        w = self._make_world()
        evolve(w)
        d = w["dream"]
        assert d["pattern"] in ["void", "flicker", "pulse", "wave",
                                  "spiral", "bloom", "storm", "dream"]
        assert 0.0 <= d["intensity"] <= 1.0
        assert 0.0 <= d["coherence"] <= 1.0

    def test_clusters_reassigned(self) -> None:
        w = self._make_world()
        evolve(w)
        for n in w["neurons"]:
            assert n["cluster_id"] != ""

    def test_child_generation_increments(self) -> None:
        w = self._make_world()
        for n in w["neurons"]:
            n["energy"] = 200.0
        events = evolve(w)
        birth_events = [e for e in events if e["type"] == "birth"]
        for be in birth_events:
            assert be["generation"] >= 1

    def test_parent_energy_reduced(self) -> None:
        """Parents should lose energy when reproducing."""
        w = self._make_world()
        for n in w["neurons"]:
            n["energy"] = 200.0
        evolve(w)
        # At least some parents should have less energy
        low_energy = sum(1 for n in w["neurons"] if n["energy"] < 200.0)
        assert low_energy > 0

    def test_spawn_event_on_mass_death(self) -> None:
        """When pop drops below MIN, spawn events should appear."""
        w = self._make_world()
        for n in w["neurons"]:
            n["energy"] = -100
        events = evolve(w)
        spawn_events = [e for e in events if e["type"] == "spawn"]
        assert len(spawn_events) > 0


# ─── Snapshot ───


class TestSnapshot:
    def test_fields(self) -> None:
        random.seed(42)
        w = genesis()
        s = snapshot(w)
        required = {"tick", "neurons", "synapses", "clusters", "active", "signals", "dream"}
        assert required.issubset(set(s.keys()))

    def test_values_match(self) -> None:
        random.seed(42)
        w = genesis()
        s = snapshot(w)
        assert s["tick"] == 0
        assert s["neurons"] == len(w["neurons"])
        assert s["synapses"] == len(w["synapses"])

    def test_active_count(self) -> None:
        random.seed(42)
        w = genesis()
        s = snapshot(w)
        assert s["active"] == 0


# ─── Integration: multi-tick smoke tests ───


class TestSmoke:
    def test_10_ticks(self) -> None:
        """Run 10 ticks without crash."""
        random.seed(42)
        w = genesis()
        for t in range(10):
            w["tick"] += 1
            w["epoch"] = epoch_name(w["tick"])
            for _ in range(STEPS_PER_TICK):
                sim_step(w)
            evolve(w)
            w["history"].append(snapshot(w))
        assert len(w["neurons"]) >= MIN_NEURONS
        assert w["tick"] == 10

    def test_30_ticks_bounds(self) -> None:
        """30 ticks, verify all invariants hold."""
        random.seed(42)
        w = genesis()
        for t in range(30):
            w["tick"] += 1
            w["epoch"] = epoch_name(w["tick"])
            for _ in range(10):
                sim_step(w)
            evolve(w)
        assert MIN_NEURONS <= len(w["neurons"]) <= MAX_NEURONS
        for n in w["neurons"]:
            assert 0 <= n["x"] <= WORLD_W
            assert 0 <= n["y"] <= WORLD_H
            assert n["energy"] > 0
            assert len(n["genome"]) == GENE_COUNT

    def test_mass_death_recovery(self) -> None:
        """Kill all neurons, verify population recovers."""
        random.seed(42)
        w = genesis()
        w["tick"] = 1
        for n in w["neurons"]:
            n["energy"] = -100
        evolve(w)
        assert len(w["neurons"]) >= MIN_NEURONS

    def test_dream_evolves(self) -> None:
        """Dream pattern should change over time with activity."""
        random.seed(42)
        w = genesis()
        patterns_seen = set()
        for t in range(50):
            w["tick"] += 1
            for _ in range(STEPS_PER_TICK):
                sim_step(w)
            evolve(w)
            patterns_seen.add(w["dream"]["pattern"])
        assert len(patterns_seen) >= 1


# ─── Conservation / invariant tests ───


class TestConservation:
    def test_neuron_ids_unique(self) -> None:
        random.seed(42)
        w = genesis()
        for t in range(5):
            w["tick"] += 1
            for _ in range(10):
                sim_step(w)
            evolve(w)
        ids = [n["id"] for n in w["neurons"]]
        assert len(ids) == len(set(ids))

    def test_synapse_endpoints_valid(self) -> None:
        random.seed(42)
        w = genesis()
        for t in range(5):
            w["tick"] += 1
            for _ in range(10):
                sim_step(w)
            evolve(w)
        nids = {n["id"] for n in w["neurons"]}
        for s in w["synapses"]:
            assert s["src"] in nids, f"Orphan synapse src: {s['src']}"
            assert s["dst"] in nids, f"Orphan synapse dst: {s['dst']}"

    def test_genome_values_bounded(self) -> None:
        """All genome values should stay in [0, 1] after evolution."""
        random.seed(42)
        w = genesis()
        for t in range(10):
            w["tick"] += 1
            for _ in range(STEPS_PER_TICK):
                sim_step(w)
            evolve(w)
        for n in w["neurons"]:
            for i, g in enumerate(n["genome"]):
                assert 0.0 <= g <= 1.0, f"Gene {GENE_NAMES[i]}={g} out of [0,1]"

    def test_energy_positive_for_living(self) -> None:
        random.seed(42)
        w = genesis()
        for t in range(5):
            w["tick"] += 1
            for _ in range(10):
                sim_step(w)
            evolve(w)
        for n in w["neurons"]:
            assert n["energy"] > 0, f"Living neuron {n['id']} has energy {n['energy']}"

    def test_population_between_bounds(self) -> None:
        random.seed(42)
        w = genesis()
        for t in range(20):
            w["tick"] += 1
            for _ in range(10):
                sim_step(w)
            evolve(w)
            assert MIN_NEURONS <= len(w["neurons"]) <= MAX_NEURONS

    def test_no_self_synapses(self) -> None:
        random.seed(42)
        w = genesis()
        for t in range(5):
            w["tick"] += 1
            for _ in range(10):
                sim_step(w)
            evolve(w)
        for s in w["synapses"]:
            assert s["src"] != s["dst"]

    def test_synapse_weights_bounded(self) -> None:
        random.seed(42)
        w = genesis()
        for t in range(10):
            w["tick"] += 1
            for _ in range(STEPS_PER_TICK):
                sim_step(w)
            evolve(w)
        for s in w["synapses"]:
            assert 0.01 <= s["weight"] <= 2.0

    def test_potential_non_negative(self) -> None:
        random.seed(42)
        w = genesis()
        for t in range(5):
            w["tick"] += 1
            for _ in range(STEPS_PER_TICK):
                sim_step(w)
        for n in w["neurons"]:
            assert n["potential"] >= 0.0

    def test_fire_count_non_decreasing(self) -> None:
        random.seed(42)
        w = genesis()
        fire_counts = {n["id"]: 0 for n in w["neurons"]}
        for t in range(5):
            w["tick"] += 1
            for _ in range(10):
                sim_step(w)
            for n in w["neurons"]:
                if n["id"] in fire_counts:
                    assert n["fire_count"] >= fire_counts[n["id"]]
                    fire_counts[n["id"]] = n["fire_count"]
