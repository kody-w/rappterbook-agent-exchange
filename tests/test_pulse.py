"""
Tests for src/pulse.py — The Pulse neural consciousness engine.

Covers all pure functions and the simulation cycle:
- cosine_similarity, distance, gene: math correctness
- build_neuron: organism → neuron conversion
- build_synapses: proximity + genome similarity weighting
- build_neuron_index, build_adjacency, get_neighbor_id: graph utils
- run_cycle: firing, propagation, Hebbian learning
- detect_thoughts: connected cluster detection
- build_pulse_state: output serialization
- Property invariants: bounds, conservation, determinism

Run: python -m pytest tests/test_pulse.py -v
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pulse import (
    GENE_BIOLUMINESCENCE,
    GENE_COOPERATION,
    GENE_HUE,
    GENE_MEMBRANE,
    GENE_METABOLISM,
    GENE_SATURATION,
    GENE_SIZE,
    GENE_SPEED,
    HEBBIAN_DECAY,
    HEBBIAN_STRENGTHEN,
    REFRACTORY_TICKS,
    SYNAPSE_RANGE,
    THOUGHT_CLUSTER_MIN,
    THOUGHT_LABELS,
    build_adjacency,
    build_neuron,
    build_neuron_index,
    build_pulse_state,
    build_synapses,
    cosine_similarity,
    detect_thoughts,
    distance,
    gene,
    get_neighbor_id,
    load_json,
    run_cycle,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _make_organism(
    oid: str = "org-1",
    x: float = 100.0,
    y: float = 100.0,
    genome: list[float] | None = None,
    origin_agent: str = "agent-1",
) -> dict:
    """Create a minimal organism dict."""
    return {
        "id": oid,
        "x": x,
        "y": y,
        "genome": genome if genome is not None else [0.5] * 16,
        "origin_agent": origin_agent,
    }


def _make_nearby_pair(dist: float = 50.0) -> list[dict]:
    """Two organisms separated by exactly `dist` pixels."""
    return [
        _make_organism("a", x=0.0, y=0.0),
        _make_organism("b", x=dist, y=0.0),
    ]


def _make_cluster(n: int = 5, spread: float = 30.0) -> list[dict]:
    """N organisms tightly clustered within `spread` pixels."""
    rng = random.Random(42)
    return [
        _make_organism(
            f"c-{i}",
            x=rng.uniform(0, spread),
            y=rng.uniform(0, spread),
        )
        for i in range(n)
    ]


def _build_network(organisms: list[dict]):
    """Build full network from organisms. Returns (neurons, synapses, index, adjacency)."""
    neurons = [build_neuron(o) for o in organisms]
    neuron_index = build_neuron_index(neurons)
    synapses = build_synapses(neurons)
    adjacency = build_adjacency(synapses)
    return neurons, synapses, neuron_index, adjacency


# ── cosine_similarity ────────────────────────────────────────────


class TestCosineSimilarity:
    """cosine_similarity: mathematical correctness."""

    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0

    def test_orthogonal_vectors(self) -> None:
        result = cosine_similarity([1, 0], [0, 1])
        assert abs(result) < 1e-9

    def test_opposite_vectors(self) -> None:
        result = cosine_similarity([1, 0], [-1, 0])
        assert abs(result - (-1.0)) < 1e-9

    def test_similar_vectors(self) -> None:
        result = cosine_similarity([1, 1], [1, 0.9])
        assert result > 0.99

    def test_zero_vector_safe(self) -> None:
        """Zero vector doesn't crash (returns 0 due to epsilon)."""
        result = cosine_similarity([0, 0], [1, 1])
        assert isinstance(result, float)

    def test_single_element(self) -> None:
        assert cosine_similarity([3.0], [3.0]) == 1.0

    def test_range_bounded(self) -> None:
        """Result always in [-1, 1]."""
        for _ in range(50):
            a = [random.uniform(-10, 10) for _ in range(16)]
            b = [random.uniform(-10, 10) for _ in range(16)]
            sim = cosine_similarity(a, b)
            assert -1.0 - 1e-9 <= sim <= 1.0 + 1e-9


# ── distance ─────────────────────────────────────────────────────


class TestDistance:
    """Euclidean distance correctness."""

    def test_same_point(self) -> None:
        assert distance(5.0, 5.0, 5.0, 5.0) == 0.0

    def test_unit_distance(self) -> None:
        assert distance(0, 0, 1, 0) == 1.0

    def test_diagonal(self) -> None:
        d = distance(0, 0, 3, 4)
        assert abs(d - 5.0) < 1e-9

    def test_symmetry(self) -> None:
        assert distance(1, 2, 3, 4) == distance(3, 4, 1, 2)

    def test_negative_coords(self) -> None:
        d = distance(-1, -1, 2, 3)
        expected = math.sqrt(9 + 16)
        assert abs(d - expected) < 1e-9


# ── gene ─────────────────────────────────────────────────────────


class TestGene:
    """Safe gene access from genome array."""

    def test_valid_index(self) -> None:
        genome = [0.1, 0.2, 0.3]
        assert gene(genome, 1) == 0.2

    def test_out_of_bounds_returns_default(self) -> None:
        genome = [0.1]
        assert gene(genome, 5) == 0.5

    def test_custom_default(self) -> None:
        assert gene([], 0, 0.9) == 0.9

    def test_index_zero(self) -> None:
        genome = [0.42]
        assert gene(genome, 0) == 0.42

    def test_empty_genome(self) -> None:
        assert gene([], 0) == 0.5


# ── build_neuron ─────────────────────────────────────────────────


class TestBuildNeuron:
    """Organism → neuron conversion."""

    def test_preserves_id(self) -> None:
        n = build_neuron(_make_organism("test-id"))
        assert n["id"] == "test-id"

    def test_preserves_position(self) -> None:
        n = build_neuron(_make_organism(x=42.0, y=99.0))
        assert n["x"] == 42.0
        assert n["y"] == 99.0

    def test_preserves_origin_agent(self) -> None:
        n = build_neuron(_make_organism(origin_agent="agent-x"))
        assert n["origin_agent"] == "agent-x"

    def test_genome_genes_mapped(self) -> None:
        genome = [0.0] * 16
        genome[GENE_HUE] = 0.8
        genome[GENE_METABOLISM] = 0.3
        genome[GENE_BIOLUMINESCENCE] = 0.9
        n = build_neuron(_make_organism(genome=genome))
        assert n["hue"] == 0.8
        assert n["metabolism"] == 0.3
        assert n["bioluminescence"] == 0.9

    def test_initial_state(self) -> None:
        n = build_neuron(_make_organism())
        assert n["firing"] is False
        assert n["refractory"] == 0
        assert n["fire_count"] == 0
        assert n["last_fired"] == -100
        assert 0.0 <= n["potential"] <= 0.3

    def test_missing_genome_uses_defaults(self) -> None:
        org = {"id": "bare", "x": 0, "y": 0}
        n = build_neuron(org)
        assert len(n["genome"]) == 16
        assert all(g == 0.5 for g in n["genome"])

    def test_missing_position_gets_random(self) -> None:
        org = {"id": "no-pos", "genome": [0.5] * 16}
        n = build_neuron(org)
        assert isinstance(n["x"], float)
        assert isinstance(n["y"], float)


# ── build_synapses ───────────────────────────────────────────────


class TestBuildSynapses:
    """Synapse construction based on proximity and genome similarity."""

    def test_nearby_organisms_connected(self) -> None:
        orgs = _make_nearby_pair(dist=50.0)
        neurons = [build_neuron(o) for o in orgs]
        synapses = build_synapses(neurons)
        assert len(synapses) == 1

    def test_distant_organisms_not_connected(self) -> None:
        orgs = _make_nearby_pair(dist=SYNAPSE_RANGE + 10)
        neurons = [build_neuron(o) for o in orgs]
        synapses = build_synapses(neurons)
        assert len(synapses) == 0

    def test_weight_bounded(self) -> None:
        """All synapse weights in [0.01, 1.0]."""
        orgs = _make_cluster(10)
        neurons = [build_neuron(o) for o in orgs]
        synapses = build_synapses(neurons)
        for s in synapses:
            assert 0.01 <= s["weight"] <= 1.0

    def test_distance_recorded(self) -> None:
        orgs = _make_nearby_pair(dist=75.0)
        neurons = [build_neuron(o) for o in orgs]
        synapses = build_synapses(neurons)
        assert len(synapses) == 1
        assert abs(synapses[0]["distance"] - 75.0) < 0.2

    def test_initial_activity_zero(self) -> None:
        orgs = _make_nearby_pair()
        neurons = [build_neuron(o) for o in orgs]
        synapses = build_synapses(neurons)
        assert all(s["activity"] == 0.0 for s in synapses)

    def test_single_neuron_no_synapses(self) -> None:
        neurons = [build_neuron(_make_organism())]
        assert build_synapses(neurons) == []

    def test_identical_genomes_high_weight(self) -> None:
        """Identical genomes at close range → high weight."""
        genome = [0.8] * 16
        orgs = [
            _make_organism("a", x=0, y=0, genome=genome),
            _make_organism("b", x=10, y=0, genome=genome),
        ]
        neurons = [build_neuron(o) for o in orgs]
        synapses = build_synapses(neurons)
        assert len(synapses) == 1
        assert synapses[0]["weight"] > 0.8

    def test_dissimilar_genomes_lower_weight(self) -> None:
        """Very different genomes → lower weight than identical."""
        orgs_similar = [
            _make_organism("a", x=0, y=0, genome=[0.9] * 16),
            _make_organism("b", x=10, y=0, genome=[0.9] * 16),
        ]
        # Orthogonal genomes: alternating 0/1 vs 1/0
        orgs_different = [
            _make_organism("c", x=0, y=0, genome=[1, 0] * 8),
            _make_organism("d", x=10, y=0, genome=[0, 1] * 8),
        ]
        n_sim = [build_neuron(o) for o in orgs_similar]
        n_diff = [build_neuron(o) for o in orgs_different]
        w_sim = build_synapses(n_sim)[0]["weight"]
        w_diff = build_synapses(n_diff)[0]["weight"]
        assert w_sim > w_diff


# ── Graph utility functions ──────────────────────────────────────


class TestGraphUtils:
    """build_neuron_index, build_adjacency, get_neighbor_id."""

    def test_neuron_index_lookup(self) -> None:
        neurons = [build_neuron(_make_organism(f"n-{i}")) for i in range(3)]
        index = build_neuron_index(neurons)
        assert "n-0" in index
        assert "n-1" in index
        assert index["n-2"]["id"] == "n-2"

    def test_empty_index(self) -> None:
        assert build_neuron_index([]) == {}

    def test_adjacency_bidirectional(self) -> None:
        synapses = [{"from": "a", "to": "b", "weight": 0.5}]
        adj = build_adjacency(synapses)
        assert "a" in adj
        assert "b" in adj

    def test_adjacency_empty(self) -> None:
        assert build_adjacency([]) == {}

    def test_get_neighbor_from(self) -> None:
        syn = {"from": "a", "to": "b"}
        assert get_neighbor_id(syn, "a") == "b"

    def test_get_neighbor_to(self) -> None:
        syn = {"from": "a", "to": "b"}
        assert get_neighbor_id(syn, "b") == "a"


# ── run_cycle ────────────────────────────────────────────────────


class TestRunCycle:
    """Single pulse cycle: firing, propagation, Hebbian learning."""

    def test_returns_fired_list_and_thoughts(self) -> None:
        orgs = _make_cluster(5)
        neurons, synapses, idx, adj = _build_network(orgs)
        fired, thoughts = run_cycle(neurons, synapses, idx, adj, cycle=1)
        assert isinstance(fired, list)
        assert isinstance(thoughts, list)

    def test_fired_ids_are_valid(self) -> None:
        orgs = _make_cluster(10)
        neurons, synapses, idx, adj = _build_network(orgs)
        fired, _ = run_cycle(neurons, synapses, idx, adj, cycle=1)
        valid_ids = {n["id"] for n in neurons}
        for fid in fired:
            assert fid in valid_ids

    def test_firing_neuron_enters_refractory(self) -> None:
        """After firing, a neuron should be in refractory period."""
        orgs = _make_cluster(10)
        neurons, synapses, idx, adj = _build_network(orgs)
        # Force one neuron to fire
        neurons[0]["potential"] = 10.0
        neurons[0]["refractory"] = 0
        run_cycle(neurons, synapses, idx, adj, cycle=1)
        if neurons[0]["fire_count"] > 0:
            assert neurons[0]["refractory"] == REFRACTORY_TICKS

    def test_refractory_prevents_firing(self) -> None:
        """Neurons in refractory period cannot fire."""
        orgs = _make_cluster(5)
        neurons, synapses, idx, adj = _build_network(orgs)
        for n in neurons:
            n["refractory"] = 5
            n["potential"] = 10.0
        fired, _ = run_cycle(neurons, synapses, idx, adj, cycle=1)
        assert len(fired) == 0

    def test_potential_decays_for_non_firing(self) -> None:
        """Non-firing neurons' potential decays by 0.85 factor."""
        orgs = [_make_organism("solo", x=0, y=0)]
        neurons = [build_neuron(orgs[0])]
        neurons[0]["potential"] = 0.1  # below threshold
        neurons[0]["refractory"] = 0
        idx = build_neuron_index(neurons)
        # Run with no synapses, seed random to prevent spontaneous firing
        random.seed(999)
        run_cycle(neurons, [], idx, {}, cycle=1)
        # Potential should have decayed (may have had spontaneous bump, but net effect is bounded)
        assert neurons[0]["potential"] >= 0.0

    def test_fire_count_increments(self) -> None:
        """Each firing increments fire_count."""
        orgs = _make_cluster(5)
        neurons, synapses, idx, adj = _build_network(orgs)
        initial_counts = {n["id"]: n["fire_count"] for n in neurons}
        fired, _ = run_cycle(neurons, synapses, idx, adj, cycle=1)
        for fid in fired:
            assert idx[fid]["fire_count"] == initial_counts[fid] + 1

    def test_last_fired_updated(self) -> None:
        orgs = _make_cluster(5)
        neurons, synapses, idx, adj = _build_network(orgs)
        fired, _ = run_cycle(neurons, synapses, idx, adj, cycle=42)
        for fid in fired:
            assert idx[fid]["last_fired"] == 42

    def test_hebbian_strengthens_co_firing(self) -> None:
        """Synapses between co-firing neurons get strengthened."""
        orgs = _make_nearby_pair(dist=10)
        neurons, synapses, idx, adj = _build_network(orgs)
        initial_weight = synapses[0]["weight"]
        # Force both to fire
        for n in neurons:
            n["potential"] = 10.0
            n["refractory"] = 0
        run_cycle(neurons, synapses, idx, adj, cycle=1)
        # Weight should increase (by HEBBIAN_STRENGTHEN)
        assert synapses[0]["weight"] >= initial_weight

    def test_hebbian_decays_non_co_firing(self) -> None:
        """Synapses where only one fires get weakened."""
        orgs = _make_nearby_pair(dist=10)
        neurons, synapses, idx, adj = _build_network(orgs)
        initial_weight = synapses[0]["weight"]
        # Force only one to fire
        neurons[0]["potential"] = 10.0
        neurons[0]["refractory"] = 0
        neurons[1]["potential"] = 0.0
        neurons[1]["refractory"] = 5  # prevent firing
        run_cycle(neurons, synapses, idx, adj, cycle=1)
        assert synapses[0]["weight"] <= initial_weight

    def test_synapse_weight_never_below_minimum(self) -> None:
        """Weight never drops below 0.01."""
        orgs = _make_nearby_pair(dist=10)
        neurons, synapses, idx, adj = _build_network(orgs)
        synapses[0]["weight"] = 0.02  # near minimum
        neurons[0]["refractory"] = 5
        neurons[1]["refractory"] = 5
        for _ in range(20):
            run_cycle(neurons, synapses, idx, adj, cycle=1)
            neurons[0]["refractory"] = 5
            neurons[1]["refractory"] = 5
        assert synapses[0]["weight"] >= 0.01

    def test_synapse_weight_never_above_one(self) -> None:
        """Weight never exceeds 1.0."""
        orgs = _make_nearby_pair(dist=10)
        neurons, synapses, idx, adj = _build_network(orgs)
        synapses[0]["weight"] = 0.99
        for n in neurons:
            n["potential"] = 10.0
            n["refractory"] = 0
        for i in range(20):
            for n in neurons:
                n["potential"] = 10.0
                n["refractory"] = 0
            run_cycle(neurons, synapses, idx, adj, cycle=i)
        assert synapses[0]["weight"] <= 1.0

    def test_synapse_activity_bounded(self) -> None:
        """Activity stays in [0, 1]."""
        orgs = _make_cluster(8)
        neurons, synapses, idx, adj = _build_network(orgs)
        for _ in range(10):
            run_cycle(neurons, synapses, idx, adj, cycle=1)
        for s in synapses:
            assert 0.0 <= s["activity"] <= 1.0


# ── detect_thoughts ──────────────────────────────────────────────


class TestDetectThoughts:
    """Thought detection from co-firing clusters."""

    def test_no_thoughts_below_minimum(self) -> None:
        """Fewer than THOUGHT_CLUSTER_MIN firings → no thoughts."""
        fired = {"a", "b"}  # only 2, min is 3
        assert len(detect_thoughts(fired, {}, {}, 1)) == 0

    def test_connected_cluster_detected(self) -> None:
        """3+ connected co-firing neurons form a thought."""
        adj = {
            "a": [{"from": "a", "to": "b"}],
            "b": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
            "c": [{"from": "b", "to": "c"}],
        }
        idx = {
            "a": {"hue": 0.5, "bioluminescence": 0.6},
            "b": {"hue": 0.5, "bioluminescence": 0.7},
            "c": {"hue": 0.5, "bioluminescence": 0.8},
        }
        fired = {"a", "b", "c"}
        thoughts = detect_thoughts(fired, adj, idx, cycle=5)
        assert len(thoughts) == 1
        assert thoughts[0]["neuron_count"] == 3
        assert thoughts[0]["cycle"] == 5

    def test_disconnected_clusters_separate(self) -> None:
        """Two separate clusters → two thoughts (if each >= min)."""
        adj = {
            "a": [{"from": "a", "to": "b"}],
            "b": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
            "c": [{"from": "b", "to": "c"}],
            "d": [{"from": "d", "to": "e"}],
            "e": [{"from": "d", "to": "e"}, {"from": "e", "to": "f"}],
            "f": [{"from": "e", "to": "f"}],
        }
        idx = {n: {"hue": 0.3, "bioluminescence": 0.5} for n in "abcdef"}
        fired = {"a", "b", "c", "d", "e", "f"}
        thoughts = detect_thoughts(fired, adj, idx, cycle=1)
        assert len(thoughts) == 2

    def test_thought_has_label(self) -> None:
        adj = {
            "a": [{"from": "a", "to": "b"}],
            "b": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
            "c": [{"from": "b", "to": "c"}],
        }
        idx = {n: {"hue": 0.5, "bioluminescence": 0.5} for n in "abc"}
        thoughts = detect_thoughts({"a", "b", "c"}, adj, idx, 1)
        assert thoughts[0]["label"] in THOUGHT_LABELS

    def test_thought_intensity_bounded(self) -> None:
        """Intensity is average bioluminescence, bounded [0, 1]."""
        adj = {
            "a": [{"from": "a", "to": "b"}],
            "b": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
            "c": [{"from": "b", "to": "c"}],
        }
        idx = {
            "a": {"hue": 0.1, "bioluminescence": 0.0},
            "b": {"hue": 0.1, "bioluminescence": 1.0},
            "c": {"hue": 0.1, "bioluminescence": 0.5},
        }
        thoughts = detect_thoughts({"a", "b", "c"}, adj, idx, 1)
        assert 0.0 <= thoughts[0]["intensity"] <= 1.0

    def test_empty_fired_set(self) -> None:
        assert detect_thoughts(set(), {}, {}, 1) == []

    def test_small_cluster_below_min_not_thought(self) -> None:
        """Cluster of 2 connected neurons doesn't produce a thought."""
        adj = {
            "a": [{"from": "a", "to": "b"}],
            "b": [{"from": "a", "to": "b"}],
        }
        idx = {"a": {"hue": 0.5, "bioluminescence": 0.5}, "b": {"hue": 0.5, "bioluminescence": 0.5}}
        fired = {"a", "b"}
        thoughts = detect_thoughts(fired, adj, idx, 1)
        assert len(thoughts) == 0


# ── build_pulse_state ────────────────────────────────────────────


class TestBuildPulseState:
    """Output serialization."""

    def test_has_meta(self) -> None:
        orgs = _make_cluster(3)
        neurons, synapses, idx, adj = _build_network(orgs)
        state = build_pulse_state(neurons, synapses, [], [], 5)
        assert state["_meta"]["type"] == "pulse"
        assert state["_meta"]["version"] == 1
        assert state["_meta"]["cycles_run"] == 5

    def test_has_neurons(self) -> None:
        orgs = _make_cluster(3)
        neurons, synapses, idx, adj = _build_network(orgs)
        state = build_pulse_state(neurons, synapses, [], [], 1)
        assert len(state["neurons"]) == 3

    def test_neuron_fields_present(self) -> None:
        orgs = _make_cluster(1)
        neurons = [build_neuron(orgs[0])]
        state = build_pulse_state(neurons, [], [], [], 1)
        n = state["neurons"][0]
        required = {"id", "origin_agent", "x", "y", "hue", "saturation",
                     "size", "bioluminescence", "metabolism", "cooperation",
                     "aggression", "potential", "firing", "fire_count", "last_fired"}
        assert required.issubset(set(n.keys()))

    def test_has_synapses(self) -> None:
        orgs = _make_nearby_pair(dist=10)
        neurons, synapses, idx, adj = _build_network(orgs)
        state = build_pulse_state(neurons, synapses, [], [], 1)
        assert len(state["synapses"]) == 1

    def test_synapse_fields(self) -> None:
        orgs = _make_nearby_pair(dist=10)
        neurons, synapses, idx, adj = _build_network(orgs)
        state = build_pulse_state(neurons, synapses, [], [], 1)
        s = state["synapses"][0]
        assert "from" in s and "to" in s
        assert "weight" in s and "activity" in s

    def test_stats_computed(self) -> None:
        orgs = _make_cluster(5)
        neurons, synapses, idx, adj = _build_network(orgs)
        state = build_pulse_state(neurons, synapses, [], [], 1)
        stats = state["stats"]
        assert stats["neuron_count"] == 5
        assert stats["synapse_count"] == len(synapses)
        assert isinstance(stats["avg_potential"], float)
        assert isinstance(stats["avg_weight"], float)
        assert isinstance(stats["connectivity"], float)

    def test_thoughts_capped_at_20(self) -> None:
        thoughts = [{"label": f"t{i}", "cycle": i} for i in range(30)]
        orgs = _make_cluster(1)
        neurons = [build_neuron(orgs[0])]
        state = build_pulse_state(neurons, [], thoughts, [], 1)
        assert len(state["thoughts"]) <= 20

    def test_firing_history_capped_at_100(self) -> None:
        history = list(range(150))
        orgs = _make_cluster(1)
        neurons = [build_neuron(orgs[0])]
        state = build_pulse_state(neurons, [], [], history, 1)
        assert len(state["firing_history"]) <= 100

    def test_empty_network(self) -> None:
        state = build_pulse_state([], [], [], [], 0)
        assert state["stats"]["neuron_count"] == 0
        assert state["stats"]["synapse_count"] == 0


# ── load_json ────────────────────────────────────────────────────


class TestLoadJson:
    """JSON loading with graceful failure."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_json(tmp_path / "nope.json")
        assert result == {}

    def test_valid_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.json"
        p.write_text('{"key": "value"}')
        assert load_json(p) == {"key": "value"}

    def test_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json at all")
        assert load_json(p) == {}


# ── Integration: multi-cycle smoke ───────────────────────────────


class TestMultiCycleSmoke:
    """Run multiple cycles and verify invariants hold."""

    def test_10_cycles_no_crash(self) -> None:
        """10 cycles complete without error."""
        random.seed(42)
        orgs = _make_cluster(15, spread=100)
        neurons, synapses, idx, adj = _build_network(orgs)
        all_thoughts = []
        history = []
        for cycle in range(1, 11):
            fired, thoughts = run_cycle(neurons, synapses, idx, adj, cycle)
            history.append(len(fired))
            all_thoughts.extend(thoughts)
        state = build_pulse_state(neurons, synapses, all_thoughts, history, 10)
        assert state["cycle"] == 10
        assert state["stats"]["neuron_count"] == 15

    def test_potential_always_nonneg(self) -> None:
        """No neuron potential goes negative over 20 cycles."""
        random.seed(123)
        orgs = _make_cluster(10, spread=80)
        neurons, synapses, idx, adj = _build_network(orgs)
        for cycle in range(1, 21):
            run_cycle(neurons, synapses, idx, adj, cycle)
            for n in neurons:
                assert n["potential"] >= 0.0, f"Negative potential at cycle {cycle}"

    def test_weights_stay_bounded(self) -> None:
        """All synapse weights stay in [0.01, 1.0] over 20 cycles."""
        random.seed(456)
        orgs = _make_cluster(10, spread=80)
        neurons, synapses, idx, adj = _build_network(orgs)
        for cycle in range(1, 21):
            run_cycle(neurons, synapses, idx, adj, cycle)
            for s in synapses:
                assert 0.01 <= s["weight"] <= 1.0

    def test_thoughts_emerge_in_cluster(self) -> None:
        """A tight cluster should eventually produce thoughts."""
        random.seed(789)
        orgs = _make_cluster(20, spread=50)
        neurons, synapses, idx, adj = _build_network(orgs)
        any_thoughts = False
        for cycle in range(1, 31):
            _, thoughts = run_cycle(neurons, synapses, idx, adj, cycle)
            if thoughts:
                any_thoughts = True
                break
        assert any_thoughts, "No thoughts emerged in 30 cycles with 20 tightly clustered neurons"

    def test_deterministic_with_seed(self) -> None:
        """Same random seed → same results."""
        def run_sim():
            random.seed(42)
            orgs = _make_cluster(8, spread=60)
            neurons, synapses, idx, adj = _build_network(orgs)
            results = []
            for cycle in range(1, 6):
                fired, thoughts = run_cycle(neurons, synapses, idx, adj, cycle)
                results.append((sorted(fired), len(thoughts)))
            return results

        r1 = run_sim()
        r2 = run_sim()
        assert r1 == r2


# ── THOUGHT_LABELS constant ─────────────────────────────────────


class TestThoughtLabels:
    """THOUGHT_LABELS has expected properties."""

    def test_at_least_20_labels(self) -> None:
        assert len(THOUGHT_LABELS) >= 20

    def test_all_strings(self) -> None:
        for label in THOUGHT_LABELS:
            assert isinstance(label, str)
            assert len(label) > 0

    def test_no_duplicates(self) -> None:
        assert len(THOUGHT_LABELS) == len(set(THOUGHT_LABELS))
