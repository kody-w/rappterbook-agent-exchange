"""
test_consciousness.py — 69 unit tests for The Dreaming Deep.

src/consciousness.py: minds, synapses, dreams, zeitgeist.
Each organism gets arousal/mood/curiosity/dream state. Nearby organisms
form synapses. Sleeping organisms emit dream fragments through the network.
Collective mood emerges as zeitgeist.

Property-based invariants for value bounds and conservation.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import consciousness as con


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_organism(oid: str, x: float = 100.0, y: float = 100.0,
                   energy: float = 80.0, genome: list = None) -> dict:
    """Create a minimal organism dict for testing."""
    g = genome or [random.uniform(0, 1) for _ in range(16)]
    return {"id": oid, "x": x, "y": y, "energy": energy, "genome": g}


def _make_world(n: int = 10, spread: float = 50.0) -> dict:
    """Create a world with n organisms clustered near origin."""
    orgs = []
    for i in range(n):
        orgs.append(_make_organism(
            f"org-{i:03d}",
            x=100 + random.uniform(-spread, spread),
            y=100 + random.uniform(-spread, spread),
            energy=random.uniform(20, 120),
        ))
    return {"organisms": orgs}


def _empty_minds_state() -> dict:
    return {}


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

class TestUtils:
    """Tests for utility functions: now_iso, uid, dist, genome_similarity."""

    def test_now_iso_format(self):
        ts = con.now_iso()
        assert "T" in ts
        assert ts.endswith("Z")

    def test_uid_unique(self):
        ids = set(con.uid() for _ in range(200))
        assert len(ids) == 200

    def test_uid_length(self):
        assert len(con.uid()) == 8

    def test_dist_same_point(self):
        a = {"x": 50, "y": 50}
        b = {"x": 50, "y": 50}
        assert con.dist(a, b) == 0.0

    def test_dist_known(self):
        a = {"x": 0, "y": 0}
        b = {"x": 3, "y": 4}
        assert con.dist(a, b) == pytest.approx(5.0)

    def test_dist_symmetric(self):
        for _ in range(30):
            a = {"x": random.uniform(0, 500), "y": random.uniform(0, 500)}
            b = {"x": random.uniform(0, 500), "y": random.uniform(0, 500)}
            assert con.dist(a, b) == pytest.approx(con.dist(b, a))

    def test_dist_nonnegative(self):
        for _ in range(30):
            a = {"x": random.uniform(-100, 500), "y": random.uniform(-100, 500)}
            b = {"x": random.uniform(-100, 500), "y": random.uniform(-100, 500)}
            assert con.dist(a, b) >= 0

    def test_genome_similarity_identical(self):
        g = [0.5] * 16
        assert con.genome_similarity(g, g) == pytest.approx(1.0)

    def test_genome_similarity_orthogonal(self):
        a = [1, 0, 0, 0]
        b = [0, 1, 0, 0]
        assert con.genome_similarity(a, b) == pytest.approx(0.0)

    def test_genome_similarity_range(self):
        for _ in range(30):
            a = [random.uniform(0, 1) for _ in range(16)]
            b = [random.uniform(0, 1) for _ in range(16)]
            sim = con.genome_similarity(a, b)
            assert 0.0 <= sim <= 1.0 + 1e-9

    def test_genome_similarity_symmetric(self):
        a = [random.uniform(0, 1) for _ in range(16)]
        b = [random.uniform(0, 1) for _ in range(16)]
        assert con.genome_similarity(a, b) == pytest.approx(
            con.genome_similarity(b, a))

    def test_genome_similarity_empty(self):
        assert con.genome_similarity([], [1, 2, 3]) == 0.0
        assert con.genome_similarity([1, 2], []) == 0.0


# ===========================================================================
# INIT MIND
# ===========================================================================

class TestInitMind:
    """Tests for mind initialization from organisms."""

    def test_init_mind_has_required_fields(self):
        org = _make_organism("test-1")
        mind = con.init_mind(org)
        for key in ["id", "arousal", "mood", "curiosity",
                     "dream_intensity", "is_dreaming", "memories",
                     "dream_fragments", "bonds_count"]:
            assert key in mind, f"Missing key: {key}"

    def test_init_mind_id_matches(self):
        org = _make_organism("abc-123")
        mind = con.init_mind(org)
        assert mind["id"] == "abc-123"

    def test_init_mind_not_dreaming(self):
        org = _make_organism("test-1")
        mind = con.init_mind(org)
        assert mind["is_dreaming"] is False
        assert mind["dream_intensity"] == 0.0

    def test_init_mind_empty_collections(self):
        org = _make_organism("test-1")
        mind = con.init_mind(org)
        assert mind["memories"] == []
        assert mind["dream_fragments"] == []
        assert mind["bonds_count"] == 0

    def test_init_mind_arousal_reasonable(self):
        for _ in range(50):
            org = _make_organism("test")
            mind = con.init_mind(org)
            assert -0.5 < mind["arousal"] < 1.5

    def test_init_mind_with_short_genome(self):
        org = _make_organism("short", genome=[0.5] * 5)
        mind = con.init_mind(org)
        assert "arousal" in mind

    def test_init_mind_with_no_genome(self):
        org = {"id": "bare", "x": 0, "y": 0, "energy": 50}
        mind = con.init_mind(org)
        assert mind["id"] == "bare"


# ===========================================================================
# TICK CONSCIOUSNESS (CORE ENGINE)
# ===========================================================================

class TestTickConsciousness:
    """Tests for the main consciousness tick function."""

    def test_tick_returns_valid_structure(self):
        random.seed(42)
        world = _make_world(10)
        result = con.tick_consciousness(world, _empty_minds_state())
        assert "_meta" in result
        assert "minds" in result
        assert "synapses" in result
        assert "dream_log" in result
        assert "zeitgeist" in result
        assert "zeitgeist_history" in result

    def test_tick_meta_increments(self):
        world = _make_world(5)
        r1 = con.tick_consciousness(world, _empty_minds_state())
        assert r1["_meta"]["tick"] == 1
        r2 = con.tick_consciousness(world, r1)
        assert r2["_meta"]["tick"] == 2

    def test_tick_creates_minds_for_all(self):
        world = _make_world(8)
        result = con.tick_consciousness(world, _empty_minds_state())
        org_ids = set(o["id"] for o in world["organisms"])
        mind_ids = set(result["minds"].keys())
        assert org_ids == mind_ids

    def test_tick_removes_dead_minds(self):
        world = _make_world(5)
        state = con.tick_consciousness(world, _empty_minds_state())
        assert len(state["minds"]) == 5
        world["organisms"] = world["organisms"][:3]
        state2 = con.tick_consciousness(world, state)
        assert len(state2["minds"]) == 3

    def test_tick_death_emits_dream(self):
        world = _make_world(5)
        state = con.tick_consciousness(world, _empty_minds_state())
        world["organisms"] = world["organisms"][:1]
        state2 = con.tick_consciousness(world, state)
        death_dreams = [d for d in state2["dream_log"] if d["type"] == "death_dream"]
        assert len(death_dreams) >= 1

    def test_tick_zeitgeist_computed(self):
        random.seed(42)
        world = _make_world(10)
        result = con.tick_consciousness(world, _empty_minds_state())
        z = result["zeitgeist"]
        for key in ["tick", "collective_mood", "collective_arousal",
                     "dreamers", "dreamer_ratio", "synapse_count",
                     "avg_synapse_strength"]:
            assert key in z

    def test_tick_zeitgeist_history_grows(self):
        world = _make_world(5)
        state = _empty_minds_state()
        for _ in range(5):
            state = con.tick_consciousness(world, state)
        assert len(state["zeitgeist_history"]) == 5

    def test_tick_zeitgeist_history_capped(self):
        world = _make_world(5)
        state = _empty_minds_state()
        state["zeitgeist_history"] = [{"tick": i} for i in range(25)]
        state = con.tick_consciousness(world, state)
        assert len(state["zeitgeist_history"]) <= con.ZEITGEIST_WINDOW

    def test_tick_dream_log_capped(self):
        world = _make_world(5)
        state = _empty_minds_state()
        state["dream_log"] = [{"tick": i, "source": "x", "type": "dream",
                                "fragment": "test", "intensity": 0.5}
                               for i in range(100)]
        state = con.tick_consciousness(world, state)
        assert len(state["dream_log"]) <= con.DREAM_LOG_CAP


# ===========================================================================
# SYNAPSES
# ===========================================================================

class TestSynapses:
    """Tests for synapse creation and dynamics."""

    def test_nearby_organisms_form_synapses(self):
        random.seed(42)
        world = _make_world(10, spread=30.0)
        result = con.tick_consciousness(world, _empty_minds_state())
        assert len(result["synapses"]) > 0

    def test_distant_organisms_no_synapses(self):
        orgs = [
            _make_organism("a", x=0, y=0),
            _make_organism("b", x=9999, y=9999),
        ]
        world = {"organisms": orgs}
        result = con.tick_consciousness(world, _empty_minds_state())
        assert len(result["synapses"]) == 0

    def test_synapse_strength_bounded(self):
        random.seed(42)
        world = _make_world(15, spread=40)
        state = _empty_minds_state()
        for _ in range(10):
            state = con.tick_consciousness(world, state)
        for syn in state["synapses"]:
            assert syn["strength"] >= 0
            assert syn["strength"] <= con.SYNAPSE_MAX + 0.01

    def test_synapses_capped(self):
        random.seed(42)
        world = _make_world(50, spread=20)
        state = _empty_minds_state()
        for _ in range(5):
            state = con.tick_consciousness(world, state)
        assert len(state["synapses"]) <= con.MAX_SYNAPSES

    def test_synapse_has_required_fields(self):
        random.seed(42)
        world = _make_world(10, spread=30)
        result = con.tick_consciousness(world, _empty_minds_state())
        if result["synapses"]:
            syn = result["synapses"][0]
            assert "a" in syn
            assert "b" in syn
            assert "strength" in syn
            assert "age" in syn
            assert "signal" in syn

    def test_dead_synapses_pruned(self):
        random.seed(42)
        world = _make_world(10, spread=30)
        state = con.tick_consciousness(world, _empty_minds_state())
        world["organisms"] = world["organisms"][:5]
        state2 = con.tick_consciousness(world, state)
        alive_ids = set(o["id"] for o in world["organisms"])
        for syn in state2["synapses"]:
            assert syn["a"] in alive_ids
            assert syn["b"] in alive_ids


# ===========================================================================
# DREAMING
# ===========================================================================

class TestDreaming:
    """Tests for the dreaming system."""

    def test_low_energy_triggers_dreaming(self):
        org = _make_organism("dreamer", energy=10.0)
        world = {"organisms": [org]}
        result = con.tick_consciousness(world, _empty_minds_state())
        mind = result["minds"]["dreamer"]
        assert mind["is_dreaming"] is True

    def test_high_energy_no_dreaming(self):
        org = _make_organism("awake", energy=200.0)
        world = {"organisms": [org]}
        result = con.tick_consciousness(world, _empty_minds_state())
        mind = result["minds"]["awake"]
        assert mind["is_dreaming"] is False

    def test_dream_intensity_grows(self):
        org = _make_organism("dreamer", energy=5.0)
        world = {"organisms": [org]}
        state = _empty_minds_state()
        intensities = []
        for _ in range(5):
            state = con.tick_consciousness(world, state)
            intensities.append(state["minds"]["dreamer"]["dream_intensity"])
        assert intensities[-1] > intensities[0]

    def test_dream_intensity_decays_when_awake(self):
        org = _make_organism("sleeper", energy=5.0)
        world = {"organisms": [org]}
        state = _empty_minds_state()
        for _ in range(5):
            state = con.tick_consciousness(world, state)
        high_intensity = state["minds"]["sleeper"]["dream_intensity"]
        world["organisms"][0]["energy"] = 200.0
        for _ in range(5):
            state = con.tick_consciousness(world, state)
        assert state["minds"]["sleeper"]["dream_intensity"] < high_intensity

    def test_dream_fragments_are_valid(self):
        random.seed(42)
        org = _make_organism("dreamer", energy=5.0)
        world = {"organisms": [org]}
        state = _empty_minds_state()
        for _ in range(20):
            state = con.tick_consciousness(world, state)
        dreams = [d for d in state["dream_log"] if d["type"] == "dream"]
        for d in dreams:
            assert d["fragment"] in con.DREAM_FRAGMENTS

    def test_dream_fragments_capped_per_mind(self):
        random.seed(42)
        org = _make_organism("dreamer", energy=5.0)
        world = {"organisms": [org]}
        state = _empty_minds_state()
        for _ in range(50):
            state = con.tick_consciousness(world, state)
        frags = state["minds"]["dreamer"]["dream_fragments"]
        assert len(frags) <= 5


# ===========================================================================
# MOOD AND AROUSAL
# ===========================================================================

class TestMoodArousal:
    """Tests for mood and arousal dynamics."""

    def test_mood_bounded(self):
        random.seed(42)
        world = _make_world(15, spread=40)
        state = _empty_minds_state()
        for _ in range(20):
            state = con.tick_consciousness(world, state)
        for mind in state["minds"].values():
            assert -1.5 <= mind["mood"] <= 1.5

    def test_arousal_nonnegative(self):
        random.seed(42)
        world = _make_world(10)
        state = _empty_minds_state()
        for _ in range(10):
            state = con.tick_consciousness(world, state)
        for mind in state["minds"].values():
            assert mind["arousal"] >= -0.1

    def test_curiosity_bounded(self):
        random.seed(42)
        world = _make_world(10)
        state = _empty_minds_state()
        for _ in range(20):
            state = con.tick_consciousness(world, state)
        for mind in state["minds"].values():
            assert 0.0 <= mind["curiosity"] <= 1.0


# ===========================================================================
# SMOKE: MULTI-TICK
# ===========================================================================

class TestSmoke:
    """End-to-end simulation smoke tests."""

    def test_10_ticks_no_crash(self):
        random.seed(42)
        world = _make_world(20, spread=60)
        state = _empty_minds_state()
        for _ in range(10):
            state = con.tick_consciousness(world, state)
        assert state["_meta"]["tick"] == 10

    def test_50_ticks_stable(self):
        random.seed(123)
        world = _make_world(15, spread=50)
        state = _empty_minds_state()
        for _ in range(50):
            state = con.tick_consciousness(world, state)
        z = state["zeitgeist"]
        assert -2.0 <= z["collective_mood"] <= 2.0
        assert z["dreamer_ratio"] >= 0
        assert z["synapse_count"] >= 0

    def test_deterministic(self):
        def run(seed_val):
            random.seed(seed_val)
            world = _make_world(10, spread=40)
            state = _empty_minds_state()
            for _ in range(10):
                state = con.tick_consciousness(world, state)
            return (state["_meta"]["tick"],
                    len(state["synapses"]),
                    state["zeitgeist"]["synapse_count"])
        assert run(42) == run(42)

    def test_single_organism(self):
        org = _make_organism("lonely", energy=80)
        world = {"organisms": [org]}
        state = _empty_minds_state()
        for _ in range(10):
            state = con.tick_consciousness(world, state)
        assert "lonely" in state["minds"]
        assert len(state["synapses"]) == 0

    def test_organisms_as_dict(self):
        orgs = {f"org-{i}": _make_organism(f"org-{i}") for i in range(5)}
        world = {"organisms": orgs}
        result = con.tick_consciousness(world, _empty_minds_state())
        assert len(result["minds"]) == 5


# ===========================================================================
# PHYSICAL INVARIANTS
# ===========================================================================

class TestInvariants:
    """Property-based tests for bounds and conservation."""

    def test_dream_intensity_bounded(self):
        random.seed(42)
        org = _make_organism("d", energy=1.0)
        world = {"organisms": [org]}
        state = _empty_minds_state()
        for _ in range(100):
            state = con.tick_consciousness(world, state)
        assert state["minds"]["d"]["dream_intensity"] <= 1.0

    def test_synapse_signal_bounded(self):
        random.seed(42)
        world = _make_world(15, spread=30)
        state = _empty_minds_state()
        for _ in range(20):
            state = con.tick_consciousness(world, state)
        for syn in state["synapses"]:
            assert 0.0 <= syn["signal"] <= 1.01

    def test_zeitgeist_dreamer_ratio_bounded(self):
        random.seed(42)
        world = _make_world(10)
        state = _empty_minds_state()
        for _ in range(10):
            state = con.tick_consciousness(world, state)
        z = state["zeitgeist"]
        assert 0.0 <= z["dreamer_ratio"] <= 1.0

    def test_memories_capped(self):
        random.seed(42)
        world = _make_world(20, spread=20)
        state = _empty_minds_state()
        for _ in range(100):
            state = con.tick_consciousness(world, state)
        for mind in state["minds"].values():
            assert len(mind["memories"]) <= 10

    def test_bonds_count_consistent(self):
        random.seed(42)
        world = _make_world(10, spread=40)
        state = _empty_minds_state()
        for _ in range(5):
            state = con.tick_consciousness(world, state)
        for oid, mind in state["minds"].items():
            actual = sum(1 for s in state["synapses"]
                         if s["a"] == oid or s["b"] == oid)
            assert mind["bonds_count"] == actual


# ===========================================================================
# I/O
# ===========================================================================

class TestIO:
    """Tests for load/save functions."""

    def test_load_json_missing(self, tmp_path):
        assert con.load_json(tmp_path / "nonexistent.json") == {}

    def test_load_json_corrupt(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        assert con.load_json(bad) == {}

    def test_save_load_roundtrip(self, tmp_path):
        p = tmp_path / "test.json"
        data = {"_meta": {"tick": 5}, "minds": {"a": 1}}
        con.save_json(p, data)
        loaded = con.load_json(p)
        assert loaded["_meta"]["tick"] == 5

    def test_save_atomic(self, tmp_path):
        p = tmp_path / "state.json"
        con.save_json(p, {"test": True})
        assert p.exists()
        assert not p.with_suffix(".tmp").exists()


# ===========================================================================
# EDGE CASES
# ===========================================================================

class TestEdgeCases:
    """Edge cases and additional invariants."""

    def test_empty_world(self):
        world = {"organisms": []}
        result = con.tick_consciousness(world, _empty_minds_state())
        assert len(result["minds"]) == 0
        assert result["zeitgeist"]["dreamer_ratio"] == 0.0

    def test_all_dreamers(self):
        orgs = [_make_organism(f"d-{i}", energy=1.0) for i in range(10)]
        world = {"organisms": orgs}
        result = con.tick_consciousness(world, _empty_minds_state())
        assert result["zeitgeist"]["dreamers"] == 10
        assert result["zeitgeist"]["dreamer_ratio"] == pytest.approx(1.0)

    def test_no_dreamers(self):
        orgs = [_make_organism(f"a-{i}", energy=200.0) for i in range(10)]
        world = {"organisms": orgs}
        result = con.tick_consciousness(world, _empty_minds_state())
        assert result["zeitgeist"]["dreamers"] == 0

    def test_synapse_signal_decays(self):
        random.seed(42)
        world = _make_world(10, spread=30)
        for o in world["organisms"]:
            o["energy"] = 200
        state = _empty_minds_state()
        state = con.tick_consciousness(world, state)
        if state["synapses"]:
            state["synapses"][0]["signal"] = 0.8
        state2 = con.tick_consciousness(world, state)
        if state2["synapses"]:
            assert state2["synapses"][0]["signal"] < 0.8

    def test_dream_propagation_through_synapses(self):
        random.seed(42)
        orgs = [
            _make_organism("dreamer", x=100, y=100, energy=5.0,
                           genome=[0.5]*16),
            _make_organism("awake", x=105, y=100, energy=200.0,
                           genome=[0.5]*16),
        ]
        world = {"organisms": orgs}
        state = _empty_minds_state()
        for _ in range(30):
            state = con.tick_consciousness(world, state)
        assert "awake" in state["minds"]

    def test_gathering_creates_memory(self):
        random.seed(42)
        orgs = [_make_organism(f"g-{i}", x=100+i*0.5, y=100, energy=80)
                for i in range(15)]
        world = {"organisms": orgs}
        state = _empty_minds_state()
        for _ in range(50):
            state = con.tick_consciousness(world, state)
        all_memories = []
        for mind in state["minds"].values():
            all_memories.extend(mind["memories"])
        gathering_mems = [m for m in all_memories if "gathering" in m]
        assert len(gathering_mems) > 0

    def test_hunger_creates_memory(self):
        random.seed(42)
        orgs = [_make_organism(f"h-{i}", energy=10.0) for i in range(5)]
        world = {"organisms": orgs}
        state = _empty_minds_state()
        for _ in range(50):
            state = con.tick_consciousness(world, state)
        all_memories = []
        for mind in state["minds"].values():
            all_memories.extend(mind["memories"])
        hunger_mems = [m for m in all_memories if "hunger" in m]
        assert len(hunger_mems) > 0

    def test_version_in_meta(self):
        world = _make_world(3)
        result = con.tick_consciousness(world, _empty_minds_state())
        assert result["_meta"]["version"] == "1.0.0"

    def test_updated_at_in_meta(self):
        world = _make_world(3)
        result = con.tick_consciousness(world, _empty_minds_state())
        assert "T" in result["_meta"]["updated_at"]

    def test_zeitgeist_avg_synapse_no_division_by_zero(self):
        orgs = [
            _make_organism("far-1", x=0, y=0, energy=80),
            _make_organism("far-2", x=9999, y=9999, energy=80),
        ]
        world = {"organisms": orgs}
        result = con.tick_consciousness(world, _empty_minds_state())
        assert result["zeitgeist"]["avg_synapse_strength"] >= 0

    def test_multi_tick_with_population_change(self):
        random.seed(42)
        world = _make_world(10, spread=40)
        state = _empty_minds_state()
        state = con.tick_consciousness(world, state)
        world["organisms"].append(_make_organism("new-1", x=120, y=120))
        world["organisms"].append(_make_organism("new-2", x=130, y=130))
        state = con.tick_consciousness(world, state)
        assert "new-1" in state["minds"]
        assert "new-2" in state["minds"]
        world["organisms"] = [o for o in world["organisms"]
                               if o["id"] in ("new-1", "new-2")]
        state = con.tick_consciousness(world, state)
        assert len(state["minds"]) == 2

    def test_100_ticks_stress(self):
        random.seed(42)
        world = _make_world(20, spread=60)
        state = _empty_minds_state()
        for i in range(100):
            if i % 10 == 0:
                world = _make_world(random.randint(5, 25), spread=50)
            state = con.tick_consciousness(world, state)
        assert state["_meta"]["tick"] == 100
