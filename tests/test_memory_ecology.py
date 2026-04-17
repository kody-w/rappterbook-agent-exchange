"""Tests for the Mars-100 cultural memory ecology module."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.mars100.memory_ecology import (
    CulturalMemory, MemoryPool, event_to_theme,
    memory_action_bias, memory_vote_bias, inherit_cultural_memory,
    MAX_POOL_SIZE, SALIENCE_DECAY, MAX_ACTION_BIAS, MAX_VOTE_BIAS, MIN_SALIENCE,
)


class TestMemoryPool:
    def test_record_new_creates_entry(self):
        pool = MemoryPool()
        pool.record("scarcity", year=5, valence=-0.6)
        assert "scarcity" in pool.entries
        mem = pool.entries["scarcity"]
        assert mem.theme == "scarcity"
        assert mem.first_year == 5
        assert mem.salience == 1.0
        assert mem.valence == pytest.approx(-0.6)

    def test_reinforce_increases_salience(self):
        pool = MemoryPool()
        pool.record("scarcity", year=5, valence=-0.6)
        pool.record("scarcity", year=8, valence=-0.3)
        mem = pool.entries["scarcity"]
        assert mem.salience > 1.0
        assert mem.reinforcements == 1
        assert mem.last_year == 8

    def test_reinforce_degrades_fidelity(self):
        pool = MemoryPool()
        pool.record("scarcity", year=5, valence=-0.6)
        pool.record("scarcity", year=8, valence=-0.3)
        assert pool.entries["scarcity"].fidelity < 1.0

    def test_decay_reduces_salience(self):
        pool = MemoryPool()
        pool.record("scarcity", year=1, valence=-0.5)
        initial = pool.entries["scarcity"].salience
        pool.decay(2)
        assert pool.entries["scarcity"].salience < initial
        assert pool.entries["scarcity"].salience == pytest.approx(initial * SALIENCE_DECAY)

    def test_decay_prunes_dead_entries(self):
        pool = MemoryPool()
        pool.record("scarcity", year=1, valence=-0.5)
        for yr in range(200):
            pool.decay(yr)
        assert "scarcity" not in pool.entries

    def test_max_pool_size_evicts_weakest(self):
        pool = MemoryPool()
        for i in range(MAX_POOL_SIZE):
            pool.record(f"theme_{i}", year=1, valence=0.1)
        pool.record("overflow", year=2, valence=0.5)
        assert len(pool.entries) == MAX_POOL_SIZE
        assert "overflow" in pool.entries

    def test_top_returns_most_salient(self):
        pool = MemoryPool()
        pool.record("weak", year=1, valence=0.1)
        pool.record("strong", year=1, valence=0.5)
        pool.record("strong", year=2, valence=0.5)
        top = pool.top(1)
        assert len(top) == 1
        assert top[0].theme == "strong"

    def test_summary_and_to_dict(self):
        pool = MemoryPool()
        pool.record("scarcity", year=1, valence=-0.5)
        s = pool.summary()
        assert s["size"] == 1
        assert len(s["top_themes"]) == 1
        d = pool.to_dict()
        assert d["size"] == 1
        assert "scarcity" in d["entries"]

    def test_empty_pool_operations(self):
        pool = MemoryPool()
        pool.decay(1)
        assert pool.top(5) == []
        assert pool.summary() == {"size": 0, "top_themes": []}
        assert pool.to_dict() == {"size": 0, "entries": {}}


class TestEventToTheme:
    def test_exact_match(self):
        assert event_to_theme("dust_storm") == "natural_disaster"
        assert event_to_theme("resource_discovery") == "abundance"

    def test_partial_match(self):
        assert event_to_theme("major_dust_storm") == "natural_disaster"
        assert event_to_theme("severe_disease_outbreak") == "plague"

    def test_unmapped_returns_none(self):
        assert event_to_theme("unknown_event") is None
        assert event_to_theme("") is None

    def test_case_insensitive(self):
        assert event_to_theme("Dust_Storm") == "natural_disaster"
        assert event_to_theme("CONFLICT") == "conflict"


class TestActionBias:
    def test_empty_pool_no_bias(self):
        pool = MemoryPool()
        assert memory_action_bias(pool) == {}

    def test_scarcity_encourages_hoarding(self):
        pool = MemoryPool()
        pool.record("scarcity", year=1, valence=-0.5)
        biases = memory_action_bias(pool)
        assert "hoard" in biases
        assert biases["hoard"] > 0

    def test_bias_capped(self):
        pool = MemoryPool()
        for i in range(20):
            pool.record("scarcity", year=i, valence=-0.5)
        biases = memory_action_bias(pool)
        for v in biases.values():
            assert -MAX_ACTION_BIAS <= v <= MAX_ACTION_BIAS

    def test_conflicting_themes_partially_cancel(self):
        pool = MemoryPool()
        pool.record("scarcity", year=1, valence=-0.5)
        pool.record("abundance", year=1, valence=0.5)
        biases = memory_action_bias(pool)
        pure_pool = MemoryPool()
        pure_pool.record("scarcity", year=1, valence=-0.5)
        pure_biases = memory_action_bias(pure_pool)
        assert biases.get("hoard", 0) < pure_biases.get("hoard", 0)


class TestVoteBias:
    def test_empty_pool_no_bias(self):
        pool = MemoryPool()
        assert memory_vote_bias(pool, "democracy") == 0.0

    def test_tyranny_memory_opposes_dictator(self):
        pool = MemoryPool()
        pool.record("tyranny", year=1, valence=-0.5)
        assert memory_vote_bias(pool, "dictator") < 0

    def test_tyranny_memory_favours_democracy(self):
        pool = MemoryPool()
        pool.record("tyranny", year=1, valence=-0.5)
        assert memory_vote_bias(pool, "democracy") > 0

    def test_vote_bias_capped(self):
        pool = MemoryPool()
        for i in range(20):
            pool.record("tyranny", year=i, valence=-0.5)
        bias = memory_vote_bias(pool, "dictator")
        assert -MAX_VOTE_BIAS <= bias <= MAX_VOTE_BIAS


class TestInheritance:
    def test_empty_pool_returns_empty(self):
        pool = MemoryPool()
        rng = random.Random(42)
        assert inherit_cultural_memory(pool, year=10, rng=rng) == []

    def test_inheritance_returns_themes(self):
        pool = MemoryPool()
        pool.record("scarcity", year=1, valence=-0.5)
        pool.record("wonder", year=2, valence=0.8)
        pool.record("conflict", year=3, valence=-0.3)
        rng = random.Random(42)
        inherited = inherit_cultural_memory(pool, year=10, rng=rng)
        assert 1 <= len(inherited) <= 3
        for entry in inherited:
            assert "theme" in entry
            assert entry["inherited_year"] == 10
            assert entry["inherited_fidelity"] <= 1.0

    def test_inheritance_deterministic_with_seed(self):
        pool = MemoryPool()
        pool.record("scarcity", year=1, valence=-0.5)
        pool.record("wonder", year=2, valence=0.8)
        pool.record("conflict", year=3, valence=-0.3)
        r1 = inherit_cultural_memory(pool, year=10, rng=random.Random(42))
        r2 = inherit_cultural_memory(pool, year=10, rng=random.Random(42))
        assert r1 == r2

    def test_inheritance_more_salient_preferred(self):
        pool = MemoryPool()
        pool.record("weak", year=1, valence=0.1)
        for i in range(10):
            pool.record("strong", year=i, valence=0.5)
        results = []
        for seed in range(50):
            inherited = inherit_cultural_memory(pool, year=20, rng=random.Random(seed))
            themes = [e["theme"] for e in inherited]
            if "strong" in themes:
                results.append(True)
        assert sum(results) > 30


class TestEngineIntegration:
    def test_engine_has_memory_pool(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42)
        assert hasattr(engine, "memory_pool")
        assert isinstance(engine.memory_pool, MemoryPool)

    def test_year_result_has_cultural_memory(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42)
        result = engine.tick()
        assert hasattr(result, "cultural_memory")
        assert isinstance(result.cultural_memory, dict)
        assert "cultural_memory" in result.to_dict()

    def test_sim_result_has_cultural_memory(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=3)
        result = engine.run()
        assert hasattr(result, "cultural_memory")
        d = result.to_dict()
        assert "cultural_memory" in d
        assert "entries" in d["cultural_memory"]

    def test_version_is_5(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=1)
        result = engine.run()
        assert result.to_dict()["_meta"]["version"] == "5.0"

    def test_cultural_memory_grows_over_time(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        assert result.to_dict()["cultural_memory"]["size"] > 0

    def test_ten_year_smoke_test(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        d = result.to_dict()
        assert len(d["years"]) == 10
        assert d["_meta"]["version"] == "5.0"
        assert "cultural_memory" in d
        for yr in d["years"]:
            assert "cultural_memory" in yr
        cm = d["cultural_memory"]
        assert cm["size"] >= 0
        assert "entries" in cm
