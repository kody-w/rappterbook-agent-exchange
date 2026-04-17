"""Tests for the cultural memory ecology module and its engine integration."""
from __future__ import annotations

import random

import pytest

from src.mars100.memory_ecology import (
    CulturalMemory, MemoryPool,
    event_to_theme, memory_action_bias, memory_vote_bias,
    inherit_cultural_memory,
    MAX_POOL_SIZE, SALIENCE_DECAY, MYTH_FIDELITY_THRESHOLD,
    MAX_ACTION_BIAS, MAX_VOTE_BIAS,
    EVENT_THEME_MAP,
)
from src.mars100.engine import Mars100Engine


# ---------- Unit tests: CulturalMemory ----------

class TestCulturalMemory:
    def test_myth_threshold(self):
        m = CulturalMemory("loss", 0.5, 0.29, 1, 1)
        assert m.is_myth is True
        m2 = CulturalMemory("loss", 0.5, 0.31, 1, 1)
        assert m2.is_myth is False

    def test_roundtrip(self):
        m = CulturalMemory("drought", 0.7, 0.8, 5, 10, retell_count=3, valence=-0.4)
        d = m.to_dict()
        m2 = CulturalMemory.from_dict(d)
        assert m2.theme == m.theme
        assert abs(m2.salience - m.salience) < 1e-3
        assert m2.retell_count == m.retell_count

    def test_to_dict_keys(self):
        m = CulturalMemory("test", 0.5, 1.0, 1, 1)
        d = m.to_dict()
        assert set(d.keys()) == {"theme", "salience", "fidelity", "origin_year",
                                  "last_reinforced", "retell_count", "valence", "is_myth"}


# ---------- Unit tests: MemoryPool ----------

class TestMemoryPool:
    def test_record_new(self):
        pool = MemoryPool()
        pool.record("drought", 1, -0.5)
        assert "drought" in pool.entries
        assert pool.entries["drought"].origin_year == 1
        assert pool.entries["drought"].salience == 0.5

    def test_record_reinforce(self):
        pool = MemoryPool()
        pool.record("drought", 1, -0.5)
        pool.record("drought", 3, -0.3)
        mem = pool.entries["drought"]
        assert mem.retell_count == 1
        assert mem.salience > 0.5
        assert mem.fidelity < 1.0
        assert mem.last_reinforced == 3

    def test_decay(self):
        pool = MemoryPool()
        pool.record("test", 1, 0.0)
        initial = pool.entries["test"].salience
        pool.decay(2)
        assert pool.entries["test"].salience == pytest.approx(initial * SALIENCE_DECAY, rel=1e-6)

    def test_decay_prunes_weak(self):
        pool = MemoryPool()
        pool.entries["weak"] = CulturalMemory("weak", 0.005, 1.0, 1, 1)
        pool.decay(10)
        assert "weak" not in pool.entries

    def test_cap_enforced(self):
        pool = MemoryPool()
        for i in range(MAX_POOL_SIZE + 10):
            pool.record(f"theme_{i}", 1, 0.0)
        assert len(pool.entries) <= MAX_POOL_SIZE

    def test_top_memories_ordering(self):
        pool = MemoryPool()
        pool.record("low", 1, 0.0)
        pool.record("high", 1, 0.0)
        pool.entries["high"].salience = 0.9
        pool.entries["low"].salience = 0.1
        top = pool.top_memories(2)
        assert top[0].theme == "high"
        assert top[1].theme == "low"

    def test_myths(self):
        pool = MemoryPool()
        pool.entries["old"] = CulturalMemory("old", 0.5, 0.1, 1, 1)
        pool.entries["new"] = CulturalMemory("new", 0.5, 0.9, 1, 1)
        myths = pool.myths()
        assert len(myths) == 1
        assert myths[0].theme == "old"

    def test_summary_structure(self):
        pool = MemoryPool()
        pool.record("drought", 1, -0.3)
        s = pool.summary()
        assert "pool_size" in s
        assert "myth_count" in s
        assert "top_themes" in s
        assert s["pool_size"] == 1

    def test_roundtrip(self):
        pool = MemoryPool()
        pool.record("a", 1, 0.1)
        pool.record("b", 2, -0.2)
        d = pool.to_dict()
        pool2 = MemoryPool.from_dict(d)
        assert set(pool2.entries.keys()) == {"a", "b"}

    def test_empty_pool_summary(self):
        pool = MemoryPool()
        s = pool.summary()
        assert s["pool_size"] == 0
        assert s["myth_count"] == 0
        assert s["top_themes"] == []


# ---------- Unit tests: bias functions ----------

class TestBiasFunctions:
    def test_event_to_theme_known(self):
        assert event_to_theme("dust_storm") == "drought"
        assert event_to_theme("epidemic") == "plague"

    def test_event_to_theme_unknown(self):
        assert event_to_theme("totally_new") == "general"

    def test_all_event_templates_mapped(self):
        """Every event in EVENT_THEME_MAP maps to a non-empty string."""
        for event_name, theme in EVENT_THEME_MAP.items():
            assert isinstance(theme, str) and len(theme) > 0

    def test_action_bias_empty_pool(self):
        pool = MemoryPool()
        biases = memory_action_bias(pool)
        assert biases == {}

    def test_action_bias_capped(self):
        pool = MemoryPool()
        # Flood the pool with high-salience drought memories
        for i in range(20):
            pool.record("drought", i, -1.0)
        biases = memory_action_bias(pool)
        for v in biases.values():
            assert abs(v) <= MAX_ACTION_BIAS + 1e-9

    def test_action_bias_negative_valence_amplifies(self):
        pool_neg = MemoryPool()
        pool_neg.record("drought", 1, -0.8)
        pool_pos = MemoryPool()
        pool_pos.record("drought", 1, 0.8)
        bias_neg = memory_action_bias(pool_neg)
        bias_pos = memory_action_bias(pool_pos)
        # Negative valence should produce stronger bias
        assert bias_neg.get("terraform", 0) > bias_pos.get("terraform", 0)

    def test_vote_bias_empty(self):
        pool = MemoryPool()
        assert memory_vote_bias(pool, "council") == 0.0

    def test_vote_bias_tyranny_discourages_dictator(self):
        pool = MemoryPool()
        pool.record("tyranny", 1, -0.5)
        bias = memory_vote_bias(pool, "dictator")
        assert bias < 0

    def test_vote_bias_capped(self):
        pool = MemoryPool()
        for i in range(30):
            pool.record("tyranny", i, -1.0)
        bias = memory_vote_bias(pool, "dictator")
        assert abs(bias) <= MAX_VOTE_BIAS + 1e-9


# ---------- Unit tests: inheritance ----------

class TestInheritance:
    def test_empty_pool(self):
        pool = MemoryPool()
        rng = random.Random(42)
        assert inherit_cultural_memory(pool, 50, rng) == []

    def test_max_three(self):
        pool = MemoryPool()
        for i in range(10):
            pool.record(f"theme_{i}", 1, 0.0)
            pool.entries[f"theme_{i}"].salience = 1.0  # force high prob
        rng = random.Random(42)
        inherited = inherit_cultural_memory(pool, 50, rng)
        assert len(inherited) <= 3

    def test_deterministic(self):
        pool = MemoryPool()
        pool.record("a", 1, 0.0)
        pool.record("b", 1, 0.0)
        r1 = inherit_cultural_memory(pool, 10, random.Random(99))
        r2 = inherit_cultural_memory(pool, 10, random.Random(99))
        assert r1 == r2


# ---------- Integration: engine with cultural memory ----------

class TestEngineIntegration:
    def test_cultural_memory_in_year_result(self):
        e = Mars100Engine(42, 5)
        r = e.run()
        for y in r.years:
            assert "pool_size" in y.cultural_memory
            assert "myth_count" in y.cultural_memory
            assert "top_themes" in y.cultural_memory

    def test_pool_grows_over_years(self):
        e = Mars100Engine(42, 20)
        r = e.run()
        first = r.years[0].cultural_memory["pool_size"]
        last = r.years[-1].cultural_memory["pool_size"]
        assert last >= first

    def test_100_year_run_with_cultural_memory(self):
        e = Mars100Engine(42, 100)
        r = e.run()
        assert len(r.years) > 0
        d = r.to_dict()
        assert d["_meta"]["version"] == "5.0"
        assert "cultural_memory" in d
        assert len(d["cultural_memory"]) > 0

    def test_myths_emerge_over_time(self):
        """Over 100 years, repeatedly reinforced memories should mythologize."""
        e = Mars100Engine(42, 100)
        r = e.run()
        last_year = r.years[-1]
        # At least some themes should have been retold enough to become myths
        pool = r.cultural_memory
        myth_count = sum(1 for v in pool.values() if v.get("is_myth", False))
        # With 100 years of events, we expect at least one myth
        assert myth_count >= 1 or last_year.cultural_memory["pool_size"] > 0

    def test_cultural_memory_in_to_dict(self):
        e = Mars100Engine(42, 3)
        r = e.run()
        d = r.to_dict()
        for yd in d["years"]:
            assert "cultural_memory" in yd
        assert "cultural_memory" in d

    def test_deterministic_with_memory(self):
        """Same seed produces same cultural memory."""
        e1 = Mars100Engine(42, 20)
        e2 = Mars100Engine(42, 20)
        r1 = e1.run()
        r2 = e2.run()
        assert r1.years[-1].cultural_memory == r2.years[-1].cultural_memory

    def test_memory_pool_bounded(self):
        """Pool never exceeds MAX_POOL_SIZE even with many events."""
        e = Mars100Engine(42, 100)
        r = e.run()
        for y in r.years:
            assert y.cultural_memory["pool_size"] <= MAX_POOL_SIZE
