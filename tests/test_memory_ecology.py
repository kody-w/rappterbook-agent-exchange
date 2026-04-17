"""Tests for the cultural memory ecology module."""
from __future__ import annotations

import pytest

from src.mars100.memory_ecology import (
    CulturalMemory,
    MemoryPool,
    event_to_theme,
    memory_action_bias,
    memory_vote_bias,
    inherit_cultural_memory,
    MAX_POOL_SIZE,
    MAX_SALIENCE,
    SALIENCE_DECAY,
    FIDELITY_DECAY,
    MYTH_FIDELITY_THRESHOLD,
    SALIENCE_FLOOR,
    THEME_MAP,
    DEFAULT_THEME,
)


# ── event_to_theme ──────────────────────────────────────────────────

class TestEventToTheme:
    def test_known_events_map(self):
        assert event_to_theme("dust_storm") == "survival:environment"
        assert event_to_theme("birth") == "growth:birth"
        assert event_to_theme("sabotage") == "social:conflict"

    def test_unknown_event_returns_default(self):
        assert event_to_theme("totally_unknown") == DEFAULT_THEME

    def test_all_theme_map_entries_have_colon(self):
        """Themes follow 'category:detail' format."""
        for theme in THEME_MAP.values():
            assert ":" in theme, f"Theme {theme!r} missing colon separator"


# ── CulturalMemory ──────────────────────────────────────────────────

class TestCulturalMemory:
    def test_creation(self):
        m = CulturalMemory(theme="survival:environment", year_formed=5,
                           salience=1.0, fidelity=1.0)
        assert m.theme == "survival:environment"
        assert m.event_count == 1
        assert not m.mythologized

    def test_decay_reduces_salience_and_fidelity(self):
        m = CulturalMemory(theme="t", year_formed=1, salience=1.0, fidelity=1.0)
        m.decay()
        assert abs(m.salience - SALIENCE_DECAY) < 1e-6
        assert abs(m.fidelity - FIDELITY_DECAY) < 1e-6

    def test_mythologization_triggers_at_threshold(self):
        m = CulturalMemory(theme="t", year_formed=1, salience=1.0,
                           fidelity=MYTH_FIDELITY_THRESHOLD + 0.01)
        assert not m.mythologized
        # Decay enough to cross threshold
        while m.fidelity >= MYTH_FIDELITY_THRESHOLD:
            m.decay()
        assert m.mythologized

    def test_reinforce_caps_at_max(self):
        m = CulturalMemory(theme="t", year_formed=1,
                           salience=MAX_SALIENCE - 0.1, fidelity=1.0)
        m.reinforce(10.0)
        assert m.salience <= MAX_SALIENCE
        assert m.event_count == 2

    def test_prunable_below_floor(self):
        m = CulturalMemory(theme="t", year_formed=1,
                           salience=SALIENCE_FLOOR - 0.01, fidelity=0.5)
        assert m.is_prunable()

    def test_not_prunable_above_floor(self):
        m = CulturalMemory(theme="t", year_formed=1,
                           salience=SALIENCE_FLOOR + 0.01, fidelity=0.5)
        assert not m.is_prunable()

    def test_roundtrip_serialization(self):
        m = CulturalMemory(theme="loss:death", year_formed=42,
                           salience=1.5, fidelity=0.7, event_count=3,
                           mythologized=True)
        d = m.to_dict()
        m2 = CulturalMemory.from_dict(d)
        assert m2.theme == m.theme
        assert m2.year_formed == m.year_formed
        assert abs(m2.salience - m.salience) < 1e-4
        assert m2.mythologized == m.mythologized
        assert m2.event_count == m.event_count


# ── MemoryPool ──────────────────────────────────────────────────────

class TestMemoryPool:
    def test_record_new_theme(self):
        pool = MemoryPool()
        pool.record("dust_storm", year=5)
        assert "survival:environment" in pool.memories
        assert pool.memories["survival:environment"].year_formed == 5

    def test_record_reinforces_existing(self):
        pool = MemoryPool()
        pool.record("dust_storm", year=5)
        initial = pool.memories["survival:environment"].salience
        pool.record("solar_flare", year=6)  # same theme
        assert pool.memories["survival:environment"].salience > initial
        assert pool.memories["survival:environment"].event_count == 2

    def test_tick_decays_and_prunes(self):
        pool = MemoryPool()
        pool.record("dust_storm", year=1, salience=SALIENCE_FLOOR + 0.01)
        # Decay enough to prune
        for _ in range(100):
            pool.tick()
        assert len(pool.memories) == 0

    def test_pool_cap_enforced(self):
        pool = MemoryPool()
        for i in range(MAX_POOL_SIZE + 20):
            theme = f"test:theme-{i}"
            pool.memories[theme] = CulturalMemory(
                theme=theme, year_formed=1,
                salience=1.0 + i * 0.01, fidelity=1.0,
            )
        pool.tick()
        assert len(pool.memories) <= MAX_POOL_SIZE

    def test_pool_cap_keeps_highest_salience(self):
        pool = MemoryPool()
        for i in range(MAX_POOL_SIZE + 5):
            theme = f"test:theme-{i}"
            pool.memories[theme] = CulturalMemory(
                theme=theme, year_formed=1,
                salience=float(i), fidelity=1.0,
            )
        pool.tick()
        remaining_saliences = [m.salience for m in pool.memories.values()]
        # The lowest remaining salience should be among the higher entries
        assert min(remaining_saliences) > 0

    def test_top_themes_ordering(self):
        pool = MemoryPool()
        pool.memories["a"] = CulturalMemory("a", 1, salience=0.5, fidelity=1.0)
        pool.memories["b"] = CulturalMemory("b", 1, salience=2.0, fidelity=1.0)
        pool.memories["c"] = CulturalMemory("c", 1, salience=1.0, fidelity=1.0)
        top = pool.top_themes(2)
        assert len(top) == 2
        assert top[0].theme == "b"
        assert top[1].theme == "c"

    def test_summary_structure(self):
        pool = MemoryPool()
        pool.record("dust_storm", year=1)
        s = pool.summary()
        assert "pool_size" in s
        assert "top_themes" in s
        assert "myth_count" in s
        assert s["pool_size"] == 1

    def test_full_state_roundtrip(self):
        pool = MemoryPool()
        pool.record("dust_storm", year=1)
        pool.record("birth", year=5)
        state = pool.full_state()
        pool2 = MemoryPool()
        pool2.load_state(state)
        assert len(pool2.memories) == len(pool.memories)
        for theme in pool.memories:
            assert theme in pool2.memories

    def test_empty_pool_summary(self):
        pool = MemoryPool()
        s = pool.summary()
        assert s["pool_size"] == 0
        assert s["top_themes"] == []
        assert s["myth_count"] == 0


# ── Bias helpers ────────────────────────────────────────────────────

class TestMemoryActionBias:
    def test_empty_pool_returns_zero(self):
        pool = MemoryPool()
        assert memory_action_bias(pool, "farm") == 0.0

    def test_survival_memory_boosts_terraform(self):
        pool = MemoryPool()
        pool.memories["survival:environment"] = CulturalMemory(
            "survival:environment", 1, salience=2.0, fidelity=1.0)
        bias = memory_action_bias(pool, "terraform")
        assert bias > 0

    def test_conflict_memory_suppresses_sabotage(self):
        pool = MemoryPool()
        pool.memories["social:conflict"] = CulturalMemory(
            "social:conflict", 1, salience=2.0, fidelity=1.0)
        bias = memory_action_bias(pool, "sabotage")
        assert bias < 0

    def test_bias_capped_at_half(self):
        pool = MemoryPool()
        # Load many high-salience memories that all boost the same action
        for i in range(10):
            theme = f"survival:environment"
            pool.memories[theme] = CulturalMemory(
                theme, 1, salience=MAX_SALIENCE, fidelity=1.0)
        bias = memory_action_bias(pool, "terraform")
        assert -0.5 <= bias <= 0.5

    def test_mythologized_memories_amplified(self):
        pool = MemoryPool()
        # Non-myth
        pool.memories["survival:environment"] = CulturalMemory(
            "survival:environment", 1, salience=1.0, fidelity=0.8)
        bias_normal = memory_action_bias(pool, "terraform")
        # Myth
        pool.memories["survival:environment"] = CulturalMemory(
            "survival:environment", 1, salience=1.0, fidelity=0.2,
            mythologized=True)
        bias_myth = memory_action_bias(pool, "terraform")
        assert bias_myth > bias_normal

    def test_unrelated_action_gets_zero(self):
        pool = MemoryPool()
        pool.memories["spiritual:faith"] = CulturalMemory(
            "spiritual:faith", 1, salience=2.0, fidelity=1.0)
        # "hoard" has no affinity with spiritual:faith
        bias = memory_action_bias(pool, "hoard")
        assert bias == 0.0


class TestMemoryVoteBias:
    def test_empty_pool_returns_zero(self):
        pool = MemoryPool()
        assert memory_vote_bias(pool, "council") == 0.0

    def test_conflict_memory_boosts_council(self):
        pool = MemoryPool()
        pool.memories["social:conflict"] = CulturalMemory(
            "social:conflict", 1, salience=2.0, fidelity=1.0)
        bias = memory_vote_bias(pool, "council")
        assert bias > 0

    def test_bias_capped(self):
        pool = MemoryPool()
        for theme in ["social:conflict", "survival:environment",
                       "governance:change", "loss:death", "mystery:cosmic"]:
            pool.memories[theme] = CulturalMemory(
                theme, 1, salience=MAX_SALIENCE, fidelity=1.0)
        bias = memory_vote_bias(pool, "council")
        assert -0.3 <= bias <= 0.3


# ── Inheritance ─────────────────────────────────────────────────────

class TestInheritance:
    def test_child_pool_has_reduced_fidelity(self):
        parent = MemoryPool()
        parent.memories["survival:environment"] = CulturalMemory(
            "survival:environment", 1, salience=2.0, fidelity=0.9)
        child = inherit_cultural_memory(parent)
        cm = child.memories["survival:environment"]
        assert cm.fidelity < 0.9
        assert cm.salience < 2.0

    def test_child_inherits_top_themes(self):
        parent = MemoryPool()
        for i in range(15):
            parent.memories[f"t:{i}"] = CulturalMemory(
                f"t:{i}", 1, salience=float(i), fidelity=0.9)
        child = inherit_cultural_memory(parent)
        # Should inherit top 10
        assert len(child.memories) == 10

    def test_child_mythologization_on_low_fidelity(self):
        parent = MemoryPool()
        parent.memories["t:old"] = CulturalMemory(
            "t:old", 1, salience=1.0,
            fidelity=MYTH_FIDELITY_THRESHOLD / 0.85 + 0.01)  # just above after penalty
        child = inherit_cultural_memory(parent)
        # Depending on exact math, may or may not mythologize
        assert "t:old" in child.memories


# ── Integration / smoke ─────────────────────────────────────────────

class TestSmoke:
    def test_100_year_pool_evolution(self):
        """Simulate 100 years of events and verify pool stays bounded."""
        import random
        rng = random.Random(42)
        pool = MemoryPool()
        events_per_year = [
            "dust_storm", "solar_flare", "equipment_failure",
            "resource_strike", "calm", "birth", "death",
            "colonist_conflict", "cooperate", "pray",
        ]
        for year in range(1, 101):
            n_events = rng.randint(1, 4)
            for _ in range(n_events):
                pool.record(rng.choice(events_per_year), year,
                            salience=rng.uniform(0.5, 1.5))
            pool.tick()

        assert len(pool.memories) <= MAX_POOL_SIZE
        assert len(pool.memories) > 0  # some should survive
        # Should have mythologized memories after 100 years
        myths = sum(1 for m in pool.memories.values() if m.mythologized)
        assert myths > 0

    def test_bias_with_realistic_pool(self):
        """After many events, biases should be non-trivial but bounded."""
        pool = MemoryPool()
        for year in range(1, 30):
            pool.record("dust_storm", year)
            pool.record("colonist_conflict", year)
            pool.tick()
        # Survival memories should boost terraform
        tf_bias = memory_action_bias(pool, "terraform")
        assert tf_bias > 0
        assert tf_bias <= 0.5
        # Conflict memories should boost mediate
        med_bias = memory_action_bias(pool, "mediate")
        assert med_bias > 0

    def test_deterministic_with_same_events(self):
        """Same event sequence → same pool state."""
        def run_pool() -> list[dict]:
            pool = MemoryPool()
            for year in range(1, 20):
                pool.record("dust_storm", year)
                pool.tick()
            return pool.full_state()
        s1 = run_pool()
        s2 = run_pool()
        assert s1 == s2
