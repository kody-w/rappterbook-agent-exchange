"""Tests for the collective memory organ."""
from __future__ import annotations

import random
import pytest
from src.mars100.collective import (
    CollectiveMemory,
    KnowledgeEntry,
    Tradition,
    KNOWLEDGE_HALF_LIFE_YEARS,
    MAX_ARCHIVE_SIZE,
    TRADITION_THRESHOLD,
    ARCHIVE_INFLUENCE_CAP,
    MIN_CONFIDENCE,
    TOPICS,
)


# ---------------------------------------------------------------------------
# KnowledgeEntry tests
# ---------------------------------------------------------------------------


class TestKnowledgeEntry:
    """Tests for individual knowledge entries."""

    def test_age_calculation(self):
        entry = KnowledgeEntry(
            id="k-0", topic="resources", source_colonist="kira-sol",
            year_created=10, content="test", confidence=0.8, conditions={},
        )
        assert entry.age(10) == 0
        assert entry.age(30) == 20
        assert entry.age(110) == 100

    def test_confidence_decay_at_half_life(self):
        entry = KnowledgeEntry(
            id="k-0", topic="resources", source_colonist="kira-sol",
            year_created=0, content="test", confidence=1.0, conditions={},
        )
        ec = entry.effective_confidence(KNOWLEDGE_HALF_LIFE_YEARS)
        assert abs(ec - 0.5) < 0.01, f"Expected ~0.5, got {ec}"

    def test_confidence_decay_zero_age(self):
        entry = KnowledgeEntry(
            id="k-0", topic="resources", source_colonist="kira-sol",
            year_created=50, content="test", confidence=0.9, conditions={},
        )
        assert entry.effective_confidence(50) == pytest.approx(0.9)

    def test_confidence_monotonically_decreases(self):
        entry = KnowledgeEntry(
            id="k-0", topic="resources", source_colonist="kira-sol",
            year_created=0, content="test", confidence=1.0, conditions={},
        )
        prev = 1.0
        for year in range(1, 101):
            ec = entry.effective_confidence(year)
            assert ec <= prev, f"Confidence increased at year {year}"
            prev = ec

    def test_roundtrip_serialization(self):
        entry = KnowledgeEntry(
            id="k-7", topic="governance", source_colonist="pax-stone",
            year_created=42, content="council works", confidence=0.75,
            conditions={"depth": 2, "resources": {"food": 0.6}},
            outcome=0.8, last_validated_year=50,
        )
        d = entry.to_dict()
        restored = KnowledgeEntry.from_dict(d)
        assert restored.id == entry.id
        assert restored.topic == entry.topic
        assert restored.confidence == entry.confidence
        assert restored.conditions == entry.conditions


# ---------------------------------------------------------------------------
# Tradition tests
# ---------------------------------------------------------------------------


class TestTradition:
    """Tests for cultural traditions."""

    def test_bonus_capped_at_double_strength(self):
        trad = Tradition(id="t-1", name="Test", action="farm",
                         year_formed=10, strength=5.0)
        # strength capped at 2.0 in bonus calc
        from src.mars100.collective import TRADITION_BONUS
        assert trad.bonus() == pytest.approx(TRADITION_BONUS * 2.0)

    def test_roundtrip_serialization(self):
        trad = Tradition(id="t-1", name="Harvest Festival", action="farm",
                         year_formed=15, strength=1.5, streak_years=7,
                         participants=8)
        d = trad.to_dict()
        restored = Tradition.from_dict(d)
        assert restored.id == trad.id
        assert restored.name == trad.name
        assert restored.strength == trad.strength
        assert restored.streak_years == trad.streak_years


# ---------------------------------------------------------------------------
# CollectiveMemory tests
# ---------------------------------------------------------------------------


class TestCollectiveMemory:
    """Tests for the collective memory organ."""

    def _make_cm(self) -> CollectiveMemory:
        return CollectiveMemory()

    # -- Snapshot semantics --

    def test_snapshot_semantics_writes_not_visible_until_next_snapshot(self):
        cm = self._make_cm()
        cm.snapshot()
        cm.store_knowledge("resources", "kira-sol", 1, "food tip",
                           0.8, conditions={})
        # Query should see nothing (snapshot was taken before write)
        results = cm.query("resources", 1)
        assert len(results) == 0

    def test_snapshot_semantics_writes_visible_after_next_snapshot(self):
        cm = self._make_cm()
        cm.snapshot()
        cm.store_knowledge("resources", "kira-sol", 1, "food tip",
                           0.8, conditions={})
        cm.snapshot()  # new tick
        results = cm.query("resources", 1)
        assert len(results) == 1

    # -- Store and query --

    def test_store_and_query_basic(self):
        cm = self._make_cm()
        cm.store_knowledge("resources", "fen-marsh", 5, "water trick",
                           0.9, conditions={"action": "terraform"})
        cm.snapshot()
        results = cm.query("resources", 5)
        assert len(results) == 1
        assert results[0].content == "water trick"

    def test_query_filters_by_topic(self):
        cm = self._make_cm()
        cm.store_knowledge("resources", "a", 1, "res", 0.9, {})
        cm.store_knowledge("governance", "b", 1, "gov", 0.9, {})
        cm.snapshot()
        assert len(cm.query("resources", 1)) == 1
        assert len(cm.query("governance", 1)) == 1
        assert len(cm.query("exploration", 1)) == 0

    def test_query_filters_low_confidence(self):
        cm = self._make_cm()
        cm.store_knowledge("resources", "a", 1, "old", 0.5, {})
        cm.snapshot()
        # At year 100, effective confidence of 0.5 with 99 years of decay
        # should be well below MIN_CONFIDENCE
        results = cm.query("resources", 100)
        assert len(results) == 0

    def test_query_returns_sorted_by_confidence(self):
        cm = self._make_cm()
        cm.store_knowledge("resources", "a", 10, "low", 0.4, {})
        cm.store_knowledge("resources", "b", 10, "high", 0.9, {})
        cm.store_knowledge("resources", "c", 10, "mid", 0.6, {})
        cm.snapshot()
        results = cm.query("resources", 10, max_results=3)
        confs = [r.effective_confidence(10) for r in results]
        assert confs == sorted(confs, reverse=True)

    def test_query_respects_max_results(self):
        cm = self._make_cm()
        for i in range(20):
            cm.store_knowledge("resources", "a", 1, f"entry-{i}", 0.9, {})
        cm.snapshot()
        results = cm.query("resources", 1, max_results=5)
        assert len(results) == 5

    # -- Archive pruning --

    def test_archive_bounded_size(self):
        cm = self._make_cm()
        for i in range(MAX_ARCHIVE_SIZE + 50):
            cm.store_knowledge("resources", "a", i % 100,
                               f"entry-{i}", 0.5, {})
        assert len(cm.archive) <= MAX_ARCHIVE_SIZE

    def test_pruning_removes_lowest_confidence(self):
        cm = self._make_cm()
        cm.store_knowledge("resources", "a", 1, "old-weak", 0.1, {})
        for i in range(MAX_ARCHIVE_SIZE):
            cm.store_knowledge("resources", "b", 50, f"strong-{i}", 0.9, {})
        # The old weak entry should have been pruned
        ids = {e.id for e in cm.archive}
        assert "k-0" not in ids

    # -- Archive stats --

    def test_archive_stats(self):
        cm = self._make_cm()
        cm.store_knowledge("resources", "a", 1, "r1", 0.8, {})
        cm.store_knowledge("governance", "b", 1, "g1", 0.7, {})
        stats = cm.archive_stats(1)
        assert stats["total_entries"] == 2
        assert stats["by_topic"]["resources"] == 1
        assert stats["by_topic"]["governance"] == 1

    # -- Action bias --

    def test_action_bias_empty_archive(self):
        cm = self._make_cm()
        cm.snapshot()
        bias = cm.action_bias("resources", 1)
        assert bias == {}

    def test_action_bias_capped(self):
        cm = self._make_cm()
        for i in range(50):
            cm.store_knowledge("resources", "a", 1, f"e-{i}", 0.99,
                               conditions={"action": "farm"}, outcome=10.0)
        cm.snapshot()
        bias = cm.action_bias("resources", 1)
        if "farm" in bias:
            assert abs(bias["farm"]) <= ARCHIVE_INFLUENCE_CAP + 0.001

    # -- Tradition lifecycle --

    def test_tradition_forms_after_threshold(self):
        cm = self._make_cm()
        for year in range(1, TRADITION_THRESHOLD + 2):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        assert len(cm.traditions) >= 1
        assert cm.traditions[0].action == "farm"

    def test_tradition_does_not_form_below_threshold(self):
        cm = self._make_cm()
        for year in range(1, TRADITION_THRESHOLD - 1):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        assert len(cm.traditions) == 0

    def test_tradition_weakens_without_support(self):
        cm = self._make_cm()
        # Form tradition
        for year in range(1, TRADITION_THRESHOLD + 2):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        assert len(cm.traditions) >= 1
        farm = [t for t in cm.traditions if t.action == "farm"]
        assert len(farm) == 1
        initial_strength = farm[0].strength
        # Action drops below 15%
        for year in range(TRADITION_THRESHOLD + 2, TRADITION_THRESHOLD + 12):
            cm.update_traditions(year, {"farm": 1, "code": 9}, 10)
        # Should have weakened or died
        farm = [t for t in cm.traditions if t.action == "farm"]
        if farm:
            assert farm[0].strength < initial_strength
        # If enough years pass, it should die
        for year in range(TRADITION_THRESHOLD + 12, TRADITION_THRESHOLD + 30):
            cm.update_traditions(year, {"farm": 0, "code": 10}, 10)
        farm_traditions = [t for t in cm.traditions if t.action == "farm"]
        assert len(farm_traditions) == 0

    def test_tradition_strengthens_with_continued_support(self):
        cm = self._make_cm()
        for year in range(1, TRADITION_THRESHOLD + 10):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        assert cm.traditions[0].strength > 1.0

    def test_no_duplicate_traditions(self):
        cm = self._make_cm()
        for year in range(1, TRADITION_THRESHOLD * 3):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        farm_traditions = [t for t in cm.traditions if t.action == "farm"]
        assert len(farm_traditions) == 1

    def test_tradition_bonuses(self):
        cm = self._make_cm()
        for year in range(1, TRADITION_THRESHOLD + 3):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        bonuses = cm.tradition_bonuses()
        assert "hydroponics" in bonuses
        assert bonuses["hydroponics"] > 0

    # -- Knowledge extraction from sub-sims --

    def test_extract_from_subsim_success(self):
        cm = self._make_cm()
        subsim = {
            "depth": 2, "colonist_id": "kira-sol",
            "expression": "(+ food water)", "result": 0.7,
        }
        resources = {"food": 0.6, "water": 0.5, "power": 0.7, "air": 0.8, "medicine": 0.4}
        entry = cm.extract_from_subsim(subsim, 25, resources)
        assert entry is not None
        assert entry.topic == "resources"
        assert entry.confidence > 0.4

    def test_extract_from_subsim_ignores_errors(self):
        cm = self._make_cm()
        subsim = {"depth": 1, "colonist_id": "a", "expression": "(/ 1 0)",
                   "result": None, "error": "division by zero"}
        entry = cm.extract_from_subsim(subsim, 10, {})
        assert entry is None

    def test_extract_from_subsim_ignores_non_numeric(self):
        cm = self._make_cm()
        subsim = {"depth": 1, "colonist_id": "a", "expression": "(list 1 2)",
                   "result": [1, 2]}
        entry = cm.extract_from_subsim(subsim, 10, {})
        assert entry is None

    def test_classify_topic_resources(self):
        cm = self._make_cm()
        assert cm._classify_topic("(+ food water)") == "resources"
        assert cm._classify_topic("(if (> power 0.5) 1 0)") == "resources"

    def test_classify_topic_governance(self):
        cm = self._make_cm()
        assert cm._classify_topic("(+ trust vote)") == "governance"

    def test_classify_topic_fallback(self):
        cm = self._make_cm()
        assert cm._classify_topic("(+ x y)") == "survival"

    # -- Serialization --

    def test_roundtrip_serialization(self):
        cm = self._make_cm()
        cm.store_knowledge("resources", "a", 1, "tip", 0.8,
                           {"action": "farm"}, outcome=0.5)
        for year in range(1, TRADITION_THRESHOLD + 3):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        d = cm.to_dict()
        restored = CollectiveMemory.from_dict(d)
        assert len(restored.archive) == len(cm.archive)
        assert len(restored.traditions) == len(cm.traditions)
        assert restored.archive[0].content == "tip"
        assert restored.traditions[0].action == "farm"

    def test_from_dict_empty(self):
        """Gracefully handles empty/missing data (state migration)."""
        cm = CollectiveMemory.from_dict({})
        assert len(cm.archive) == 0
        assert len(cm.traditions) == 0

    # -- Determinism --

    def test_deterministic_given_same_inputs(self):
        """Same sequence of operations produces identical state."""
        def build_memory():
            cm = CollectiveMemory()
            cm.store_knowledge("resources", "a", 1, "tip", 0.8, {}, outcome=0.5)
            cm.store_knowledge("governance", "b", 2, "gov", 0.7, {}, outcome=0.3)
            for year in range(1, 10):
                cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
            return cm.to_dict()
        d1 = build_memory()
        d2 = build_memory()
        assert d1 == d2

    # -- Events emitted --

    def test_events_emitted_on_store(self):
        cm = self._make_cm()
        cm.store_knowledge("resources", "a", 1, "tip", 0.8, {})
        assert len(cm.events) == 1
        assert cm.events[0]["type"] == "knowledge_stored"

    def test_events_emitted_on_tradition_form(self):
        cm = self._make_cm()
        for year in range(1, TRADITION_THRESHOLD + 3):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        tradition_events = [e for e in cm.events if e["type"] == "tradition_formed"]
        assert len(tradition_events) >= 1

    def test_events_emitted_on_tradition_death(self):
        cm = self._make_cm()
        # Form
        for year in range(1, TRADITION_THRESHOLD + 3):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        # Kill
        for year in range(TRADITION_THRESHOLD + 3, TRADITION_THRESHOLD + 30):
            cm.update_traditions(year, {"farm": 0, "code": 10}, 10)
        death_events = [e for e in cm.events if e["type"] == "tradition_died"]
        assert len(death_events) >= 1

    # -- Property-based invariants --

    def test_archive_never_exceeds_max_size(self):
        """Property: archive size is always bounded."""
        cm = self._make_cm()
        rng = random.Random(42)
        for year in range(1, 500):
            topic = rng.choice(list(TOPICS))
            cm.store_knowledge(topic, "test", year, f"entry-{year}",
                               rng.random(), {})
            assert len(cm.archive) <= MAX_ARCHIVE_SIZE

    def test_confidence_always_in_bounds(self):
        """Property: confidence values are always [0, 1]."""
        cm = self._make_cm()
        cm.store_knowledge("resources", "a", 1, "test", 1.5, {})
        cm.store_knowledge("resources", "b", 1, "test", -0.5, {})
        for entry in cm.archive:
            assert 0.0 <= entry.confidence <= 1.0

    def test_tradition_strength_bounded(self):
        """Property: tradition strength doesn't grow unbounded."""
        cm = self._make_cm()
        for year in range(1, 200):
            cm.update_traditions(year, {"farm": 8, "code": 2}, 10)
        for trad in cm.traditions:
            assert trad.strength <= 3.0
