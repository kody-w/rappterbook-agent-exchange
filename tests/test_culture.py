"""Tests for the Mars-100 oral tradition / institutional memory system."""
from __future__ import annotations

import random
import pytest

from src.mars100.culture import (
    Tradition, OralHistory,
    TRADITION_CATEGORIES, MAX_ACTIVE_CANON, MAX_ARCHIVE,
    CATEGORY_ACTION_BIAS, BIAS_STRENGTH,
    tradition_from_death, tradition_from_governance,
    tradition_from_subsim, tradition_from_crisis, tradition_from_meta,
)


# ── Tradition dataclass ──────────────────────────────────────

class TestTradition:
    def test_creation(self):
        t = Tradition(id="t1", year_created=10, source="death",
                      category="cautionary", text="beware dust",
                      author_id="kira-sol")
        assert t.trust_rating == 0.5
        assert t.citations == 0
        assert not t.archived

    def test_to_dict_roundtrip(self):
        t = Tradition(id="t1", year_created=10, source="subsim",
                      category="technical", text="power is key",
                      author_id="rust-vega", trust_rating=0.8,
                      citations=3, archived=True)
        d = t.to_dict()
        t2 = Tradition.from_dict(d)
        assert t2.id == t.id
        assert t2.trust_rating == pytest.approx(t.trust_rating, abs=1e-3)
        assert t2.archived is True
        assert t2.citations == 3

    def test_all_categories_valid(self):
        for cat in TRADITION_CATEGORIES:
            t = Tradition(id=f"t-{cat}", year_created=1, source="test",
                          category=cat, text="x", author_id="a")
            assert t.category == cat

    def test_from_dict_defaults(self):
        t = Tradition.from_dict({"id": "t1", "year_created": 1})
        assert t.source == "unknown"
        assert t.category == "survival"
        assert t.text == ""


# ── OralHistory ──────────────────────────────────────────────

class TestOralHistory:
    def _make_tradition(self, tid: str, category: str = "survival",
                        trust: float = 0.5, archived: bool = False) -> Tradition:
        return Tradition(id=tid, year_created=1, source="test",
                         category=category, text=f"tradition {tid}",
                         author_id="a", trust_rating=trust, archived=archived)

    def test_empty(self):
        oh = OralHistory()
        assert oh.active_canon == []
        assert oh.archive == []
        assert oh.action_biases() == {}

    def test_add_via_propose_commit(self):
        rng = random.Random(42)
        oh = OralHistory()
        t = self._make_tradition("t1")
        oh.propose(t)
        assert len(oh.active_canon) == 0  # not yet committed
        accepted = oh.commit(rng)
        assert len(accepted) == 1
        assert len(oh.active_canon) == 1

    def test_merge_on_duplicate_category_source(self):
        rng = random.Random(42)
        oh = OralHistory()
        t1 = Tradition(id="t1", year_created=1, source="crisis",
                       category="survival", text="food shortage",
                       author_id="a", trust_rating=0.5)
        oh.traditions.append(t1)
        t2 = Tradition(id="t2", year_created=5, source="crisis",
                       category="survival", text="another shortage",
                       author_id="b", trust_rating=0.5)
        oh.propose(t2)
        accepted = oh.commit(rng)
        assert len(oh.active_canon) == 1  # merged, not added
        assert oh.traditions[0].citations == 1
        assert oh.traditions[0].trust_rating == pytest.approx(0.55, abs=0.01)

    def test_cap_enforcement(self):
        rng = random.Random(42)
        oh = OralHistory()
        for i in range(MAX_ACTIVE_CANON + 5):
            # use unique source per tradition to prevent merging
            t = Tradition(id=f"t{i}", year_created=i, source=f"src-{i}",
                          category="survival", text=f"tradition {i}",
                          author_id="a", trust_rating=0.3 + i * 0.01)
            oh.traditions.append(t)
        oh._enforce_cap(rng)
        assert len(oh.active_canon) <= MAX_ACTIVE_CANON
        assert len(oh.archive) >= 5

    def test_by_category(self):
        oh = OralHistory()
        oh.traditions.append(self._make_tradition("t1", "survival"))
        oh.traditions.append(self._make_tradition("t2", "governance"))
        oh.traditions.append(self._make_tradition("t3", "survival"))
        assert len(oh.by_category("survival")) == 2
        assert len(oh.by_category("governance")) == 1
        assert len(oh.by_category("technical")) == 0

    def test_by_source(self):
        oh = OralHistory()
        t1 = Tradition(id="t1", year_created=1, source="death",
                       category="cautionary", text="x", author_id="a")
        t2 = Tradition(id="t2", year_created=2, source="subsim",
                       category="technical", text="y", author_id="b")
        oh.traditions.extend([t1, t2])
        assert len(oh.by_source("death")) == 1
        assert len(oh.by_source("subsim")) == 1
        assert len(oh.by_source("governance")) == 0

    def test_action_biases_empty(self):
        oh = OralHistory()
        assert oh.action_biases() == {}

    def test_action_biases_survival(self):
        oh = OralHistory()
        oh.traditions.append(self._make_tradition("t1", "survival", trust=1.0))
        biases = oh.action_biases()
        assert biases.get("farm", 0) > 0
        assert biases.get("terraform", 0) > 0
        assert biases.get("sabotage", 0) == 0

    def test_action_biases_scale_with_trust(self):
        oh = OralHistory()
        oh.traditions.append(self._make_tradition("t1", "technical", trust=0.5))
        biases_low = oh.action_biases()
        oh.traditions[0].trust_rating = 1.0
        biases_high = oh.action_biases()
        assert biases_high.get("code", 0) > biases_low.get("code", 0)

    def test_archived_traditions_dont_bias(self):
        oh = OralHistory()
        oh.traditions.append(self._make_tradition("t1", "survival", trust=1.0,
                                                   archived=True))
        assert oh.action_biases() == {}

    def test_governance_modifier_council(self):
        oh = OralHistory()
        t = Tradition(id="t1", year_created=10, source="governance",
                      category="governance", text="Colony adopted council governance",
                      author_id="a", trust_rating=0.8)
        oh.traditions.append(t)
        mod = oh.governance_modifier("council")
        assert mod > 0

    def test_governance_modifier_bounded(self):
        oh = OralHistory()
        for i in range(10):
            t = Tradition(id=f"t{i}", year_created=i, source=f"gov-{i}",
                          category="governance",
                          text="Colony adopted council governance",
                          author_id="a", trust_rating=1.0)
            oh.traditions.append(t)
        mod = oh.governance_modifier("council")
        assert -0.3 <= mod <= 0.3

    def test_meta_awareness_boost_empty(self):
        oh = OralHistory()
        assert oh.meta_awareness_boost() == 0.0

    def test_meta_awareness_boost_positive(self):
        oh = OralHistory()
        oh.traditions.append(self._make_tradition("t1", "spiritual", trust=1.0))
        assert oh.meta_awareness_boost() > 0

    def test_drift_trust(self):
        rng = random.Random(42)
        oh = OralHistory()
        oh.traditions.append(self._make_tradition("t1", "survival", trust=0.5))
        initial = oh.traditions[0].trust_rating
        for _ in range(100):
            oh.drift_trust(rng)
        # trust should have drifted (unlikely to be exactly the same)
        assert oh.traditions[0].trust_rating != initial
        # should remain in bounds
        assert 0.0 <= oh.traditions[0].trust_rating <= 1.0

    def test_serialisation_roundtrip(self):
        oh = OralHistory()
        oh.traditions.append(self._make_tradition("t1", "survival", trust=0.8))
        oh.traditions.append(self._make_tradition("t2", "governance", trust=0.3,
                                                   archived=True))
        d = oh.to_dict()
        oh2 = OralHistory.from_dict(d)
        assert len(oh2.traditions) == 2
        assert len(oh2.active_canon) == 1
        assert len(oh2.archive) == 1

    def test_pending_cleared_after_commit(self):
        rng = random.Random(42)
        oh = OralHistory()
        oh.propose(self._make_tradition("t1"))
        oh.commit(rng)
        assert len(oh._pending) == 0


# ── Factory functions ────────────────────────────────────────

class TestTraditionFactories:
    def test_from_death(self):
        t = tradition_from_death("kira-sol", "Kira Sol", 92, "asphyxiation", 1)
        assert t.source == "death"
        assert t.category == "cautionary"
        assert "Kira Sol" in t.text
        assert "asphyxiation" in t.text
        assert t.trust_rating == 0.7

    def test_from_governance(self):
        t = tradition_from_governance(16, "council", "pax-stone", 2)
        assert t.source == "governance"
        assert t.category == "governance"
        assert "council" in t.text
        assert t.trust_rating == 0.6

    def test_from_subsim_shallow(self):
        t = tradition_from_subsim(30, "luna-tide", 2, "resource surplus predicted", 3)
        assert t.category == "technical"
        assert t.trust_rating == pytest.approx(0.7, abs=0.01)

    def test_from_subsim_deep(self):
        t = tradition_from_subsim(50, "aura-kai", 3, "meta insight", 4)
        assert t.category == "spiritual"
        assert t.trust_rating == pytest.approx(0.85, abs=0.01)

    def test_from_crisis(self):
        t = tradition_from_crisis(85, "oxygen", 5)
        assert t.source == "crisis"
        assert t.category == "survival"
        assert "oxygen" in t.text

    def test_from_meta(self):
        t = tradition_from_meta(45, "ora-flame", "we are variables", 6)
        assert t.source == "meta"
        assert t.category == "spiritual"
        assert "variables" in t.text


# ── Integration / property tests ─────────────────────────────

class TestOralHistoryProperties:
    def test_action_bias_values_non_negative(self):
        """All action biases should be >= 0 (traditions encourage, not penalise)."""
        oh = OralHistory()
        rng = random.Random(42)
        for cat in TRADITION_CATEGORIES:
            t = Tradition(id=f"t-{cat}", year_created=1, source=f"s-{cat}",
                          category=cat, text="x", author_id="a",
                          trust_rating=rng.random())
            oh.traditions.append(t)
        biases = oh.action_biases()
        for action, val in biases.items():
            assert val >= 0, f"Negative bias for {action}: {val}"

    def test_cap_never_exceeded_under_load(self):
        """Fuzz: adding many traditions never exceeds the active cap."""
        rng = random.Random(123)
        oh = OralHistory()
        for i in range(200):
            t = Tradition(id=f"t{i}", year_created=i, source=f"src-{i}",
                          category=rng.choice(TRADITION_CATEGORIES),
                          text=f"tradition {i}", author_id="a",
                          trust_rating=rng.random())
            oh.propose(t)
            if i % 5 == 0:
                oh.commit(rng)
        oh.commit(rng)
        assert len(oh.active_canon) <= MAX_ACTIVE_CANON

    def test_archive_cap_respected(self):
        """Archived traditions are pruned when exceeding MAX_ARCHIVE."""
        rng = random.Random(42)
        oh = OralHistory()
        for i in range(MAX_ARCHIVE + 50):
            t = Tradition(id=f"t{i}", year_created=i, source=f"src-{i}",
                          category="survival", text=f"old {i}",
                          author_id="a", trust_rating=0.01, archived=True)
            oh.traditions.append(t)
        # add one more active to trigger enforcement
        oh.propose(Tradition(id="new", year_created=999, source="new",
                             category="survival", text="new",
                             author_id="a", trust_rating=0.9))
        oh.commit(rng)
        assert len(oh.archive) <= MAX_ARCHIVE

    def test_trust_stays_bounded_after_many_merges(self):
        """Trust rating must always stay in [0, 1] even after many merges."""
        rng = random.Random(42)
        oh = OralHistory()
        base = Tradition(id="t1", year_created=1, source="crisis",
                         category="survival", text="x", author_id="a",
                         trust_rating=0.9)
        oh.traditions.append(base)
        for i in range(100):
            dup = Tradition(id=f"dup-{i}", year_created=i, source="crisis",
                            category="survival", text="y", author_id="b")
            oh.propose(dup)
            oh.commit(rng)
        assert 0.0 <= oh.traditions[0].trust_rating <= 1.0
