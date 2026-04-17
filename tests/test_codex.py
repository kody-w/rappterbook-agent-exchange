"""Tests for the Memory Codex — colony cultural memory."""
from __future__ import annotations

import math
import pytest
from src.mars100.codex import (
    Codex, CodexEntry, ACTIVE_THRESHOLD, DECAY_RATES, MAX_ENTRIES,
    imprint_child,
)


# ── CodexEntry ─────────────────────────────────────────────────────
class TestCodexEntry:
    def test_defaults(self):
        e = CodexEntry(event_name="storm", entry_type="event")
        assert e.strength == 1.0
        assert e.impact == 0.5
        assert e.year_added == 0

    def test_roundtrip(self):
        e = CodexEntry("storm", "event", strength=0.75, impact=0.9, year_added=5, detail="big")
        d = e.to_dict()
        e2 = CodexEntry.from_dict(d)
        assert e2.event_name == "storm"
        assert abs(e2.strength - 0.75) < 1e-5
        assert e2.detail == "big"

    def test_from_dict_defaults(self):
        e = CodexEntry.from_dict({"event_name": "x", "entry_type": "law"})
        assert e.strength == 1.0


# ── Codex core ─────────────────────────────────────────────────────
class TestCodex:
    def test_empty(self):
        c = Codex()
        assert c.get_active() == []
        assert c.snapshot()["total_entries"] == 0

    def test_add_event(self):
        c = Codex()
        c.add_event("dust_storm", impact=0.8)
        assert len(c.entries) == 1
        assert c.entries[0].entry_type == "event"

    def test_duplicate_event_reinforces(self):
        c = Codex()
        c.add_event("dust_storm", impact=0.8)
        c.tick_decay()  # decay so strength < 1.0
        old = c.entries[0].strength
        c.add_event("dust_storm", impact=0.8)
        assert len(c.entries) == 1
        assert c.entries[0].strength > old

    def test_reinforce_cap(self):
        c = Codex()
        c.add_event("x", impact=0.5)
        for _ in range(50):
            c.reinforce("x", 1.0)
        assert c.entries[0].strength <= 1.0

    def test_add_ancestor_wisdom(self):
        c = Codex()
        c.add_ancestor_wisdom("col-0", [{"event": "storm", "valence": -0.5},
                                          {"event": "harvest", "valence": 0.3}])
        assert len(c.entries) == 1
        assert c.entries[0].entry_type == "ancestor"

    def test_add_ancestor_empty_memories(self):
        c = Codex()
        c.add_ancestor_wisdom("col-0", [])
        assert len(c.entries) == 0

    def test_add_law(self):
        c = Codex()
        c.add_law("water ration policy", year=12)
        assert c.entries[0].entry_type == "law"
        assert c.entries[0].year_added == 12

    def test_decay_event(self):
        c = Codex()
        c.add_event("storm")
        c.tick_decay()
        expected = 1.0 * (1.0 - DECAY_RATES["event"])
        assert abs(c.entries[0].strength - expected) < 1e-6

    def test_decay_law_slower(self):
        c = Codex()
        c.add_event("storm")
        c.add_law("law1")
        for _ in range(10):
            c.tick_decay()
        assert c.entries[1].strength > c.entries[0].strength

    def test_get_active_filters(self):
        c = Codex()
        c.add_event("storm")
        c.entries[0].strength = 0.01  # below threshold
        assert c.get_active() == []

    def test_reinforce_missing_noop(self):
        c = Codex()
        c.reinforce("nonexistent")  # should not raise


# ── LisPy bindings ─────────────────────────────────────────────────
class TestCodexBindings:
    def test_empty_bindings(self):
        c = Codex()
        b = c.get_bindings()
        assert b["codex-wisdom"] == 0.0
        assert b["codex-trauma"] == 0.0
        assert b["codex-law-count"] == 0.0
        assert b["codex-strength"] == 0.0
        assert b["codex-memory"] == 0.0

    def test_bindings_with_entries(self):
        c = Codex()
        c.add_event("storm", impact=0.8)
        c.add_ancestor_wisdom("col-1", [{"event": "e", "valence": 0.6}])
        c.add_law("law1")
        b = c.get_bindings()
        assert b["codex-trauma"] > 0
        assert b["codex-wisdom"] > 0
        assert b["codex-law-count"] == 1.0
        assert b["codex-memory"] == 3.0

    def test_bindings_are_all_float(self):
        c = Codex()
        c.add_event("x")
        for v in c.get_bindings().values():
            assert isinstance(v, float)


# ── Serialisation ──────────────────────────────────────────────────
class TestCodexSerialization:
    def test_roundtrip(self):
        c = Codex()
        c.add_event("storm", impact=0.7)
        c.add_law("law1")
        c.tick_decay()
        d = c.to_dict()
        c2 = Codex.from_dict(d)
        assert len(c2.entries) == len(c.entries)
        assert abs(c2.entries[0].strength - c.entries[0].strength) < 1e-6

    def test_from_dict_empty(self):
        c = Codex.from_dict({})
        assert len(c.entries) == 0

    def test_snapshot_shape(self):
        c = Codex()
        c.add_event("x")
        s = c.snapshot()
        assert "total_entries" in s
        assert "active_entries" in s
        assert "bindings" in s


# ── Limits ─────────────────────────────────────────────────────────
class TestCodexLimits:
    def test_prune_over_max(self):
        c = Codex()
        for i in range(MAX_ENTRIES + 50):
            c.add_event(f"event-{i}", impact=0.5)
        # manually set half to inactive
        for i in range(0, MAX_ENTRIES + 50, 2):
            c.entries[i].strength = 0.001
        c._prune()
        assert len(c.entries) <= MAX_ENTRIES

    def test_prune_preserves_active(self):
        c = Codex()
        for i in range(MAX_ENTRIES + 10):
            c.add_event(f"ev-{i}")
        c._prune()
        for e in c.entries:
            if e.strength >= ACTIVE_THRESHOLD:
                assert e in c.entries


# ── Child imprinting ───────────────────────────────────────────────
class TestImprintChild:
    def test_no_mutation_input(self):
        c = Codex()
        stats = {"paranoia": 0.5, "faith": 0.5, "empathy": 0.5}
        original = dict(stats)
        imprint_child(c, stats)
        assert stats == original  # input not mutated

    def test_high_trauma_nudges(self):
        c = Codex()
        for i in range(20):
            c.add_event(f"disaster-{i}", impact=0.9)
        stats = {"paranoia": 0.5, "faith": 0.5, "empathy": 0.5}
        out = imprint_child(c, stats)
        assert out["paranoia"] >= 0.5
        assert out["faith"] <= 0.5

    def test_high_wisdom_nudges(self):
        c = Codex()
        for i in range(10):
            c.add_ancestor_wisdom(f"anc-{i}", [{"event": "e", "valence": 0.8}])
        stats = {"paranoia": 0.5, "faith": 0.5, "empathy": 0.5}
        out = imprint_child(c, stats)
        assert out["empathy"] >= 0.5

    def test_empty_codex_noop(self):
        c = Codex()
        stats = {"paranoia": 0.5, "faith": 0.5, "empathy": 0.5}
        out = imprint_child(c, stats)
        assert out == stats

    def test_nudge_capped(self):
        c = Codex()
        for i in range(100):
            c.add_event(f"mega-{i}", impact=1.0)
        stats = {"paranoia": 0.99, "faith": 0.01, "empathy": 0.99}
        out = imprint_child(c, stats)
        assert out["paranoia"] <= 1.0
        assert out["faith"] >= 0.0
        assert out["empathy"] <= 1.0


# ── Decay invariants ──────────────────────────────────────────────
class TestDecayInvariants:
    def test_strength_monotonically_decreases(self):
        c = Codex()
        c.add_event("x", impact=1.0)
        prev = c.entries[0].strength
        for _ in range(50):
            c.tick_decay()
            assert c.entries[0].strength <= prev
            prev = c.entries[0].strength

    def test_law_outlives_event(self):
        c = Codex()
        c.add_event("ev")
        c.add_law("law")
        for _ in range(100):
            c.tick_decay()
        assert c.entries[1].strength > c.entries[0].strength

    def test_event_fades_below_threshold(self):
        c = Codex()
        c.add_event("ev", impact=1.0)
        years = 0
        while c.entries[0].strength >= ACTIVE_THRESHOLD and years < 200:
            c.tick_decay()
            years += 1
        assert c.entries[0].strength < ACTIVE_THRESHOLD
        assert years < 100  # events should fade in reasonable time

    def test_law_still_active_at_100(self):
        c = Codex()
        c.add_law("constitution")
        for _ in range(100):
            c.tick_decay()
        assert c.entries[0].strength >= ACTIVE_THRESHOLD
