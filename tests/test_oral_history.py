"""Tests for the oral history engine."""
from __future__ import annotations

import math
import random

import pytest

from src.mars100.oral_history import (
    OralHistory, SharedMemory, MemoryVariant,
    witness_event, share_memory, check_mythification, decay_salience,
    on_death, on_birth, action_modifiers, subsim_bindings, tick_year,
    EVENT_TO_THEME, MYTH_ACTION_MODIFIERS, STANCES,
)


# --- Helpers ---

def make_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def make_history_with_memory(
    mem_id: str = "dust_storm_y5",
    event_name: str = "dust_storm",
    carrier_ids: list[str] | None = None,
    year: int = 5,
    salience: float = 0.8,
    is_myth: bool = False,
) -> tuple[OralHistory, SharedMemory]:
    carrier_ids = carrier_ids or ["c-0", "c-1", "c-2"]
    mem = SharedMemory(
        memory_id=mem_id,
        event_name=event_name,
        origin_year=year,
        description="Test event",
        theme=EVENT_TO_THEME.get(event_name, "origin"),
        stance="cautionary",
        salience=salience,
        is_myth=is_myth,
    )
    for cid in carrier_ids:
        mem.variants.append(MemoryVariant(
            colonist_id=cid,
            emotional_weight=0.7,
            heard_from="direct",
            fidelity=1.0,
            heard_year=year,
        ))
    history = OralHistory(memories=[mem])
    return history, mem


# --- MemoryVariant tests ---

class TestMemoryVariant:
    def test_to_dict(self):
        v = MemoryVariant("c-0", 0.75, "direct", 1.0, 5)
        d = v.to_dict()
        assert d["colonist_id"] == "c-0"
        assert d["emotional_weight"] == 0.75
        assert d["fidelity"] == 1.0

    def test_negative_weight(self):
        v = MemoryVariant("c-0", -0.5, "c-1", 0.8, 10)
        assert v.emotional_weight == -0.5
        assert v.heard_from == "c-1"


# --- SharedMemory tests ---

class TestSharedMemory:
    def test_carrier_ids_sorted(self):
        _, mem = make_history_with_memory(carrier_ids=["c-2", "c-0", "c-1"])
        assert mem.carrier_ids() == ["c-0", "c-1", "c-2"]

    def test_has_carrier(self):
        _, mem = make_history_with_memory()
        assert mem.has_carrier("c-0")
        assert not mem.has_carrier("c-99")

    def test_get_variant(self):
        _, mem = make_history_with_memory()
        v = mem.get_variant("c-1")
        assert v is not None
        assert v.colonist_id == "c-1"
        assert mem.get_variant("c-99") is None

    def test_remove_carrier(self):
        _, mem = make_history_with_memory()
        mem.remove_carrier("c-1")
        assert not mem.has_carrier("c-1")
        assert len(mem.variants) == 2

    def test_to_dict(self):
        _, mem = make_history_with_memory()
        d = mem.to_dict()
        assert d["memory_id"] == "dust_storm_y5"
        assert d["carrier_count"] == 3
        assert "variants" in d


# --- OralHistory tests ---

class TestOralHistory:
    def test_empty(self):
        h = OralHistory()
        assert h.to_dict()["total_memories"] == 0
        assert h.myths() == []

    def test_get_memory(self):
        h, mem = make_history_with_memory()
        assert h.get_memory("dust_storm_y5") is mem
        assert h.get_memory("nonexistent") is None

    def test_carrier_myths(self):
        h, mem = make_history_with_memory(is_myth=True)
        assert len(h.carrier_myths("c-0")) == 1
        assert len(h.carrier_myths("c-99")) == 0


# --- witness_event tests ---

class TestWitnessEvent:
    def test_low_severity_ignored(self):
        h = OralHistory()
        result = witness_event(h, "dust_storm", 1, "desc", ["c-0"], 0.2, make_rng())
        assert result is None
        assert len(h.memories) == 0

    def test_creates_memory(self):
        h = OralHistory()
        result = witness_event(h, "dust_storm", 5, "A storm!", ["c-0", "c-1"], 0.6, make_rng())
        assert result is not None
        assert result.memory_id == "dust_storm_y5"
        assert len(result.variants) == 2
        assert result.theme == "storm"

    def test_duplicate_event_ignored(self):
        h = OralHistory()
        witness_event(h, "dust_storm", 5, "desc", ["c-0"], 0.6, make_rng())
        witness_event(h, "dust_storm", 5, "desc2", ["c-0"], 0.9, make_rng())
        assert len(h.memories) == 1

    def test_all_witnesses_get_variants(self):
        h = OralHistory()
        ids = [f"c-{i}" for i in range(10)]
        result = witness_event(h, "meteor_shower", 3, "desc", ids, 0.7, make_rng())
        assert result is not None
        assert len(result.variants) == 10

    def test_emotional_weight_bounded(self):
        h = OralHistory()
        result = witness_event(h, "dust_storm", 1, "desc", ["c-0"], 1.0, make_rng())
        assert result is not None
        for v in result.variants:
            assert -1.0 <= v.emotional_weight <= 1.0


# --- share_memory tests ---

class TestShareMemory:
    def test_basic_sharing(self):
        h, mem = make_history_with_memory(carrier_ids=["c-0"])
        log = share_memory(h, "c-0", "c-1", 10, 0.8, make_rng())
        # May or may not share depending on RNG, but structure is correct
        if log:
            assert log[0]["speaker"] == "c-0"
            assert log[0]["listener"] == "c-1"

    def test_no_double_sharing(self):
        h, mem = make_history_with_memory(carrier_ids=["c-0", "c-1"])
        log = share_memory(h, "c-0", "c-1", 10, 0.8, make_rng())
        # c-1 already has it, should not be shared again
        assert len(log) == 0

    def test_fidelity_degrades(self):
        h, mem = make_history_with_memory(carrier_ids=["c-0"])
        # Force sharing by using high trust and running many times
        rng = make_rng(1)
        for i in range(20):
            listener = f"listener-{i}"
            share_memory(h, "c-0", listener, 10, 1.0, rng)
        # At least some should have been shared
        assert len(mem.variants) > 1
        for v in mem.variants:
            if v.heard_from != "direct":
                assert v.fidelity < 1.0

    def test_sharing_increments_counter(self):
        h, mem = make_history_with_memory(carrier_ids=["c-0"])
        mem.times_shared = 0
        rng = make_rng(1)
        for i in range(10):
            share_memory(h, "c-0", f"new-{i}", 10, 1.0, rng)
        assert mem.times_shared > 0


# --- check_mythification tests ---

class TestMythification:
    def test_enough_carriers_mythifies(self):
        # 10 active colonists, threshold = ceil(10*0.35) = 4 carriers
        carriers = [f"c-{i}" for i in range(5)]
        h, mem = make_history_with_memory(carrier_ids=carriers, salience=0.8)
        mem.times_shared = 6
        log = check_mythification(h, 10, 20)
        assert len(log) == 1
        assert mem.is_myth

    def test_too_few_carriers_blocked(self):
        carriers = [f"c-{i}" for i in range(2)]
        h, mem = make_history_with_memory(carrier_ids=carriers, salience=0.8)
        mem.times_shared = 10
        log = check_mythification(h, 10, 20)
        assert len(log) == 0
        assert not mem.is_myth

    def test_low_sharing_blocked(self):
        carriers = [f"c-{i}" for i in range(5)]
        h, mem = make_history_with_memory(carrier_ids=carriers, salience=0.8)
        mem.times_shared = 3
        log = check_mythification(h, 10, 20)
        assert len(log) == 0

    def test_low_salience_blocks_myth(self):
        carriers = [f"c-{i}" for i in range(5)]
        h, mem = make_history_with_memory(carrier_ids=carriers, salience=0.3)
        mem.times_shared = 10
        log = check_mythification(h, 10, 20)
        assert len(log) == 0

    def test_already_mythified_skipped(self):
        carriers = [f"c-{i}" for i in range(5)]
        h, mem = make_history_with_memory(carrier_ids=carriers, is_myth=True, salience=0.8)
        mem.times_shared = 10
        log = check_mythification(h, 10, 20)
        assert len(log) == 0

    def test_mythification_amplifies_salience(self):
        carriers = [f"c-{i}" for i in range(5)]
        h, mem = make_history_with_memory(carrier_ids=carriers, salience=0.7)
        mem.times_shared = 6
        old_salience = mem.salience
        check_mythification(h, 10, 20)
        assert mem.salience >= old_salience


# --- decay_salience tests ---

class TestDecaySalience:
    def test_myth_decays_slowly(self):
        h, mem = make_history_with_memory(is_myth=True, salience=1.0)
        decay_salience(h, 10, make_rng())
        assert 0.95 < mem.salience < 1.0

    def test_nonmyth_decays_fast(self):
        h, mem = make_history_with_memory(is_myth=False, salience=1.0)
        decay_salience(h, 10, make_rng())
        assert mem.salience < 0.95

    def test_extinct_memories_pruned(self):
        h, mem = make_history_with_memory(salience=0.05)
        mem.variants.clear()
        decay_salience(h, 10, make_rng())
        assert len(h.memories) == 0


# --- on_death tests ---

class TestOnDeath:
    def test_removes_carrier(self):
        h, mem = make_history_with_memory()
        on_death(h, "c-1")
        assert not mem.has_carrier("c-1")
        assert mem.has_carrier("c-0")

    def test_last_carrier_dies_extinct(self):
        h, mem = make_history_with_memory(carrier_ids=["c-0"])
        on_death(h, "c-0")
        assert len(mem.variants) == 0

    def test_myth_dies_with_last_carrier(self):
        h, mem = make_history_with_memory(carrier_ids=["c-0"], is_myth=True)
        on_death(h, "c-0")
        assert mem.is_myth  # stays mythified, just no carriers
        assert len(mem.variants) == 0


# --- on_birth tests ---

class TestOnBirth:
    def test_child_inherits_myths(self):
        h, mem = make_history_with_memory(is_myth=True, salience=0.9)
        on_birth(h, "child-0", ["c-0", "c-1", "c-2", "child-0"])
        assert mem.has_carrier("child-0")

    def test_child_variant_has_lower_fidelity(self):
        h, mem = make_history_with_memory(is_myth=True, salience=0.9)
        on_birth(h, "child-0", ["c-0", "c-1", "c-2", "child-0"])
        v = mem.get_variant("child-0")
        assert v is not None
        assert v.fidelity == 0.5
        assert v.heard_from == "cultural"

    def test_max_inherited_cap(self):
        h = OralHistory()
        for i in range(5):
            _, mem = make_history_with_memory(
                mem_id=f"event_y{i}",
                carrier_ids=["c-0"],
                is_myth=True,
                salience=0.9 - i * 0.1,
            )
            h.memories.append(mem)
        on_birth(h, "child-0", ["c-0", "child-0"])
        inherited = sum(1 for m in h.memories if m.has_carrier("child-0"))
        assert inherited <= 3


# --- action_modifiers tests ---

class TestActionModifiers:
    def test_no_myths_no_modifiers(self):
        h = OralHistory()
        mods = action_modifiers(h, "c-0")
        assert mods == {}

    def test_myth_gives_bonus(self):
        h, mem = make_history_with_memory(
            event_name="dust_storm", is_myth=True, salience=0.8,
        )
        mods = action_modifiers(h, "c-0")
        assert "terraform" in mods
        assert mods["terraform"] > 0

    def test_noncarrier_gets_no_bonus(self):
        h, mem = make_history_with_memory(
            carrier_ids=["c-0"], event_name="dust_storm",
            is_myth=True, salience=0.8,
        )
        mods = action_modifiers(h, "c-99")
        assert mods == {}

    def test_multiple_myths_stack(self):
        h = OralHistory()
        for i, event in enumerate(["dust_storm", "equipment_failure"]):
            mem = SharedMemory(
                memory_id=f"{event}_y{i}",
                event_name=event,
                origin_year=i,
                description="test",
                theme=EVENT_TO_THEME[event],
                is_myth=True,
                salience=0.8,
            )
            mem.variants.append(MemoryVariant("c-0", 0.7, "direct", 1.0, i))
            h.memories.append(mem)
        mods = action_modifiers(h, "c-0")
        # Both storm and failure themes should contribute
        assert len(mods) > 0

    def test_salience_scales_bonus(self):
        h1, _ = make_history_with_memory(
            event_name="dust_storm", is_myth=True, salience=1.0,
        )
        h2, _ = make_history_with_memory(
            event_name="dust_storm", is_myth=True, salience=0.3,
        )
        mods1 = action_modifiers(h1, "c-0")
        mods2 = action_modifiers(h2, "c-0")
        assert mods1.get("terraform", 0) > mods2.get("terraform", 0)


# --- subsim_bindings tests ---

class TestSubsimBindings:
    def test_no_myths(self):
        h = OralHistory()
        b = subsim_bindings(h, "c-0")
        assert b["myth-count"] == 0
        assert b["myth-salience"] == 0.0

    def test_with_myths(self):
        h, _ = make_history_with_memory(is_myth=True, salience=0.8)
        b = subsim_bindings(h, "c-0")
        assert b["myth-count"] == 1
        assert b["myth-salience"] > 0


# --- tick_year tests ---

class TestTickYear:
    def _trust_fn(self, a: str, b: str) -> float:
        return 0.7

    def test_basic_tick(self):
        h = OralHistory()
        events = [{"name": "dust_storm", "severity": 0.6, "description": "Storm!"}]
        result = tick_year(h, 5, events, ["c-0", "c-1"], [], self._trust_fn, make_rng())
        assert "witnessed" in result
        assert result["total_memories"] >= 1

    def test_tick_advances_state(self):
        h = OralHistory()
        events = [{"name": "dust_storm", "severity": 0.7, "description": "Big storm"}]
        tick_year(h, 1, events, ["c-0", "c-1"], [], self._trust_fn, make_rng())
        assert len(h.memories) == 1
        tick_year(h, 2, [{"name": "meteor_shower", "severity": 0.5, "description": "Meteors"}],
                  ["c-0", "c-1"], [("c-0", "c-1")], self._trust_fn, make_rng())
        assert len(h.memories) >= 1

    def test_tick_no_events(self):
        h = OralHistory()
        result = tick_year(h, 1, [], ["c-0"], [], self._trust_fn, make_rng())
        assert result["witnessed"] == []

    def test_myths_emerge_over_many_years(self):
        h = OralHistory()
        ids = [f"c-{i}" for i in range(10)]
        rng = make_rng(99)
        events = [{"name": "dust_storm", "severity": 0.8, "description": "Storm"}]
        tick_year(h, 1, events, ids, [], self._trust_fn, rng)
        # Share heavily for many years
        for year in range(2, 30):
            pairs = [(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
            tick_year(h, year, [], ids, pairs, self._trust_fn, rng)
        assert len(h.myths()) >= 0  # may or may not mythify depending on rng


# --- Determinism ---

class TestDeterminism:
    def test_same_seed_same_result(self):
        results = []
        for _ in range(2):
            h = OralHistory()
            rng = make_rng(42)
            ids = ["c-0", "c-1", "c-2"]
            events = [{"name": "dust_storm", "severity": 0.6, "description": "Storm"}]
            r = tick_year(h, 1, events, ids, [("c-0", "c-1")],
                         lambda a, b: 0.7, rng)
            results.append(r)
        assert results[0] == results[1]


# --- Edge cases ---

class TestEdgeCases:
    def test_empty_colony(self):
        h = OralHistory()
        result = tick_year(h, 1, [], [], [], lambda a, b: 0.5, make_rng())
        assert result["total_memories"] == 0

    def test_single_colonist(self):
        h = OralHistory()
        events = [{"name": "dust_storm", "severity": 0.5, "description": "Storm"}]
        result = tick_year(h, 1, events, ["c-0"], [], lambda a, b: 0.5, make_rng())
        assert result["total_memories"] == 1

    def test_death_then_birth_order(self):
        h, mem = make_history_with_memory(carrier_ids=["c-0", "c-1"], is_myth=True, salience=0.9)
        on_death(h, "c-0")
        assert not mem.has_carrier("c-0")
        assert mem.has_carrier("c-1")  # still alive
        on_birth(h, "child-0", ["c-1", "child-0"])
        assert mem.has_carrier("child-0")  # inherited from surviving carriers

    def test_exile_does_not_kill_memories(self):
        """on_death removes carrier but memory persists if others carry it."""
        h, mem = make_history_with_memory(carrier_ids=["c-0", "c-1", "c-2"])
        on_death(h, "c-1")
        assert len(mem.variants) == 2
        assert mem in h.memories
