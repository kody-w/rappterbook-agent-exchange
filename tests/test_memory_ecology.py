"""Tests for memory ecology — intergenerational cultural transmission."""
from __future__ import annotations

import random
import pytest
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, MemoryEntry
from src.mars100.memory_ecology import (
    AncestralMemory, CrystalKnowledge, ColonyMemoryBank,
    classify_memory_theme, archive_dead_colonist, inherit_parent_memories,
    teach_memory, dream_ancestral, attempt_crystallization, _theme_to_bonuses,
    apply_crystal_bonuses, decay_memories, colony_instinct, memory_ecology_summary,
)


def _make_colonist(cid="c1", name="Ada"):
    return Colonist(
        id=cid, name=name, element="fire", archetype="explorer",
        stats=ColonistStats(resolve=0.5, improvisation=0.5, empathy=0.5,
                            hoarding=0.5, faith=0.5, paranoia=0.5),
        skills=ColonistSkills(terraforming=0.5, hydroponics=0.5, mediation=0.5,
                              coding=0.5, prayer=0.5, sabotage=0.5),
        decision_expr="(+ resolve empathy)",
    )


def _make_bank(n=3):
    bank = ColonyMemoryBank()
    for i in range(n):
        bank.ancestral_memories.append(AncestralMemory(
            colonist_id=f"c{i}", colonist_name=f"Col-{i}",
            year_recorded=i, year_archived=10,
            event="dust storm hit the hab", emotional_valence=0.8,
            theme="survival", legacy_strength=0.5,
        ))
    return bank


class TestClassify:
    def test_survival(self): assert classify_memory_theme("dust storm hit") == "survival"
    def test_social(self): assert classify_memory_theme("betrayal in council vote") == "social"
    def test_cosmic(self): assert classify_memory_theme("alien signal detected") == "cosmic"
    def test_growth(self): assert classify_memory_theme("child born on farm") == "growth"
    def test_mundane(self): assert classify_memory_theme("nothing happened") == "mundane"
    def test_multi(self): assert classify_memory_theme("storm water food alliance") == "survival"


class TestAncestralMemory:
    def test_roundtrip(self):
        am = AncestralMemory("c1", "Ada", 5, 10, "storm", 0.7, "survival", 0.5)
        d = am.to_dict()
        assert d["valence"] == 0.7
        r = AncestralMemory.from_dict(d)
        assert r.emotional_valence == 0.7


class TestCrystalKnowledge:
    def test_roundtrip(self):
        ck = CrystalKnowledge("survival", 0.6, ["c1"], 15, {"resolve": 0.02})
        r = CrystalKnowledge.from_dict(ck.to_dict())
        assert r.stat_bonuses == {"resolve": 0.02}


class TestColonyMemoryBank:
    def test_empty(self):
        assert ColonyMemoryBank().to_dict()["stats"]["total_ancestors"] == 0

    def test_roundtrip(self):
        bank = _make_bank(2)
        bank.total_dreams = 5
        r = ColonyMemoryBank.from_dict(bank.to_dict())
        assert len(r.ancestral_memories) == 2 and r.total_dreams == 5


class TestArchive:
    def test_basic(self):
        col = _make_colonist()
        col.add_memory(1, "dust storm hit", 0.9)
        col.add_memory(2, "found water", -0.5)
        col.add_memory(3, "quiet day", 0.1)
        bank = ColonyMemoryBank()
        assert archive_dead_colonist(col, 10, bank, max_memories=2) == 2
        assert bank.ancestral_memories[0].emotional_valence == 0.9

    def test_empty(self):
        assert archive_dead_colonist(_make_colonist(), 10, ColonyMemoryBank()) == 0

    def test_cap(self):
        col = _make_colonist()
        for i in range(10):
            col.add_memory(i, f"event {i}", float(i) / 10)
        bank = ColonyMemoryBank()
        assert archive_dead_colonist(col, 20, bank, max_memories=5) == 5

    def test_legacy_strength(self):
        col = _make_colonist()
        for i in range(50):
            col.add_memory(i, f"event {i}", 0.5)
        bank = ColonyMemoryBank()
        archive_dead_colonist(col, 60, bank, max_memories=3)
        assert bank.ancestral_memories[0].legacy_strength == 1.0


class TestInherit:
    def test_basic(self):
        rng = random.Random(42)
        bank = _make_bank(3)
        child = _make_colonist("ch", "Junior")
        parent = _make_colonist("c0", "Col-0")
        assert inherit_parent_memories(child, [parent], bank, rng) > 0
        assert any("[inherited]" in m.event for m in child.memories)

    def test_no_ancestors(self):
        assert inherit_parent_memories(_make_colonist(), [], ColonyMemoryBank(), random.Random(42)) == 0

    def test_dampened(self):
        rng = random.Random(42)
        bank = ColonyMemoryBank()
        bank.ancestral_memories.append(AncestralMemory("c0", "P", 1, 10, "storm", 1.0, "survival", 1.0))
        child = _make_colonist("ch", "Jr")
        inherit_parent_memories(child, [_make_colonist("c0", "P")], bank, rng)
        if child.memories:
            assert abs(child.memories[0].emotional_valence) <= 0.31


class TestTeach:
    def test_success(self):
        t = _make_colonist("t", "Teacher")
        t.add_memory(1, "discovery", 0.9)
        s = _make_colonist("s", "Student")
        bank = ColonyMemoryBank()
        assert teach_memory(t, s, bank, random.Random(42))
        assert bank.total_teachings == 1
        assert "[taught by Teacher]" in s.memories[0].event

    def test_no_memories(self):
        assert not teach_memory(_make_colonist(), _make_colonist("s"), ColonyMemoryBank(), random.Random(42))

    def test_dampened(self):
        t = _make_colonist()
        t.add_memory(1, "event", 1.0)
        s = _make_colonist("s", "Student")
        teach_memory(t, s, ColonyMemoryBank(), random.Random(42))
        assert abs(s.memories[0].emotional_valence) == pytest.approx(0.8)


class TestDream:
    def test_with_ancestors(self):
        rng = random.Random(1)
        bank = _make_bank(3)
        col = _make_colonist()
        col.stats.faith = 1.0
        results = [dream_ancestral(col, bank, rng) for _ in range(20)]
        assert any(r is not None for r in results) and bank.total_dreams > 0

    def test_empty_bank(self):
        assert dream_ancestral(_make_colonist(), ColonyMemoryBank(), random.Random(42)) is None

    def test_adds_memory(self):
        rng = random.Random(1)
        bank = _make_bank(3)
        col = _make_colonist()
        col.stats.faith = 1.0
        for _ in range(50):
            dream_ancestral(col, bank, rng)
        assert any("[dream of" in m.event for m in col.memories)


class TestCrystallization:
    def test_enough_contributors(self):
        crystal = attempt_crystallization(_make_bank(3), 20, min_contributors=3)
        assert crystal and crystal.theme == "survival"

    def test_too_few(self):
        assert attempt_crystallization(_make_bank(2), 20, min_contributors=3) is None

    def test_no_duplicate(self):
        bank = _make_bank(3)
        attempt_crystallization(bank, 20)
        assert attempt_crystallization(bank, 21) is None


class TestBonuses:
    def test_applied(self):
        bank = _make_bank(3)
        attempt_crystallization(bank, 20)
        col = _make_colonist()
        old = col.stats.resolve
        apply_crystal_bonuses([col], bank)
        assert col.stats.resolve > old

    def test_capped(self):
        bank = _make_bank(3)
        attempt_crystallization(bank, 20)
        for c in bank.crystals:
            c.stat_bonuses = {"resolve": 10.0}
        col = _make_colonist()
        col.stats.resolve = 0.99
        apply_crystal_bonuses([col], bank)
        assert col.stats.resolve <= 1.0


class TestDecay:
    def test_reduces(self):
        bank = _make_bank(3)
        old = bank.ancestral_memories[0].legacy_strength
        decay_memories(bank, rate=0.1)
        assert bank.ancestral_memories[0].legacy_strength < old

    def test_removes_weak(self):
        bank = ColonyMemoryBank()
        bank.ancestral_memories.append(AncestralMemory("c0", "X", 1, 2, "old", 0.1, "mundane", 0.02))
        assert decay_memories(bank, rate=0.99) == 1

    def test_crystal_decay(self):
        bank = _make_bank(3)
        attempt_crystallization(bank, 20)
        old = bank.crystals[0].strength
        decay_memories(bank, rate=0.1)
        assert bank.crystals[0].strength < old


class TestInstinct:
    def test_proportions(self):
        assert abs(sum(colony_instinct(_make_bank(3)).values()) - 1.0) < 0.01

    def test_empty(self):
        assert colony_instinct(ColonyMemoryBank()) == {}


class TestSummary:
    def test_structure(self):
        bank = _make_bank(3)
        bank.total_dreams = 10
        s = memory_ecology_summary(bank)
        assert s["total_ancestors"] == 3 and s["strongest_ancestor"] is not None

    def test_empty(self):
        assert memory_ecology_summary(ColonyMemoryBank())["strongest_ancestor"] is None


class TestThemeBonuses:
    def test_known(self):
        for t in ["survival", "social", "cosmic", "growth"]:
            assert all(v > 0 for v in _theme_to_bonuses(t, 1.0).values())

    def test_scaling(self):
        b1 = _theme_to_bonuses("survival", 0.5)
        b2 = _theme_to_bonuses("survival", 1.0)
        assert b2["resolve"] == pytest.approx(b1["resolve"] * 2.0)
