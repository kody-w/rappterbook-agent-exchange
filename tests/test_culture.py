"""Tests for the Mars-100 culture engine."""
from __future__ import annotations

import random

import pytest

from src.mars100.culture import (
    ANNUAL_DECAY,
    INITIAL_STRENGTH,
    MAX_MEMORIES,
    MYTH_MIN_AGE,
    MYTH_RETELL_THRESHOLD,
    NORM_MIN_RITUAL_AGE,
    RETELL_RECOVERY,
    RITUAL_CARRIER_FRACTION,
    STAGE_MYTH,
    STAGE_NORM,
    STAGE_RITUAL,
    STAGE_STORY,
    STAGES,
    CulturalMemory,
    CultureState,
    _check_promotion,
    create_memory_from_event,
    cultural_summary_for_emergence,
    generate_narratives,
    tick_culture,
    transmit_to_child,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def culture() -> CultureState:
    return CultureState()


@pytest.fixture
def three_colonists() -> list[str]:
    return ["c-0", "c-1", "c-2"]


# ---------------------------------------------------------------------------
# CulturalMemory
# ---------------------------------------------------------------------------

class TestCulturalMemory:
    def test_default_values(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="storm",
                             narrative="The great storm")
        assert mem.stage == STAGE_STORY
        assert mem.strength == INITIAL_STRENGTH
        assert mem.alive is False  # no carriers
        assert mem.age(10) == 9

    def test_alive_requires_carriers_and_strength(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n", carriers=["c-0"])
        assert mem.alive is True
        mem.strength = 0.0
        assert mem.alive is False

    def test_roundtrip_serialization(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=5, origin_event="dust storm",
                             narrative="We survived", stage=STAGE_MYTH,
                             strength=0.6, carriers=["c-1", "c-2"],
                             retellings=7, mutations=2, codified_year=None)
        restored = CulturalMemory.from_dict(mem.to_dict())
        assert restored.id == mem.id
        assert restored.stage == STAGE_MYTH
        assert restored.strength == pytest.approx(0.6, abs=0.01)
        assert restored.carriers == ["c-1", "c-2"]
        assert restored.retellings == 7
        assert restored.mutations == 2

    def test_codified_year_in_dict_only_when_set(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n")
        assert "codified_year" not in mem.to_dict()
        mem.codified_year = 50
        assert mem.to_dict()["codified_year"] == 50


# ---------------------------------------------------------------------------
# CultureState
# ---------------------------------------------------------------------------

class TestCultureState:
    def test_empty_state(self, culture: CultureState) -> None:
        assert culture.living_memories() == []
        assert culture.by_stage(STAGE_STORY) == []

    def test_roundtrip(self, culture: CultureState, three_colonists: list[str]) -> None:
        create_memory_from_event(culture, 1, "ev", "narr", three_colonists)
        d = culture.to_dict()
        restored = CultureState.from_dict(d)
        assert len(restored.memories) == 1
        assert restored.total_created == 1


# ---------------------------------------------------------------------------
# create_memory_from_event
# ---------------------------------------------------------------------------

class TestCreateMemory:
    def test_creates_story(self, culture: CultureState,
                           three_colonists: list[str]) -> None:
        mem = create_memory_from_event(culture, 1, "dust storm",
                                       "The Great Dust Storm", three_colonists)
        assert mem.stage == STAGE_STORY
        assert mem.carriers == three_colonists
        assert culture.total_created == 1
        assert len(culture.memories) == 1

    def test_increments_id(self, culture: CultureState) -> None:
        create_memory_from_event(culture, 1, "a", "A", ["c-0"])
        create_memory_from_event(culture, 2, "b", "B", ["c-0"])
        assert culture.memories[0].id == "mem-0"
        assert culture.memories[1].id == "mem-1"
        assert culture.next_id == 2


# ---------------------------------------------------------------------------
# tick_culture
# ---------------------------------------------------------------------------

class TestTickCulture:
    def test_decay_reduces_strength(self, culture: CultureState,
                                     rng: random.Random) -> None:
        create_memory_from_event(culture, 1, "ev", "n", ["c-0"])
        result = tick_culture(culture, 2, ["c-0"], [], [], 0, rng)
        assert culture.memories[0].strength < INITIAL_STRENGTH
        assert culture.memories[0].strength == pytest.approx(
            INITIAL_STRENGTH - ANNUAL_DECAY, abs=0.001)

    def test_dead_carriers_removed(self, culture: CultureState,
                                    rng: random.Random) -> None:
        create_memory_from_event(culture, 1, "ev", "n", ["c-0", "c-1"])
        tick_culture(culture, 2, ["c-0"], ["c-1"], [], 0, rng)
        assert "c-1" not in culture.memories[0].carriers

    def test_memory_dies_when_all_carriers_dead(self, culture: CultureState,
                                                 rng: random.Random) -> None:
        create_memory_from_event(culture, 1, "ev", "n", ["c-0"])
        result = tick_culture(culture, 2, [], ["c-0"], [], 0, rng)
        assert result["died_count"] >= 1

    def test_cooperation_spreads_memory(self, culture: CultureState,
                                         rng: random.Random) -> None:
        create_memory_from_event(culture, 1, "ev", "n", ["c-0"])
        tick_culture(culture, 2, ["c-0", "c-1"], [], [("c-0", "c-1")], 0, rng)
        assert "c-1" in culture.memories[0].carriers
        assert culture.memories[0].retellings >= 1

    def test_cooperation_restores_strength(self, culture: CultureState,
                                            rng: random.Random) -> None:
        mem = create_memory_from_event(culture, 1, "ev", "n", ["c-0"])
        mem.strength = 0.3
        tick_culture(culture, 2, ["c-0", "c-1"], [], [("c-0", "c-1")], 0, rng)
        assert culture.memories[0].strength > 0.3

    def test_subsim_boost(self, culture: CultureState,
                           rng: random.Random) -> None:
        mem = create_memory_from_event(culture, 1, "ev", "n", ["c-0"])
        strength_before = mem.strength
        tick_culture(culture, 2, ["c-0"], [], [], 5, rng)
        # subsim_count > 3 → +0.02 minus decay
        expected = strength_before - ANNUAL_DECAY + 0.02
        assert culture.memories[0].strength == pytest.approx(expected, abs=0.001)

    def test_max_memories_cap(self, culture: CultureState,
                               rng: random.Random) -> None:
        for i in range(MAX_MEMORIES + 10):
            create_memory_from_event(culture, i + 1, f"ev-{i}", f"n-{i}", ["c-0"])
        tick_culture(culture, MAX_MEMORIES + 11, ["c-0"], [], [], 0, rng)
        assert len(culture.memories) <= MAX_MEMORIES

    def test_norms_survive_pruning(self, culture: CultureState,
                                    rng: random.Random) -> None:
        mem = create_memory_from_event(culture, 1, "ev", "n", ["c-0"])
        mem.stage = STAGE_NORM
        mem.codified_year = 50
        mem.strength = 0.0  # dead norm should survive
        mem.carriers = []
        # Fill to cap with other memories
        for i in range(MAX_MEMORIES + 5):
            create_memory_from_event(culture, i + 2, f"ev-{i}", f"n-{i}", ["c-0"])
        tick_culture(culture, MAX_MEMORIES + 10, ["c-0"], [], [], 0, rng)
        norm_ids = [m.id for m in culture.memories if m.stage == STAGE_NORM]
        assert "mem-0" in norm_ids


# ---------------------------------------------------------------------------
# _check_promotion
# ---------------------------------------------------------------------------

class TestPromotion:
    def test_story_to_myth(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n", carriers=["c-0"],
                             retellings=MYTH_RETELL_THRESHOLD)
        result = _check_promotion(mem, 1 + MYTH_MIN_AGE, 10)
        assert result == STAGE_MYTH

    def test_story_not_promoted_too_young(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n", carriers=["c-0"],
                             retellings=MYTH_RETELL_THRESHOLD)
        assert _check_promotion(mem, 2, 10) is None

    def test_story_not_promoted_too_few_retellings(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n", carriers=["c-0"], retellings=2)
        assert _check_promotion(mem, 1 + MYTH_MIN_AGE, 10) is None

    def test_myth_to_ritual(self) -> None:
        carriers = [f"c-{i}" for i in range(6)]
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n", stage=STAGE_MYTH,
                             carriers=carriers, strength=0.5)
        result = _check_promotion(mem, 20, 10)
        assert result == STAGE_RITUAL

    def test_myth_not_promoted_low_carrier_fraction(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n", stage=STAGE_MYTH,
                             carriers=["c-0"], strength=0.5)
        assert _check_promotion(mem, 20, 10) is None

    def test_ritual_to_norm(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n", stage=STAGE_RITUAL,
                             carriers=["c-0"], strength=0.6)
        result = _check_promotion(mem, 1 + NORM_MIN_RITUAL_AGE, 10)
        assert result == STAGE_NORM

    def test_ritual_not_promoted_too_young(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n", stage=STAGE_RITUAL,
                             carriers=["c-0"], strength=0.6)
        assert _check_promotion(mem, 5, 10) is None

    def test_norm_stays_norm(self) -> None:
        mem = CulturalMemory(id="m0", origin_year=1, origin_event="ev",
                             narrative="n", stage=STAGE_NORM,
                             carriers=["c-0"], strength=0.9)
        assert _check_promotion(mem, 100, 10) is None


# ---------------------------------------------------------------------------
# transmit_to_child
# ---------------------------------------------------------------------------

class TestTransmitToChild:
    def test_strong_memory_transmitted(self, culture: CultureState,
                                        rng: random.Random) -> None:
        mem = create_memory_from_event(culture, 1, "ev", "n", ["c-0", "c-1"])
        mem.strength = 1.0
        count = transmit_to_child(culture, "child-0", ["c-0", "c-1"], rng)
        assert count >= 1
        assert "child-0" in mem.carriers

    def test_weak_memory_less_likely(self, culture: CultureState) -> None:
        """Weak memories transmitted less often (statistical)."""
        rng = random.Random(99)
        mem = create_memory_from_event(culture, 1, "ev", "n", ["c-0"])
        mem.strength = 0.1
        transmitted = 0
        for trial in range(100):
            rng_t = random.Random(trial)
            cs = CultureState()
            m = create_memory_from_event(cs, 1, "ev", "n", ["c-0"])
            m.strength = 0.1
            transmitted += transmit_to_child(cs, "child-0", ["c-0"], rng_t)
        assert transmitted < 80  # should be much less than 100

    def test_no_parents_no_transmission(self, culture: CultureState,
                                         rng: random.Random) -> None:
        create_memory_from_event(culture, 1, "ev", "n", ["c-0"])
        count = transmit_to_child(culture, "child-0", ["c-99"], rng)
        assert count == 0


# ---------------------------------------------------------------------------
# generate_narratives
# ---------------------------------------------------------------------------

class TestGenerateNarratives:
    def test_high_severity_event_generates_story(self, rng: random.Random) -> None:
        events = [{"severity": 0.8, "description": "massive dust storm"}]
        result = generate_narratives(events, [], None, 5, rng)
        assert len(result) >= 1
        assert "dust storm" in result[0][0]

    def test_low_severity_skipped(self, rng: random.Random) -> None:
        events = [{"severity": 0.2, "description": "breeze"}]
        result = generate_narratives(events, [], None, 5, rng)
        assert len(result) == 0

    def test_death_generates_story(self, rng: random.Random) -> None:
        deaths = [{"name": "Nova Sol", "cause": "radiation"}]
        result = generate_narratives([], deaths, None, 5, rng)
        assert len(result) == 1
        assert "Nova Sol" in result[0][1]

    def test_governance_generates_story(self, rng: random.Random) -> None:
        gov = {"passed": True, "gov_type": "council"}
        result = generate_narratives([], [], gov, 5, rng)
        assert len(result) == 1

    def test_max_two_per_year(self, rng: random.Random) -> None:
        events = [{"severity": 0.9, "description": f"ev-{i}"} for i in range(5)]
        deaths = [{"name": f"n-{i}", "cause": "c"} for i in range(5)]
        result = generate_narratives(events, deaths, None, 5, rng)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# cultural_summary_for_emergence
# ---------------------------------------------------------------------------

class TestCulturalSummary:
    def test_empty_culture(self, culture: CultureState) -> None:
        summary = cultural_summary_for_emergence(culture, 100)
        assert summary["total_created"] == 0
        assert summary["living"] == 0
        assert summary["survival_rate"] == 0.0

    def test_summary_with_memories(self, culture: CultureState) -> None:
        m1 = create_memory_from_event(culture, 1, "ev1", "n1", ["c-0"])
        m1.retellings = 3
        m1.mutations = 1
        m2 = create_memory_from_event(culture, 5, "ev2", "n2", ["c-0"])
        m2.stage = STAGE_MYTH
        summary = cultural_summary_for_emergence(culture, 100)
        assert summary["total_created"] == 2
        assert summary["living"] == 2
        assert summary["most_retold"]["id"] == "mem-0"
        assert summary["most_mutated"]["id"] == "mem-0"
        assert summary["oldest_memory"]["origin_year"] == 1


# ---------------------------------------------------------------------------
# Integration: culture through full engine tick
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    """Verify culture engine integrates with Mars100Engine."""

    def test_engine_runs_with_culture(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        assert "culture" in result.to_dict()
        culture = result.to_dict()["culture"]
        assert "total_created" in culture
        assert culture["total_created"] >= 0

    def test_culture_data_in_year_results(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "culture" in d
            assert "living_count" in d["culture"]

    def test_memories_accumulate_over_years(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        culture = result.to_dict()["culture"]
        assert culture["total_created"] > 0

    def test_version_bumped(self) -> None:
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        assert result.to_dict()["_meta"]["version"] == "5.0"

    def test_full_100_year_sim_with_culture(self) -> None:
        """Smoke test: 100 years with culture should not crash."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        culture = result.to_dict()["culture"]
        assert culture["total_created"] > 0
        # Some memories should have been promoted after 100 years
        assert culture["total_promoted"] >= 0
        # Survival rate should be between 0 and 1
        assert 0.0 <= culture["survival_rate"] <= 1.0
