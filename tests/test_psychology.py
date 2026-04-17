"""Tests for the psychology organ (engine v8.0)."""
from __future__ import annotations

import random
import pytest
from dataclasses import dataclass, field
from typing import Any

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.mars100.psychology import (
    PsychTickResult,
    tick_psychology,
    compute_psych_pressure,
    accumulate_stress,
    update_morale,
    check_breakdown,
    form_bonds,
    process_grief,
    BREAKDOWN_STRESS_THRESHOLD,
    BREAKDOWN_RESET_STRESS,
    BREAKDOWN_MORALE_DROP,
    BOND_TRUST_THRESHOLD,
    BOND_AFFECTION_THRESHOLD,
    MAX_BONDS,
    GRIEF_DEATH,
    GRIEF_EXILE,
)


# -- test helpers ------------------------------------------------------------

@dataclass
class FakeStats:
    resolve: float = 0.5
    paranoia: float = 0.3
    empathy: float = 0.4
    improvisation: float = 0.5
    hoarding: float = 0.3
    faith: float = 0.3


@dataclass
class FakeWallet:
    food: float = 10.0
    water: float = 10.0
    power: float = 10.0
    medicine: float = 5.0

    def total_wealth(self) -> float:
        return self.food + self.water + self.power + self.medicine


@dataclass
class FakeColonist:
    id: str
    name: str
    stats: FakeStats = field(default_factory=FakeStats)
    stress: float = 0.0
    morale: float = 0.7
    bonds: list[str] = field(default_factory=list)
    breakdown_year: int | None = None
    birth_year: int = 0
    wallet: FakeWallet = field(default_factory=FakeWallet)
    death_year: int | None = None
    memories: list = field(default_factory=list)

    def is_active(self) -> bool:
        return self.death_year is None

    def add_memory(self, year: int, text: str, sentiment: float) -> None:
        self.memories.append({"year": year, "text": text, "sentiment": sentiment})


@dataclass
class FakeRelationship:
    trust: float = 0.5
    affection: float = 0.5


class FakeSocialGraph:
    def __init__(self, default_trust: float = 0.5, default_affection: float = 0.5):
        self._default = FakeRelationship(default_trust, default_affection)
        self._overrides: dict[tuple[str, str], FakeRelationship] = {}

    def get(self, from_id: str, to_id: str) -> FakeRelationship:
        return self._overrides.get((from_id, to_id), self._default)

    def set(self, from_id: str, to_id: str, trust: float, affection: float) -> None:
        self._overrides[(from_id, to_id)] = FakeRelationship(trust, affection)

    def colony_cohesion(self, ids: list[str]) -> float:
        return 0.5


@dataclass
class FakeResources:
    food: float = 0.5
    water: float = 0.5
    power: float = 0.5

    def to_dict(self) -> dict:
        return {"food": self.food, "water": self.water, "power": self.power}


class FakeCulture:
    def __init__(self, traditions: list | None = None):
        self.traditions = traditions or []


@dataclass
class FakeEvent:
    severity: float = 0.5
    name: str = "test_event"


def _make_colonists(n: int = 3) -> list[FakeColonist]:
    return [FakeColonist(id=f"col-{i}", name=f"Colonist {i}") for i in range(n)]


# -- unit tests: accumulate_stress -------------------------------------------

class TestAccumulateStress:
    def test_baseline_decay(self):
        """With no stressors, stress should decay."""
        result = accumulate_stress(0.3, 0.5, 0.3, 0.4, [], 0, 0.0, 0, 0, False, False, 0)
        assert result < 0.3

    def test_high_severity_events_increase_stress(self):
        result = accumulate_stress(0.1, 0.5, 0.3, 0.4, [0.8, 0.9], 0, 0.0, 0, 0,
                                   False, False, 0)
        assert result > 0.1

    def test_critical_resources_increase_stress(self):
        result = accumulate_stress(0.1, 0.5, 0.3, 0.4, [], 3, 0.0, 0, 0, False, False, 0)
        assert result > 0.1

    def test_bonded_deaths_increase_stress(self):
        base = accumulate_stress(0.1, 0.5, 0.3, 0.4, [], 0, 0.0, 0, 0, False, False, 0)
        grief = accumulate_stress(0.1, 0.5, 0.3, 0.4, [], 0, 0.0, 2, 0, False, False, 0)
        assert grief > base

    def test_resolve_reduces_stress(self):
        high_resolve = accumulate_stress(0.3, 0.9, 0.3, 0.4, [0.8], 0, 0.0, 0, 0,
                                          False, False, 0)
        low_resolve = accumulate_stress(0.3, 0.1, 0.3, 0.4, [0.8], 0, 0.0, 0, 0,
                                         False, False, 0)
        assert high_resolve < low_resolve

    def test_paranoia_amplifies_stress(self):
        low_paranoia = accumulate_stress(0.3, 0.5, 0.1, 0.4, [0.8], 0, 0.0, 0, 0,
                                          False, False, 0)
        high_paranoia = accumulate_stress(0.3, 0.5, 0.9, 0.4, [0.8], 0, 0.0, 0, 0,
                                           False, False, 0)
        assert high_paranoia > low_paranoia

    def test_traditions_comfort(self):
        no_trad = accumulate_stress(0.3, 0.5, 0.3, 0.4, [0.5], 0, 0.0, 0, 0,
                                     False, False, 0)
        many_trad = accumulate_stress(0.3, 0.5, 0.3, 0.4, [0.5], 0, 0.0, 0, 0,
                                       False, False, 10)
        assert many_trad < no_trad

    def test_clamped_0_1(self):
        low = accumulate_stress(0.0, 0.9, 0.0, 0.0, [], 0, 0.0, 0, 0, False, False, 20)
        assert low >= 0.0
        high = accumulate_stress(0.99, 0.0, 0.9, 0.0, [1.0, 1.0, 1.0], 5, 0.8,
                                  3, 3, True, True, 0)
        assert high <= 1.0

    def test_meta_awareness_stress(self):
        without = accumulate_stress(0.3, 0.5, 0.3, 0.4, [], 0, 0.0, 0, 0, False, False, 0)
        with_meta = accumulate_stress(0.3, 0.5, 0.3, 0.4, [], 0, 0.0, 0, 0, True, False, 0)
        assert with_meta > without


# -- unit tests: update_morale -----------------------------------------------

class TestUpdateMorale:
    def test_high_stress_lowers_morale(self):
        result = update_morale(0.7, 0.9, 0.5)
        assert result < 0.7

    def test_low_stress_raises_morale(self):
        result = update_morale(0.3, 0.0, 0.8)
        assert result > 0.3

    def test_clamped(self):
        low = update_morale(0.0, 1.0, 0.0)
        assert low >= 0.0
        high = update_morale(1.0, 0.0, 1.0)
        assert high <= 1.0


# -- unit tests: check_breakdown ---------------------------------------------

class TestCheckBreakdown:
    def test_no_breakdown_below_threshold(self):
        rng = random.Random(42)
        for _ in range(100):
            assert not check_breakdown(BREAKDOWN_STRESS_THRESHOLD - 0.01, rng)

    def test_breakdown_at_max_stress(self):
        """At max stress, breakdown should be very likely."""
        rng = random.Random(42)
        results = [check_breakdown(1.0, rng) for _ in range(100)]
        assert sum(results) > 50  # > 50% of the time

    def test_deterministic(self):
        a = [check_breakdown(0.9, random.Random(99)) for _ in range(20)]
        b = [check_breakdown(0.9, random.Random(99)) for _ in range(20)]
        assert a == b


# -- unit tests: process_grief -----------------------------------------------

class TestProcessGrief:
    def test_removes_dead(self):
        bonds = ["col-1", "col-2", "col-3"]
        cleaned, deaths, exiles = process_grief(bonds, {"col-2"}, set())
        assert "col-2" not in cleaned
        assert deaths == 1
        assert exiles == 0

    def test_removes_exiled(self):
        bonds = ["col-1", "col-2"]
        cleaned, deaths, exiles = process_grief(bonds, set(), {"col-1"})
        assert "col-1" not in cleaned
        assert deaths == 0
        assert exiles == 1

    def test_no_departures(self):
        bonds = ["col-1", "col-2"]
        cleaned, deaths, exiles = process_grief(bonds, set(), set())
        assert cleaned == bonds
        assert deaths == 0
        assert exiles == 0


# -- unit tests: form_bonds --------------------------------------------------

class TestFormBonds:
    def test_bonds_form_with_high_trust(self):
        colonists = _make_colonists(2)
        social = FakeSocialGraph(
            default_trust=BOND_TRUST_THRESHOLD + 0.1,
            default_affection=BOND_AFFECTION_THRESHOLD + 0.1,
        )
        rng = random.Random(42)
        # Run multiple times to hit probability
        formed = []
        for seed in range(100):
            for c in colonists:
                c.bonds = []
            formed_this = form_bonds(colonists, social, random.Random(seed))
            formed.extend(formed_this)
        assert len(formed) > 0, "Should form at least one bond over 100 attempts"

    def test_no_bonds_with_low_trust(self):
        colonists = _make_colonists(2)
        social = FakeSocialGraph(default_trust=0.1, default_affection=0.1)
        rng = random.Random(42)
        formed = form_bonds(colonists, social, rng)
        assert len(formed) == 0

    def test_max_bonds_respected(self):
        colonists = _make_colonists(2)
        colonists[0].bonds = [f"x-{i}" for i in range(MAX_BONDS)]
        social = FakeSocialGraph(
            default_trust=BOND_TRUST_THRESHOLD + 0.2,
            default_affection=BOND_AFFECTION_THRESHOLD + 0.2,
        )
        formed = form_bonds(colonists, social, random.Random(42))
        assert len(formed) == 0

    def test_dead_colonists_excluded(self):
        colonists = _make_colonists(2)
        colonists[1].death_year = 1  # dead
        social = FakeSocialGraph(
            default_trust=BOND_TRUST_THRESHOLD + 0.2,
            default_affection=BOND_AFFECTION_THRESHOLD + 0.2,
        )
        formed = form_bonds(colonists, social, random.Random(42))
        assert len(formed) == 0


# -- unit tests: compute_psych_pressure --------------------------------------

class TestComputePsychPressure:
    def test_no_pressure_at_baseline(self):
        result = compute_psych_pressure(0.3, 0.7)
        assert len(result) == 0

    def test_low_morale_boosts_mediate(self):
        result = compute_psych_pressure(0.3, 0.2)
        assert "mediate" in result
        assert result["mediate"] > 0

    def test_high_stress_boosts_rest(self):
        result = compute_psych_pressure(0.8, 0.5)
        assert "rest" in result
        assert result["rest"] > 0


# -- integration tests: tick_psychology --------------------------------------

class TestTickPsychology:
    def test_basic_tick(self):
        colonists = _make_colonists(3)
        social = FakeSocialGraph()
        resources = FakeResources()
        events = [FakeEvent(severity=0.5)]
        culture = FakeCulture(traditions=["tradition1"])
        rng = random.Random(42)
        result = tick_psychology(colonists, social, resources, events, 1, culture, set(), rng)
        assert "avg_stress" in result
        assert "avg_morale" in result
        assert "breakdowns" in result

    def test_grief_on_departed(self):
        colonists = _make_colonists(3)
        colonists[0].bonds = ["col-1"]  # col-0 bonded with col-1
        social = FakeSocialGraph()
        resources = FakeResources()
        events = [FakeEvent(severity=0.3)]
        culture = FakeCulture()
        rng = random.Random(42)
        result = tick_psychology(colonists, social, resources, events, 1, culture,
                                 {"col-1"}, rng)
        assert result["grief_events"] > 0
        assert "col-1" not in colonists[0].bonds

    def test_empty_colony(self):
        """Handles no active colonists gracefully."""
        social = FakeSocialGraph()
        resources = FakeResources()
        culture = FakeCulture()
        result = tick_psychology([], social, resources, [], 1, culture, set(), random.Random(0))
        assert result["avg_stress"] == 0.0
        assert result["avg_morale"] == 0.0

    def test_high_stress_triggers_breakdown(self):
        colonists = _make_colonists(3)
        for c in colonists:
            c.stress = 0.95  # very high
        social = FakeSocialGraph()
        resources = FakeResources(food=0.1, water=0.1, power=0.1)
        events = [FakeEvent(severity=0.9)]
        culture = FakeCulture()
        # Run multiple seeds to find breakdowns
        got_breakdown = False
        for seed in range(50):
            for c in colonists:
                c.stress = 0.95
                c.breakdown_year = None
            result = tick_psychology(colonists, social, resources, events, 5, culture,
                                     set(), random.Random(seed))
            if result["breakdowns"] > 0:
                got_breakdown = True
                break
        assert got_breakdown, "Expected at least one breakdown at high stress"

    def test_deterministic(self):
        def run(seed):
            colonists = _make_colonists(5)
            for i, c in enumerate(colonists):
                c.stress = 0.3 + i * 0.1
            social = FakeSocialGraph()
            resources = FakeResources()
            events = [FakeEvent(severity=0.6)]
            culture = FakeCulture(traditions=["a", "b"])
            return tick_psychology(colonists, social, resources, events, 10,
                                   culture, set(), random.Random(seed))

        a = run(42)
        b = run(42)
        assert a == b


# -- PsychTickResult tests ---------------------------------------------------

class TestPsychTickResult:
    def test_to_dict(self):
        r = PsychTickResult(
            avg_stress=0.456789,
            avg_morale=0.654321,
            breakdowns=[{"colonist_id": "c1", "name": "A", "year": 5}],
            bonds_formed=[{"a": "c1", "b": "c2"}],
            bonds_broken=[],
            grief_events=[],
        )
        d = r.to_dict()
        assert d["breakdowns"] == 1
        assert len(d["breakdown_details"]) == 1
        assert d["bonds_formed"] == 1
        assert d["avg_stress"] == 0.4568  # rounded


# -- property tests ----------------------------------------------------------

class TestPropertyInvariants:
    """Stress and morale stay bounded for any reasonable inputs."""

    @pytest.mark.parametrize("stress", [0.0, 0.3, 0.5, 0.7, 1.0])
    @pytest.mark.parametrize("resolve", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("paranoia", [0.0, 0.5, 1.0])
    def test_stress_bounded(self, stress, resolve, paranoia):
        result = accumulate_stress(
            stress, resolve, paranoia, 0.5,
            [0.5, 0.8], 2, 0.6, 1, 1, True, True, 5,
        )
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("morale", [0.0, 0.3, 0.5, 0.7, 1.0])
    @pytest.mark.parametrize("stress", [0.0, 0.5, 1.0])
    def test_morale_bounded(self, morale, stress):
        result = update_morale(morale, stress, 0.5)
        assert 0.0 <= result <= 1.0

    def test_gini_helper(self):
        from src.mars100.psychology import _simple_gini
        assert _simple_gini([]) == 0.0
        assert _simple_gini([1.0]) == 0.0
        assert 0.0 <= _simple_gini([1, 2, 3, 4, 5]) <= 1.0
        assert _simple_gini([1, 1, 1, 1]) == 0.0
        gini_unequal = _simple_gini([0, 0, 0, 100])
        assert gini_unequal > 0.5
