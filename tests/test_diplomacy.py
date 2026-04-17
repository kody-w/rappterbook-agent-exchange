"""Tests for the Mars-100 diplomacy engine."""
from __future__ import annotations

import random
import pytest

from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import Resources, RESOURCE_NAMES
from src.mars100.diplomacy import (
    Faction, Pact, DiplomacyState, DiplomacyYearResult,
    detect_factions, manage_pacts, compute_action_modifiers,
    tick_diplomacy, check_crisis, STAT_TO_FACTION,
    CRISIS_THRESHOLD, MIN_FACTION_SIZE, FORMATION_YEARS,
    ALLIANCE_COOP_BOOST, RIVALRY_SABOTAGE_BOOST, CRISIS_COOP_BOOST,
    _factions_compatible,
)
from src.mars100.engine import Mars100Engine


# --------------- Helpers --------------------------------------------------

def _make_colonist(cid: str, dominant: str, **stat_overrides) -> Colonist:
    """Create a colonist with a clear dominant stat."""
    stats_kwargs = {s: 0.3 for s in ("resolve", "improvisation", "empathy",
                                       "hoarding", "faith", "paranoia")}
    stats_kwargs[dominant] = 0.9
    stats_kwargs.update(stat_overrides)
    return Colonist(
        id=cid, name=f"Test-{cid}", element="fire", archetype="pioneer",
        stats=ColonistStats(**stats_kwargs),
        skills=ColonistSkills(),
        decision_expr="(+ resolve empathy)",
        birth_year=0,
    )


def _make_faction_colonists(dominant: str, count: int, prefix: str = "") -> list[Colonist]:
    """Create *count* colonists all sharing the same dominant stat."""
    return [_make_colonist(f"{prefix}c-{i}", dominant) for i in range(count)]


# --------------- Faction detection ----------------------------------------

class TestFactionDetection:
    def test_detects_faction_from_dominant_stat(self):
        colonists = _make_faction_colonists("resolve", 3)
        state = DiplomacyState()
        events = detect_factions(colonists, state, year=1)
        assert len(state.factions) == 1
        assert state.factions[0].name == "The Resolute"
        assert len(state.factions[0].member_ids) == 3

    def test_ignores_small_groups(self):
        colonists = [_make_colonist("c-0", "resolve")]
        state = DiplomacyState()
        detect_factions(colonists, state, year=1)
        assert len(state.factions) == 0

    def test_multiple_factions(self):
        colonists = (
            _make_faction_colonists("resolve", 3, prefix="r-") +
            _make_faction_colonists("empathy", 2, prefix="e-")
        )
        state = DiplomacyState()
        detect_factions(colonists, state, year=1)
        names = {f.name for f in state.factions}
        assert "The Resolute" in names
        assert "The Empaths" in names

    def test_faction_solidifies_after_formation_years(self):
        colonists = _make_faction_colonists("faith", 3)
        state = DiplomacyState()
        for yr in range(1, FORMATION_YEARS + 2):
            detect_factions(colonists, state, year=yr)
        assert state.factions[0].solidified is True

    def test_faction_not_solidified_before_formation(self):
        colonists = _make_faction_colonists("faith", 3)
        state = DiplomacyState()
        detect_factions(colonists, state, year=1)
        assert state.factions[0].solidified is False

    def test_faction_pruned_when_below_min(self):
        colonists = _make_faction_colonists("resolve", 3)
        state = DiplomacyState()
        detect_factions(colonists, state, year=1)
        assert len(state.factions) == 1
        detect_factions([colonists[0]], state, year=2)
        assert len(state.factions) == 0

    def test_inactive_colonists_excluded(self):
        colonists = _make_faction_colonists("resolve", 3)
        colonists[0].die(1, "test")
        colonists[1].die(1, "test")
        state = DiplomacyState()
        detect_factions(colonists, state, year=2)
        assert len(state.factions) == 0

    def test_faction_reformed_on_membership_shift(self):
        group_a = _make_faction_colonists("resolve", 3, prefix="a-")
        state = DiplomacyState()
        detect_factions(group_a, state, year=1)
        assert state.factions[0].stable_years == 1
        group_b = _make_faction_colonists("resolve", 3, prefix="b-")
        events = detect_factions(group_b, state, year=2)
        reformed = [e for e in events if e["type"] == "faction_reformed"]
        assert len(reformed) == 1
        assert state.factions[0].stable_years == 0


# --------------- Crisis detection -----------------------------------------

class TestCrisis:
    def test_no_crisis_with_healthy_resources(self):
        r = Resources(food=0.5, water=0.5, power=0.5, air=0.5, medicine=0.5)
        assert check_crisis(r) == []

    def test_crisis_detected(self):
        r = Resources(food=0.1, water=0.5, power=0.0, air=0.5, medicine=0.5)
        critical = check_crisis(r)
        assert "food" in critical
        assert "power" in critical
        assert "water" not in critical

    def test_threshold_boundary(self):
        r = Resources(food=CRISIS_THRESHOLD, water=CRISIS_THRESHOLD - 0.01,
                      power=0.5, air=0.5, medicine=0.5)
        critical = check_crisis(r)
        assert "food" not in critical
        assert "water" in critical


# --------------- Pact management ------------------------------------------

class TestPacts:
    def test_alliance_formed_between_compatible_factions(self):
        state = DiplomacyState(factions=[
            Faction("The Resolute", "resolve", ["c-0", "c-1"],
                    formed_year=1, stable_years=3, solidified=True),
            Faction("The Faithful", "faith", ["c-2", "c-3"],
                    formed_year=1, stable_years=3, solidified=True),
        ])
        rng = random.Random(42)
        formed, expired = manage_pacts(state, year=5, rng=rng)
        assert len(formed) >= 1
        assert formed[0]["kind"] == "alliance"

    def test_pact_expires(self):
        pact = Pact("A", "B", "alliance", start_year=1, duration=1, reason="test")
        state = DiplomacyState(pacts=[pact])
        _, expired = manage_pacts(state, year=2, rng=random.Random(1))
        assert len(expired) == 1
        assert len(state.pacts) == 0

    def test_no_duplicate_pacts(self):
        fa = Faction("The Resolute", "resolve", ["c-0", "c-1"],
                     formed_year=1, stable_years=3, solidified=True)
        fb = Faction("The Faithful", "faith", ["c-2", "c-3"],
                     formed_year=1, stable_years=3, solidified=True)
        state = DiplomacyState(factions=[fa, fb])
        rng = random.Random(42)
        manage_pacts(state, year=5, rng=rng)
        n_pacts = len(state.pacts)
        manage_pacts(state, year=6, rng=rng)
        assert len(state.pacts) == n_pacts


class TestFactionCompatibility:
    def test_compatible_pairs(self):
        fa = Faction("A", "resolve", [])
        fb = Faction("B", "faith", [])
        assert _factions_compatible(fa, fb) is True

    def test_incompatible_pairs(self):
        fa = Faction("A", "resolve", [])
        fb = Faction("B", "paranoia", [])
        assert _factions_compatible(fa, fb) is False


# --------------- Action modifiers -----------------------------------------

class TestActionModifiers:
    def test_crisis_boosts_cooperation(self):
        colonists = _make_faction_colonists("resolve", 3)
        resources = Resources(food=0.05, water=0.5, power=0.5, air=0.5, medicine=0.5)
        state = DiplomacyState()
        detect_factions(colonists, state, year=1)
        mods = compute_action_modifiers(state, colonists, resources)
        for cid in [c.id for c in colonists]:
            assert mods[cid]["cooperate"] == CRISIS_COOP_BOOST
            assert mods[cid]["sabotage"] < 1.0

    def test_crisis_boosts_relevant_actions(self):
        colonists = _make_faction_colonists("resolve", 2)
        resources = Resources(food=0.05, water=0.5, power=0.01, air=0.5, medicine=0.5)
        state = DiplomacyState()
        mods = compute_action_modifiers(state, colonists, resources)
        cid = colonists[0].id
        assert mods[cid].get("farm", 1.0) > 1.0
        assert mods[cid].get("code", 1.0) > 1.0

    def test_alliance_boosts_cooperation(self):
        fa = Faction("The Resolute", "resolve", ["c-0"],
                     formed_year=1, stable_years=3, solidified=True)
        fb = Faction("The Faithful", "faith", ["c-1"],
                     formed_year=1, stable_years=3, solidified=True)
        pact = Pact("The Resolute", "The Faithful", "alliance",
                    start_year=1, duration=5, reason="test")
        state = DiplomacyState(factions=[fa, fb], pacts=[pact])
        colonists = [_make_colonist("c-0", "resolve"), _make_colonist("c-1", "faith")]
        resources = Resources()
        mods = compute_action_modifiers(state, colonists, resources)
        assert mods["c-0"]["cooperate"] > 1.0

    def test_embargo_boosts_sabotage(self):
        fa = Faction("The Resolute", "resolve", ["c-0"],
                     formed_year=1, stable_years=3, solidified=True)
        fb = Faction("The Hoarders", "hoarding", ["c-1"],
                     formed_year=1, stable_years=3, solidified=True)
        pact = Pact("The Resolute", "The Hoarders", "embargo",
                    start_year=1, duration=5, reason="test")
        state = DiplomacyState(factions=[fa, fb], pacts=[pact])
        colonists = [_make_colonist("c-0", "resolve"), _make_colonist("c-1", "hoarding")]
        resources = Resources()
        mods = compute_action_modifiers(state, colonists, resources)
        assert mods["c-0"]["sabotage"] > 1.0

    def test_unaffiliated_colonist_gets_no_modifiers(self):
        colonists = [_make_colonist("loner", "resolve")]
        resources = Resources()
        state = DiplomacyState()
        mods = compute_action_modifiers(state, colonists, resources)
        assert mods["loner"] == {}

    def test_faction_identity_bonus(self):
        fa = Faction("The Empaths", "empathy", ["c-0"],
                     formed_year=1, stable_years=3, solidified=True)
        state = DiplomacyState(factions=[fa])
        colonists = [_make_colonist("c-0", "empathy")]
        resources = Resources()
        mods = compute_action_modifiers(state, colonists, resources)
        assert mods["c-0"].get("mediate", 1.0) > 1.0


# --------------- tick_diplomacy integration --------------------------------

class TestTickDiplomacy:
    def test_returns_year_result(self):
        colonists = _make_faction_colonists("resolve", 3)
        resources = Resources()
        state = DiplomacyState()
        result = tick_diplomacy(state, colonists, resources, year=1, rng=random.Random(42))
        assert isinstance(result, DiplomacyYearResult)
        assert isinstance(result.action_modifiers, dict)

    def test_crisis_tracking(self):
        colonists = _make_faction_colonists("resolve", 3)
        resources = Resources(food=0.01, water=0.5, power=0.5, air=0.5, medicine=0.5)
        state = DiplomacyState()
        result = tick_diplomacy(state, colonists, resources, year=1, rng=random.Random(42))
        assert result.crisis_active is True
        assert state.crisis_active is True
        assert state.crisis_years == 1
        resources.food = 0.5
        result2 = tick_diplomacy(state, colonists, resources, year=2, rng=random.Random(42))
        assert result2.crisis_active is False
        assert state.crisis_years == 0

    def test_crisis_history_logged(self):
        colonists = _make_faction_colonists("resolve", 2)
        resources = Resources(power=0.0, food=0.5, water=0.5, air=0.5, medicine=0.5)
        state = DiplomacyState()
        tick_diplomacy(state, colonists, resources, year=1, rng=random.Random(1))
        assert any(h["type"] == "crisis_started" for h in state.history)
        resources.power = 0.5
        tick_diplomacy(state, colonists, resources, year=2, rng=random.Random(1))
        assert any(h["type"] == "crisis_ended" for h in state.history)


# --------------- Serialization --------------------------------------------

class TestSerialization:
    def test_faction_roundtrip(self):
        f = Faction("Test", "resolve", ["a", "b"], 1, 3, True)
        assert Faction.from_dict(f.to_dict()).name == "Test"

    def test_pact_roundtrip(self):
        p = Pact("A", "B", "alliance", 1, 5, "reason")
        p2 = Pact.from_dict(p.to_dict())
        assert p2.kind == "alliance"
        assert p2.duration == 5

    def test_state_roundtrip(self):
        state = DiplomacyState(
            factions=[Faction("Test", "faith", ["x"], 1, 2, True)],
            pacts=[Pact("A", "B", "embargo", 1, 3, "test")],
            crisis_active=True, crisis_years=2,
        )
        d = state.to_dict()
        s2 = DiplomacyState.from_dict(d)
        assert len(s2.factions) == 1
        assert s2.crisis_active is True

    def test_year_result_to_dict(self):
        result = DiplomacyYearResult(
            factions_detected=[{"type": "detected"}],
            pacts_formed=[{"kind": "alliance"}],
            pacts_expired=[],
            crisis_active=False,
            action_modifiers={"c-0": {"cooperate": 1.5}},
        )
        d = result.to_dict()
        assert d["modifier_count"] == 1

    def test_state_summary(self):
        state = DiplomacyState(
            factions=[Faction("T", "resolve", ["x"], 1, 3, True)],
        )
        s = state.summary(5)
        assert s["year"] == 5
        assert s["num_solidified"] == 1

    def test_faction_of(self):
        state = DiplomacyState(
            factions=[Faction("T", "resolve", ["a", "b"], 1, 2, True)]
        )
        assert state.faction_of("a") is not None
        assert state.faction_of("z") is None

    def test_pact_between(self):
        state = DiplomacyState(
            pacts=[Pact("A", "B", "alliance", 1, 5, "test")]
        )
        assert state.pact_between("A", "B") is not None
        assert state.pact_between("B", "A") is not None
        assert state.pact_between("A", "C") is None


# --------------- Engine integration smoke tests ----------------------------

class TestEngineIntegration:
    def test_single_tick_with_diplomacy(self):
        engine = Mars100Engine(seed=42, total_years=1)
        result = engine.tick()
        assert "diplomacy" in result.to_dict()
        assert "num_factions" in result.diplomacy

    def test_10_year_run_with_diplomacy(self):
        engine = Mars100Engine(seed=42, total_years=10)
        sim = engine.run()
        d = sim.to_dict()
        assert "diplomacy" in d
        assert len(d["years"]) == 10
        for yr in d["years"]:
            assert "diplomacy" in yr

    def test_25_year_run_factions_emerge(self):
        engine = Mars100Engine(seed=42, total_years=25)
        sim = engine.run()
        any_factions = any(
            yr.diplomacy.get("num_factions", 0) > 0
            for yr in sim.years
        )
        assert any_factions, "No factions emerged in 25 years"

    def test_determinism_preserved(self):
        """Diplomacy must not break determinism."""
        r1 = Mars100Engine(seed=123, total_years=10).run()
        r2 = Mars100Engine(seed=123, total_years=10).run()
        for y1, y2 in zip(r1.years, r2.years):
            assert y1.actions == y2.actions
            assert y1.diplomacy == y2.diplomacy

    def test_crisis_affects_actions(self):
        """When a resource is critical, diplomacy should boost cooperation."""
        engine = Mars100Engine(seed=42, total_years=1)
        engine.resources.power = 0.0
        engine.resources.air = 0.05
        result = engine.tick()
        assert result.diplomacy.get("crisis_active", False) is True

    def test_sim_result_has_diplomacy(self):
        engine = Mars100Engine(seed=42, total_years=5)
        sim = engine.run()
        d = sim.to_dict()
        assert "diplomacy" in d
        assert "factions" in d["diplomacy"]


# --------------- Property-based invariants --------------------------------

class TestInvariants:
    def test_modifiers_always_positive(self):
        """All action modifiers must be > 0."""
        for seed in range(5):
            engine = Mars100Engine(seed=seed, total_years=15)
            for _ in range(15):
                if not engine._active_colonists():
                    break
                diplo = tick_diplomacy(
                    engine.diplomacy_state, engine._active_colonists(),
                    engine.resources, engine.year + 1, engine.rng,
                )
                for cid, mods in diplo.action_modifiers.items():
                    for action, val in mods.items():
                        assert val > 0, f"Modifier for {cid}/{action} was {val}"
                engine.tick()

    def test_faction_names_always_from_mapping(self):
        """All faction names must come from STAT_TO_FACTION."""
        valid_names = set(STAT_TO_FACTION.values())
        engine = Mars100Engine(seed=42, total_years=30)
        engine.run()
        for f in engine.diplomacy_state.factions:
            assert f.name in valid_names, f"Unknown faction: {f.name}"

    def test_resources_bounded_with_diplomacy(self):
        """Resources stay in [0, 1] even with diplomacy modifiers."""
        engine = Mars100Engine(seed=42, total_years=50)
        sim = engine.run()
        for yr in sim.years:
            for res, val in yr.resources_after.items():
                assert 0.0 <= val <= 1.0, f"Year {yr.year} {res}={val}"

    def test_stat_to_faction_covers_all_stats(self):
        """Every stat in STAT_NAMES maps to a faction name."""
        from src.mars100.colonist import STAT_NAMES as sn
        for stat in sn:
            assert stat in STAT_TO_FACTION, f"No faction for stat: {stat}"
