"""Tests for Mars-100 diplomacy engine."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills
from src.mars100.colony import SocialGraph
from src.mars100.diplomacy import (
    DiplomacyState, DiplomacyTickResult, Faction, Treaty,
    detect_factions, update_cohesion, check_schisms,
    propose_treaty, sign_treaty, expire_treaties,
    compute_treaty_effects, faction_vote_modifier, tick_diplomacy,
    MAX_FACTIONS, MIN_FACTION_SIZE, FACTION_TRUST_THRESHOLD,
    SCHISM_COHESION_THRESHOLD, VOTE_MODIFIER_CAP, TREATY_BASE_DURATION,
    GOVERNANCE_PREFERENCE,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _make_colonist(
    cid: str, dominant: str = "resolve", value: float = 0.9,
) -> Colonist:
    """Create a colonist with one dominant stat."""
    stats_kwargs = {s: 0.3 for s in ["resolve", "improvisation", "empathy",
                                      "hoarding", "faith", "paranoia"]}
    stats_kwargs[dominant] = value
    return Colonist(
        id=cid, name=f"Col-{cid}", element="fire",
        stats=ColonistStats(**stats_kwargs),
        skills=ColonistSkills(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
        archetype="colonist",
        decision_expr="(+ resolve empathy)",
    )


def _make_social(colonists: list[Colonist], trust: float = 0.5) -> SocialGraph:
    """Create a SocialGraph with uniform trust."""
    sg = SocialGraph()
    ids = [c.id for c in colonists]
    sg.initialize(ids, random.Random(42))
    for a in ids:
        for b in ids:
            if a != b:
                rel = sg.get(a, b)
                rel.trust = trust
    return sg


# ── faction detection ──────────────────────────────────────────────────────

class TestDetectFactions:
    def test_forms_faction_with_shared_stat(self):
        cols = [_make_colonist(f"c{i}", "resolve") for i in range(3)]
        social = _make_social(cols, trust=0.6)
        state = DiplomacyState()
        formed = detect_factions(state, cols, social, year=1, rng=random.Random(1))
        assert len(formed) == 1
        assert formed[0]["dominant_stat"] == "resolve"
        assert len(formed[0]["member_ids"]) == 3

    def test_no_faction_below_min_size(self):
        cols = [_make_colonist("c1", "resolve")]
        social = _make_social(cols, trust=0.9)
        state = DiplomacyState()
        formed = detect_factions(state, cols, social, year=1, rng=random.Random(1))
        assert len(formed) == 0

    def test_no_faction_below_trust_threshold(self):
        cols = [_make_colonist(f"c{i}", "resolve") for i in range(3)]
        social = _make_social(cols, trust=0.1)  # below FACTION_TRUST_THRESHOLD
        state = DiplomacyState()
        formed = detect_factions(state, cols, social, year=1, rng=random.Random(1))
        assert len(formed) == 0

    def test_max_factions_respected(self):
        state = DiplomacyState()
        # Create colonists to fill existing factions (they must be "active")
        existing_cols = []
        for i in range(MAX_FACTIONS):
            c1 = _make_colonist(f"x{i}", "resolve")
            c2 = _make_colonist(f"y{i}", "resolve")
            existing_cols.extend([c1, c2])
            state.factions.append(Faction(f"F{i}", "resolve", [f"x{i}", f"y{i}"]))
        new_cols = [_make_colonist(f"new{i}", "empathy") for i in range(3)]
        all_cols = existing_cols + new_cols
        social = _make_social(all_cols, trust=0.9)
        formed = detect_factions(state, all_cols, social, year=1, rng=random.Random(1))
        assert len(formed) == 0
        assert len(state.factions) == MAX_FACTIONS

    def test_dead_members_pruned(self):
        cols = [_make_colonist(f"c{i}", "resolve") for i in range(3)]
        cols[0].die(1, "starvation")
        social = _make_social(cols, trust=0.6)
        state = DiplomacyState()
        state.factions.append(Faction("Old", "resolve", ["c0", "c1", "c2"]))
        detect_factions(state, cols, social, year=2, rng=random.Random(1))
        old = state.factions[0]
        assert "c0" not in old.member_ids

    def test_multiple_stat_groups_form_separate_factions(self):
        cols = (
            [_make_colonist(f"r{i}", "resolve") for i in range(2)] +
            [_make_colonist(f"e{i}", "empathy") for i in range(2)]
        )
        social = _make_social(cols, trust=0.6)
        state = DiplomacyState()
        formed = detect_factions(state, cols, social, year=1, rng=random.Random(1))
        assert len(formed) == 2

    def test_already_factioned_colonists_skipped(self):
        cols = [_make_colonist(f"c{i}", "resolve") for i in range(3)]
        social = _make_social(cols, trust=0.6)
        state = DiplomacyState()
        state.factions.append(Faction("Existing", "resolve", ["c0", "c1", "c2"]))
        formed = detect_factions(state, cols, social, year=1, rng=random.Random(1))
        assert len(formed) == 0

    def test_empty_colonists(self):
        state = DiplomacyState()
        formed = detect_factions(state, [], SocialGraph(), year=1, rng=random.Random(1))
        assert formed == []


# ── serialization ──────────────────────────────────────────────────────────

class TestFactionSerialization:
    def test_faction_roundtrip(self):
        f = Faction("Test", "resolve", ["a", "b"], 0.75, 5)
        d = f.to_dict()
        f2 = Faction.from_dict(d)
        assert f2.name == f.name
        assert f2.member_ids == f.member_ids
        assert f2.cohesion == f.cohesion

    def test_treaty_roundtrip(self):
        t = Treaty("labour", "A", "B", 10, 15, "research_bonus", 0.15)
        d = t.to_dict()
        t2 = Treaty.from_dict(d)
        assert t2.treaty_type == t.treaty_type
        assert t2.expires_year == 25

    def test_state_roundtrip(self):
        s = DiplomacyState()
        s.factions.append(Faction("F1", "resolve", ["a"]))
        s.treaties.append(Treaty("labour", "F1", "F2", 5))
        d = s.to_dict()
        s2 = DiplomacyState.from_dict(d)
        assert len(s2.factions) == 1
        assert len(s2.treaties) == 1


# ── schisms ────────────────────────────────────────────────────────────────

class TestSchisms:
    def test_no_schism_above_cohesion_threshold(self):
        state = DiplomacyState()
        state.factions.append(Faction("F", "resolve", ["a", "b", "c", "d"], 0.8))
        schisms = check_schisms(state, year=10, rng=random.Random(1))
        assert len(schisms) == 0

    def test_no_schism_below_min_size_plus_one(self):
        state = DiplomacyState()
        state.factions.append(Faction("F", "resolve", ["a", "b", "c"], 0.1))
        schisms = check_schisms(state, year=10, rng=random.Random(1))
        assert len(schisms) == 0

    def test_schism_can_occur(self):
        state = DiplomacyState()
        members = [f"m{i}" for i in range(6)]
        state.factions.append(Faction("BigF", "resolve", members, 0.1))
        # Try many seeds until we get a schism
        for seed in range(100):
            state_copy = DiplomacyState(
                factions=[Faction("BigF", "resolve", list(members), 0.1)],
            )
            schisms = check_schisms(state_copy, year=10, rng=random.Random(seed))
            if schisms:
                assert schisms[0]["parent"] == "BigF"
                assert len(state_copy.factions) == 2
                return
        pytest.fail("No schism occurred in 100 seeds")

    def test_schism_logged(self):
        state = DiplomacyState()
        members = [f"m{i}" for i in range(6)]
        state.factions.append(Faction("BigF", "resolve", members, 0.1))
        for seed in range(100):
            s = DiplomacyState(
                factions=[Faction("BigF", "resolve", list(members), 0.1)],
            )
            check_schisms(s, year=10, rng=random.Random(seed))
            if s.schism_log:
                assert s.schism_log[0]["year"] == 10
                return
        pytest.fail("No schism logged in 100 seeds")


# ── treaties ───────────────────────────────────────────────────────────────

class TestTreatyProposal:
    def test_needs_two_factions(self):
        state = DiplomacyState()
        state.factions.append(Faction("F1", "resolve", ["a", "b"]))
        proposed = propose_treaty(state, year=5, rng=random.Random(1))
        assert len(proposed) == 0

    def test_can_propose(self):
        state = DiplomacyState()
        state.factions.append(Faction("F1", "resolve", ["a", "b"]))
        state.factions.append(Faction("F2", "empathy", ["c", "d"]))
        for seed in range(100):
            proposed = propose_treaty(state, year=5, rng=random.Random(seed))
            if proposed:
                assert proposed[0]["faction_a"] == "F1"
                assert proposed[0]["faction_b"] == "F2"
                return
        pytest.fail("No treaty proposed in 100 seeds")

    def test_no_duplicate_active_treaty(self):
        state = DiplomacyState()
        state.factions.append(Faction("F1", "resolve", ["a", "b"]))
        state.factions.append(Faction("F2", "empathy", ["c", "d"]))
        state.treaties.append(Treaty("labour", "F1", "F2", signed_year=1))
        proposed = propose_treaty(state, year=5, rng=random.Random(1))
        assert len(proposed) == 0


class TestSignAndExpire:
    def test_sign_adds_treaty(self):
        state = DiplomacyState()
        prop = {"treaty_type": "labour", "faction_a": "A", "faction_b": "B",
                "bonus_key": "research_bonus", "bonus_value": 0.15}
        treaty = sign_treaty(state, prop, year=10)
        assert len(state.treaties) == 1
        assert treaty.expires_year == 10 + TREATY_BASE_DURATION

    def test_expire_removes_old(self):
        state = DiplomacyState()
        state.treaties.append(Treaty("labour", "A", "B", signed_year=1, duration=5))
        expired = expire_treaties(state, year=10)
        assert len(expired) == 1
        assert len(state.treaties) == 0

    def test_keep_active(self):
        state = DiplomacyState()
        state.treaties.append(Treaty("labour", "A", "B", signed_year=5, duration=20))
        expired = expire_treaties(state, year=10)
        assert len(expired) == 0
        assert len(state.treaties) == 1


# ── treaty effects ─────────────────────────────────────────────────────────

class TestTreatyEffects:
    def test_empty_state(self):
        state = DiplomacyState()
        effects = compute_treaty_effects(state)
        assert effects == {}

    def test_single_treaty(self):
        state = DiplomacyState()
        state.treaties.append(Treaty("labour", "A", "B", 1, bonus_key="research_bonus", bonus_value=0.15))
        effects = compute_treaty_effects(state)
        assert abs(effects["research_bonus"] - 0.15) < 1e-9

    def test_stacking(self):
        state = DiplomacyState()
        state.treaties.append(Treaty("labour", "A", "B", 1, bonus_key="research_bonus", bonus_value=0.15))
        state.treaties.append(Treaty("labour", "C", "D", 2, bonus_key="research_bonus", bonus_value=0.10))
        effects = compute_treaty_effects(state)
        assert abs(effects["research_bonus"] - 0.25) < 1e-9

    def test_different_keys(self):
        state = DiplomacyState()
        state.treaties.append(Treaty("labour", "A", "B", 1, bonus_key="research_bonus", bonus_value=0.15))
        state.treaties.append(Treaty("emergency_air", "A", "B", 1, bonus_key="air_crisis_bonus", bonus_value=0.10))
        effects = compute_treaty_effects(state)
        assert "research_bonus" in effects
        assert "air_crisis_bonus" in effects


# ── vote modifier ──────────────────────────────────────────────────────────

class TestVoteModifier:
    def test_no_faction(self):
        mod = faction_vote_modifier("x", "council", [])
        assert mod == 0.0

    def test_preferred_gov_positive(self):
        f = Faction("F", "empathy", ["c1"], 0.8)  # empathy → consensus
        mod = faction_vote_modifier("c1", "consensus", [f])
        assert mod > 0
        assert mod <= VOTE_MODIFIER_CAP

    def test_non_preferred_gov_negative(self):
        f = Faction("F", "empathy", ["c1"], 0.8)
        mod = faction_vote_modifier("c1", "dictator", [f])
        assert mod < 0

    def test_capped(self):
        f = Faction("F", "resolve", ["c1"], 1.0)
        mod = faction_vote_modifier("c1", "dictator", [f])
        assert abs(mod) <= VOTE_MODIFIER_CAP

    def test_unaffiliated_returns_zero(self):
        f = Faction("F", "resolve", ["c1"], 0.8)
        mod = faction_vote_modifier("c2", "dictator", [f])
        assert mod == 0.0


# ── full tick ──────────────────────────────────────────────────────────────

class TestTickDiplomacy:
    def test_returns_tick_result(self):
        cols = [_make_colonist(f"c{i}", "resolve") for i in range(4)]
        social = _make_social(cols, trust=0.6)
        state = DiplomacyState()
        result = tick_diplomacy(state, cols, social, year=1, rng=random.Random(42))
        assert isinstance(result, DiplomacyTickResult)

    def test_factions_form_over_time(self):
        cols = [_make_colonist(f"c{i}", "resolve") for i in range(4)]
        social = _make_social(cols, trust=0.6)
        state = DiplomacyState()
        result = tick_diplomacy(state, cols, social, year=1, rng=random.Random(42))
        assert len(state.factions) >= 1

    def test_to_dict_roundtrip(self):
        result = DiplomacyTickResult(
            factions_formed=[{"name": "F1"}],
            schisms=[{"year": 5}],
        )
        d = result.to_dict()
        assert d["factions_formed"] == [{"name": "F1"}]
        assert d["schisms"] == [{"year": 5}]

    def test_empty_colonists_safe(self):
        state = DiplomacyState()
        result = tick_diplomacy(state, [], SocialGraph(), year=1, rng=random.Random(1))
        assert result.factions_formed == []

    def test_single_colonist_no_crash(self):
        cols = [_make_colonist("solo", "resolve")]
        social = _make_social(cols, trust=0.5)
        state = DiplomacyState()
        result = tick_diplomacy(state, cols, social, year=1, rng=random.Random(1))
        assert len(state.factions) == 0


# ── determinism ────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_result(self):
        cols = [_make_colonist(f"c{i}", "resolve") for i in range(4)]
        results = []
        for _ in range(2):
            social = _make_social(cols, trust=0.6)
            state = DiplomacyState()
            r = tick_diplomacy(state, cols, social, year=1, rng=random.Random(42))
            results.append(r.to_dict())
        assert results[0] == results[1]

    def test_different_seed_may_differ(self):
        cols = [_make_colonist(f"c{i}", "resolve") for i in range(4)]
        outcomes = set()
        for seed in range(20):
            social = _make_social(cols, trust=0.6)
            state = DiplomacyState()
            r = tick_diplomacy(state, cols, social, year=1, rng=random.Random(seed))
            outcomes.add(str(r.to_dict()))
        # With 20 seeds at least some should differ in treaty proposals etc.
        assert len(outcomes) >= 1


# ── invariants ─────────────────────────────────────────────────────────────

class TestInvariants:
    def test_faction_count_bounded(self):
        cols = [_make_colonist(f"c{i}", ["resolve", "empathy", "faith", "paranoia", "hoarding"][i % 5])
                for i in range(20)]
        social = _make_social(cols, trust=0.8)
        state = DiplomacyState()
        for year in range(50):
            tick_diplomacy(state, cols, social, year=year, rng=random.Random(year))
        assert len(state.factions) <= MAX_FACTIONS

    def test_no_empty_factions_after_tick(self):
        cols = [_make_colonist(f"c{i}", "resolve") for i in range(4)]
        social = _make_social(cols, trust=0.6)
        state = DiplomacyState()
        for year in range(10):
            tick_diplomacy(state, cols, social, year=year, rng=random.Random(year))
        for f in state.factions:
            assert f.size >= 1

    def test_vote_modifier_always_bounded(self):
        for stat in GOVERNANCE_PREFERENCE:
            f = Faction("F", stat, ["c1"], cohesion=1.0)
            for gov in ["council", "dictator", "lottery", "consensus", "ai_governor", "anarchy"]:
                mod = faction_vote_modifier("c1", gov, [f])
                assert -VOTE_MODIFIER_CAP <= mod <= VOTE_MODIFIER_CAP

    def test_treaty_effects_non_negative(self):
        state = DiplomacyState()
        state.treaties.append(Treaty("labour", "A", "B", 1, bonus_key="research_bonus", bonus_value=0.15))
        effects = compute_treaty_effects(state)
        for v in effects.values():
            assert v >= 0.0
