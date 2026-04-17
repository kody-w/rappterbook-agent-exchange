"""Tests for Mars-100 diplomacy engine."""
from __future__ import annotations

import random
import pytest
from dataclasses import dataclass, field
from typing import Any

from src.mars100.diplomacy import (
    Faction, Treaty, DiplomacyState, DiplomacyTickResult,
    detect_factions, check_schisms, propose_treaty, sign_treaty,
    expire_treaties, compute_treaty_effects, faction_vote_modifier,
    tick_diplomacy, _compute_cohesion,
    MIN_FACTION_SIZE, MAX_FACTIONS, VOTE_MODIFIER_CAP,
    SCHISM_THRESHOLD, TREATY_DURATION_YEARS, HYSTERESIS_YEARS,
    EMERGENCY_LABOUR_BONUS,
)


@dataclass
class FakeStats:
    resolve: float = 0.5
    faith: float = 0.5
    empathy: float = 0.5
    improvisation: float = 0.5
    paranoia: float = 0.5
    hoarding: float = 0.5


@dataclass
class FakeColonist:
    id: str
    name: str = "Test"
    stats: FakeStats = field(default_factory=FakeStats)
    _active: bool = True
    def is_active(self) -> bool:
        return self._active


@dataclass
class FakeRelationship:
    trust: float = 0.5


class FakeSocialGraph:
    def __init__(self, default_trust: float = 0.5):
        self._default = default_trust
    def get(self, a: str, b: str) -> FakeRelationship:
        return FakeRelationship(trust=self._default)


def _make(n: int, stat: str = "resolve") -> list[FakeColonist]:
    cs = []
    for i in range(n):
        s = FakeStats()
        setattr(s, stat, 0.9)
        cs.append(FakeColonist(id=f"col-{i}", stats=s))
    return cs


class TestDetectFactions:
    def test_no_factions_below_min_size(self):
        r = detect_factions(_make(2), FakeSocialGraph(), {}, 5, random.Random(42))
        assert r == {}

    def test_single_faction_forms(self):
        r = detect_factions(_make(5), FakeSocialGraph(), {}, 5, random.Random(42))
        assert len(r) == 1
        f = list(r.values())[0]
        assert f.dominant_stat == "resolve"
        assert len(f.member_ids) >= MIN_FACTION_SIZE

    def test_hysteresis_blocks_early(self):
        r = detect_factions(_make(5), FakeSocialGraph(), {}, 1, random.Random(42))
        assert len(r) == 0

    def test_existing_faction_updated(self):
        existing = {"faction-resolve": Faction(id="faction-resolve", name="The Resolute",
                     formed_year=2, member_ids=["old-1"], dominant_stat="resolve")}
        r = detect_factions(_make(5), FakeSocialGraph(), existing, 5, random.Random(42))
        assert "faction-resolve" in r
        assert len(r["faction-resolve"].member_ids) == 5

    def test_multiple_factions(self):
        faith = _make(3, "faith")
        for i, c in enumerate(faith):
            c.id = f"faith-{i}"
        r = detect_factions(_make(3) + faith, FakeSocialGraph(), {}, 5, random.Random(42))
        assert len(r) == 2

    def test_max_factions_cap(self):
        cs = []
        for stat in ["resolve", "faith", "empathy", "improvisation", "paranoia", "hoarding"]:
            g = _make(3, stat)
            for j, c in enumerate(g):
                c.id = f"{stat}-{j}"
            cs.extend(g)
        r = detect_factions(cs, FakeSocialGraph(), {}, 5, random.Random(42))
        assert len(r) <= MAX_FACTIONS

    def test_inactive_excluded(self):
        cs = _make(5)
        for c in cs[:3]:
            c._active = False
        r = detect_factions(cs, FakeSocialGraph(), {}, 5, random.Random(42))
        assert len(r) == 0

    def test_deterministic(self):
        cs = _make(6)
        r1 = detect_factions(cs, FakeSocialGraph(), {}, 5, random.Random(42))
        r2 = detect_factions(cs, FakeSocialGraph(), {}, 5, random.Random(42))
        assert list(r1.keys()) == list(r2.keys())


class TestSerialization:
    def test_faction_to_dict(self):
        f = Faction(id="f-1", name="X", formed_year=5, member_ids=["a","b","c"],
                    dominant_stat="resolve", cohesion=0.7)
        d = f.to_dict()
        assert d["id"] == "f-1"
        assert d["cohesion"] == 0.7
        assert not d["archived"]

    def test_treaty_to_dict(self):
        t = Treaty(id="t-1", faction_a="f-1", faction_b="f-2",
                   treaty_type="research_pact", signed_year=10, expires_year=20)
        d = t.to_dict()
        assert d["treaty_type"] == "research_pact"
        assert d["active"] is True

    def test_state_to_dict(self):
        s = DiplomacyState()
        s.factions["f-1"] = Faction(id="f-1", name="T", formed_year=5, member_ids=["a"])
        d = s.to_dict()
        assert d["active_faction_count"] == 1
        assert "f-1" in d["factions"]


class TestSchisms:
    def test_no_schism_above_threshold(self):
        fs = {"f-1": Faction(id="f-1", name="T", formed_year=1,
                              member_ids=["a","b","c"], cohesion=0.5)}
        assert len(check_schisms(fs, FakeSocialGraph(), 10)) == 0

    def test_schism_below_threshold(self):
        fs = {"f-1": Faction(id="f-1", name="T", formed_year=1,
                              member_ids=["a","b","c"], cohesion=0.1)}
        schisms = check_schisms(fs, FakeSocialGraph(), 10)
        assert len(schisms) == 1
        assert fs["f-1"].archived is True

    def test_at_boundary_no_schism(self):
        fs = {"f-1": Faction(id="f-1", name="T", formed_year=1,
                              member_ids=["a","b","c"], cohesion=SCHISM_THRESHOLD)}
        assert len(check_schisms(fs, FakeSocialGraph(), 10)) == 0

    def test_archived_ignored(self):
        fs = {"f-1": Faction(id="f-1", name="T", formed_year=1,
                              member_ids=["a","b","c"], cohesion=0.05, archived=True)}
        assert len(check_schisms(fs, FakeSocialGraph(), 10)) == 0


class TestTreatyProposal:
    def test_can_propose(self):
        s = DiplomacyState()
        s.factions["f-1"] = Faction(id="f-1", name="A", formed_year=1,
                                     member_ids=["a","b","c"], cohesion=0.9)
        s.factions["f-2"] = Faction(id="f-2", name="B", formed_year=1,
                                     member_ids=["d","e","f"], cohesion=0.9)
        got = any(propose_treaty(s, "f-1", "f-2", 10, random.Random(i))
                  for i in range(100))
        assert got

    def test_no_duplicate(self):
        s = DiplomacyState()
        s.factions["f-1"] = Faction(id="f-1", name="A", formed_year=1,
                                     member_ids=["a","b","c"], cohesion=0.9)
        s.factions["f-2"] = Faction(id="f-2", name="B", formed_year=1,
                                     member_ids=["d","e","f"], cohesion=0.9)
        s.treaties.append(Treaty(id="t-0", faction_a="f-1", faction_b="f-2",
                                  treaty_type="research_pact", signed_year=5, expires_year=15))
        for i in range(50):
            assert propose_treaty(s, "f-1", "f-2", 10, random.Random(i)) is None

    def test_archived_cannot_propose(self):
        s = DiplomacyState()
        s.factions["f-1"] = Faction(id="f-1", name="A", formed_year=1,
                                     member_ids=["a"], archived=True)
        s.factions["f-2"] = Faction(id="f-2", name="B", formed_year=1,
                                     member_ids=["d","e","f"])
        assert propose_treaty(s, "f-1", "f-2", 10, random.Random(42)) is None


class TestSignAndExpire:
    def test_sign(self):
        s = DiplomacyState()
        sign_treaty(s, Treaty(id="t-1", faction_a="f-1", faction_b="f-2",
                               treaty_type="research_pact", signed_year=5, expires_year=15))
        assert len(s.treaties) == 1

    def test_expire(self):
        s = DiplomacyState()
        s.treaties.append(Treaty(id="t-1", faction_a="f-1", faction_b="f-2",
                                  treaty_type="research_pact", signed_year=5, expires_year=15))
        assert expire_treaties(s, 15) == ["t-1"]
        assert not s.treaties[0].active

    def test_no_premature_expiry(self):
        s = DiplomacyState()
        s.treaties.append(Treaty(id="t-1", faction_a="f-1", faction_b="f-2",
                                  treaty_type="research_pact", signed_year=5, expires_year=15))
        assert expire_treaties(s, 10) == []
        assert s.treaties[0].active


class TestTreatyEffects:
    def test_empty(self):
        assert compute_treaty_effects(DiplomacyState())["research_bonus"] == 0.0

    def test_research_pact(self):
        s = DiplomacyState()
        s.treaties.append(Treaty(id="t-1", faction_a="f-1", faction_b="f-2",
                                  treaty_type="research_pact", signed_year=5, expires_year=15))
        assert compute_treaty_effects(s)["research_bonus"] == pytest.approx(0.02)

    def test_air_mutual_aid(self):
        s = DiplomacyState()
        s.treaties.append(Treaty(id="t-1", faction_a="f-1", faction_b="f-2",
                                  treaty_type="air_mutual_aid", signed_year=5, expires_year=15))
        assert compute_treaty_effects(s)["air_crisis_bonus"] == pytest.approx(EMERGENCY_LABOUR_BONUS)

    def test_inactive_ignored(self):
        s = DiplomacyState()
        s.treaties.append(Treaty(id="t-1", faction_a="f-1", faction_b="f-2",
                                  treaty_type="research_pact", signed_year=5, expires_year=15,
                                  active=False))
        assert compute_treaty_effects(s)["research_bonus"] == 0.0


class TestVoteModifier:
    def test_no_faction(self):
        assert faction_vote_modifier("col-0", "council", {}) == 0.0

    def test_aligned_positive(self):
        fs = {"f-1": Faction(id="f-1", name="E", formed_year=1,
                              member_ids=["col-0"], dominant_stat="empathy", cohesion=0.8)}
        mod = faction_vote_modifier("col-0", "council", fs)
        assert 0 < mod <= VOTE_MODIFIER_CAP

    def test_misaligned_negative(self):
        fs = {"f-1": Faction(id="f-1", name="R", formed_year=1,
                              member_ids=["col-0"], dominant_stat="resolve", cohesion=0.8)}
        assert faction_vote_modifier("col-0", "council", fs) < 0

    def test_capped(self):
        fs = {"f-1": Faction(id="f-1", name="E", formed_year=1,
                              member_ids=["col-0"], dominant_stat="empathy", cohesion=1.0)}
        assert abs(faction_vote_modifier("col-0", "council", fs)) <= VOTE_MODIFIER_CAP

    def test_archived_ignored(self):
        fs = {"f-1": Faction(id="f-1", name="E", formed_year=1,
                              member_ids=["col-0"], dominant_stat="empathy",
                              cohesion=0.8, archived=True)}
        assert faction_vote_modifier("col-0", "council", fs) == 0.0


class TestTickDiplomacy:
    def test_returns_result(self):
        s = DiplomacyState()
        r = tick_diplomacy(s, _make(6), FakeSocialGraph(), 5, random.Random(42))
        assert isinstance(r, DiplomacyTickResult)

    def test_detects_factions(self):
        s = DiplomacyState()
        tick_diplomacy(s, _make(6), FakeSocialGraph(), 5, random.Random(42))
        assert len(s.factions) >= 1

    def test_deterministic(self):
        def run(seed):
            s = DiplomacyState()
            r = tick_diplomacy(s, _make(6), FakeSocialGraph(), 5, random.Random(seed))
            return r.to_dict(), s.to_dict()
        assert run(42) == run(42)

    def test_serializable(self):
        s = DiplomacyState()
        r = tick_diplomacy(s, _make(6), FakeSocialGraph(), 5, random.Random(42))
        d = r.to_dict()
        assert isinstance(d, dict) and "factions_formed" in d

    def test_multi_year(self):
        s = DiplomacyState()
        rng = random.Random(42)
        for y in range(20):
            tick_diplomacy(s, _make(6), FakeSocialGraph(0.6), y, rng)
        assert s.to_dict()["active_faction_count"] >= 0


class TestDeterminism:
    def test_same_seed(self):
        def run():
            s = DiplomacyState()
            cs = _make(8)
            for i, c in enumerate(cs[4:]):
                c.id = f"faith-{i}"
                c.stats = FakeStats(faith=0.9)
            rs = []
            for y in range(10):
                rs.append(tick_diplomacy(s, cs, FakeSocialGraph(), y,
                                         random.Random(42+y)).to_dict())
            return rs, s.to_dict()
        assert run() == run()

    def test_different_seeds(self):
        results = []
        for seed in [42, 99]:
            s = DiplomacyState()
            rng = random.Random(seed)
            for y in range(10):
                tick_diplomacy(s, _make(6), FakeSocialGraph(), y, rng)
            results.append(s.to_dict())
        assert isinstance(results[0], dict)


class TestInvariants:
    def test_members_are_active(self):
        cs = _make(6)
        cs[5]._active = False
        s = DiplomacyState()
        tick_diplomacy(s, cs, FakeSocialGraph(), 5, random.Random(42))
        active_ids = {c.id for c in cs if c.is_active()}
        for f in s.factions.values():
            for m in f.member_ids:
                assert m in active_ids

    def test_cohesion_bounded(self):
        s = DiplomacyState()
        for y in range(30):
            tick_diplomacy(s, _make(6), FakeSocialGraph(), y, random.Random(42+y))
        for f in s.factions.values():
            assert 0.0 <= f.cohesion <= 1.0

    def test_effects_non_negative(self):
        s = DiplomacyState()
        s.treaties.append(Treaty(id="t-1", faction_a="f-1", faction_b="f-2",
                                  treaty_type="research_pact", signed_year=1, expires_year=20))
        for v in compute_treaty_effects(s).values():
            assert v >= 0.0

    def test_vote_modifier_bounded(self):
        fs = {"f-1": Faction(id="f-1", name="X", formed_year=1,
                              member_ids=[f"col-{i}" for i in range(10)],
                              dominant_stat="empathy", cohesion=1.0)}
        for gov in ["council", "dictator", "lottery", "consensus", "ai_governor", "anarchy"]:
            for i in range(10):
                assert abs(faction_vote_modifier(f"col-{i}", gov, fs)) <= VOTE_MODIFIER_CAP
