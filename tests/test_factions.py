"""Tests for Mars-100 factions module."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import SocialGraph
from src.mars100.factions import (
    Faction, FactionState,
    faction_tick, attempt_formation, attempt_recruitment,
    check_splits, cleanup_dead, update_influence,
    faction_bloc_vote, action_weight_modifier,
    _stat_distance, _ideology_from_dominant,
    MIN_FACTION_SIZE, MAX_FACTIONS,
    FORMATION_TRUST_THRESHOLD, STAT_SIMILARITY_THRESHOLD,
)


def _make_colonist(cid: str, dominant: str = "resolve", **overrides) -> Colonist:
    """Helper: create a colonist with given dominant stat."""
    stats = {s: 0.3 for s in ("resolve", "improvisation", "empathy",
                                "hoarding", "faith", "paranoia")}
    stats[dominant] = 0.9
    return Colonist(
        id=cid, name=f"Test-{cid}", element="fire", archetype="test",
        stats=ColonistStats.from_dict(stats),
        skills=ColonistSkills(), decision_expr="(+ resolve empathy)",
        **overrides,
    )


def _make_social(colonists: list[Colonist], trust: float = 0.7) -> SocialGraph:
    """Helper: create a social graph with uniform trust."""
    sg = SocialGraph()
    rng = random.Random(1)
    ids = [c.id for c in colonists]
    sg.initialize(ids, rng)
    for a in ids:
        for b in ids:
            if a != b:
                rel = sg.get(a, b)
                rel.trust = trust
                rel.affection = 0.5
                rel.respect = 0.5
    return sg


class TestStatDistance:
    def test_identical(self):
        s = {"a": 0.5, "b": 0.5}
        assert _stat_distance(s, s) == 0.0

    def test_different(self):
        s1 = {"a": 0.0, "b": 0.0}
        s2 = {"a": 1.0, "b": 1.0}
        d = _stat_distance(s1, s2)
        assert d > 0.9

    def test_partial_overlap(self):
        s1 = {"a": 0.5, "b": 0.5, "c": 0.5}
        s2 = {"a": 0.5, "b": 0.5, "d": 0.0}
        d = _stat_distance(s1, s2)
        assert d == 0.0  # only shared keys "a" and "b" compared


class TestIdeology:
    def test_mapping(self):
        assert _ideology_from_dominant("resolve") == "militarist"
        assert _ideology_from_dominant("empathy") == "communalist"
        assert _ideology_from_dominant("faith") == "spiritualist"

    def test_unknown(self):
        assert _ideology_from_dominant("made_up") == "pragmatist"


class TestFactionState:
    def test_empty(self):
        fs = FactionState()
        assert fs.active_factions() == []
        assert fs.faction_for("nobody") is None

    def test_serialization(self):
        fs = FactionState()
        f = Faction(id="f-0", name="Test", founded_year=5,
                    ideology="militarist", member_ids=["a", "b"])
        fs.factions["f-0"] = f
        d = fs.to_dict()
        assert "f-0" in d["factions"]
        fs2 = FactionState.from_dict(d)
        assert fs2.factions["f-0"].name == "Test"
        assert fs2.factions["f-0"].member_ids == ["a", "b"]

    def test_faction_for(self):
        fs = FactionState()
        f = Faction(id="f-0", name="Test", founded_year=1,
                    ideology="militarist", member_ids=["a", "b"])
        fs.factions["f-0"] = f
        assert fs.faction_for("a") is f
        assert fs.faction_for("c") is None


class TestFormation:
    def test_similar_stats_high_trust(self):
        """Colonists with similar dominant stats and high trust form a faction."""
        colonists = [_make_colonist("a", "resolve"),
                     _make_colonist("b", "resolve"),
                     _make_colonist("c", "resolve")]
        social = _make_social(colonists, trust=0.8)
        fs = FactionState()
        events = attempt_formation(colonists, social, fs, year=5, rng=random.Random(42))
        assert len(events) >= 1
        assert events[0]["type"] == "faction_formed"
        assert all(c.faction_id is not None for c in colonists)

    def test_different_stats_no_formation(self):
        """Colonists with very different dominant stats don't form."""
        c_a = _make_colonist("a", "resolve")
        c_a.stats.empathy = 0.1
        c_a.stats.faith = 0.1
        c_b = _make_colonist("b", "empathy")
        c_b.stats.resolve = 0.1
        c_b.stats.paranoia = 0.9
        c_c = _make_colonist("c", "faith")
        c_c.stats.resolve = 0.1
        c_c.stats.hoarding = 0.9
        colonists = [c_a, c_b, c_c]
        social = _make_social(colonists, trust=0.8)
        fs = FactionState()
        events = attempt_formation(colonists, social, fs, year=5, rng=random.Random(42))
        assert len(events) == 0

    def test_low_trust_no_formation(self):
        """High stat similarity but low trust doesn't form."""
        colonists = [_make_colonist("a", "resolve"),
                     _make_colonist("b", "resolve")]
        social = _make_social(colonists, trust=0.2)
        fs = FactionState()
        events = attempt_formation(colonists, social, fs, year=5, rng=random.Random(42))
        assert len(events) == 0

    def test_max_factions_respected(self):
        """Formation stops at MAX_FACTIONS."""
        fs = FactionState()
        for i in range(MAX_FACTIONS):
            fs.factions[f"f-{i}"] = Faction(
                id=f"f-{i}", name=f"F{i}", founded_year=1,
                ideology="militarist", member_ids=[f"x{i}", f"y{i}"])
        colonists = [_make_colonist("a", "resolve"),
                     _make_colonist("b", "resolve")]
        social = _make_social(colonists, trust=0.9)
        events = attempt_formation(colonists, social, fs, year=5, rng=random.Random(42))
        assert len(events) == 0

    def test_already_affiliated_skipped(self):
        """Colonists already in a faction are not considered for formation."""
        c1 = _make_colonist("a", "resolve")
        c1.faction_id = "existing"
        c2 = _make_colonist("b", "resolve")
        colonists = [c1, c2]
        social = _make_social(colonists, trust=0.9)
        fs = FactionState()
        events = attempt_formation(colonists, social, fs, year=5, rng=random.Random(42))
        assert len(events) == 0  # only 1 unaffiliated, below MIN_FACTION_SIZE


class TestRecruitment:
    def test_recruit_unaffiliated(self):
        """Faction recruits a nearby unaffiliated colonist."""
        c1 = _make_colonist("a", "resolve")
        c2 = _make_colonist("b", "resolve")
        c3 = _make_colonist("c", "resolve")
        c1.faction_id = "f-0"
        c2.faction_id = "f-0"
        social = _make_social([c1, c2, c3], trust=0.8)
        fs = FactionState()
        fs.factions["f-0"] = Faction(
            id="f-0", name="Test", founded_year=1,
            ideology="militarist", member_ids=["a", "b"])
        # Run multiple times — recruitment is probabilistic
        any_recruited = False
        for seed in range(20):
            c3_copy = _make_colonist("c", "resolve")
            events = attempt_recruitment([c1, c2, c3_copy], social, fs,
                                         year=5, rng=random.Random(seed))
            if events:
                any_recruited = True
                break
        assert any_recruited


class TestSplits:
    def test_split_on_low_trust(self):
        """Faction splits when internal trust drops below threshold."""
        colonists = [_make_colonist(f"c{i}", "resolve") for i in range(6)]
        for c in colonists:
            c.faction_id = "f-0"
        social = _make_social(colonists, trust=0.2)  # below SPLIT_THRESHOLD
        fs = FactionState()
        fs.factions["f-0"] = Faction(
            id="f-0", name="United", founded_year=1,
            ideology="militarist",
            member_ids=[c.id for c in colonists])
        # Probabilistic — try multiple seeds
        any_split = False
        for seed in range(30):
            fs_copy = FactionState.from_dict(fs.to_dict())
            cols_copy = [_make_colonist(f"c{i}", "resolve") for i in range(6)]
            for c in cols_copy:
                c.faction_id = "f-0"
            events = check_splits(cols_copy, social, fs_copy, year=10,
                                  rng=random.Random(seed))
            if events:
                any_split = True
                break
        assert any_split

    def test_no_split_small_faction(self):
        """Factions with < 4 members don't split."""
        colonists = [_make_colonist("a", "resolve"),
                     _make_colonist("b", "resolve")]
        for c in colonists:
            c.faction_id = "f-0"
        social = _make_social(colonists, trust=0.1)
        fs = FactionState()
        fs.factions["f-0"] = Faction(
            id="f-0", name="Tiny", founded_year=1,
            ideology="militarist", member_ids=["a", "b"])
        events = check_splits(colonists, social, fs, year=10, rng=random.Random(42))
        assert len(events) == 0


class TestCleanup:
    def test_dead_removed(self):
        """Dead colonists are removed from faction membership."""
        c1 = _make_colonist("a", "resolve")
        c2 = _make_colonist("b", "resolve")
        c2.alive = False
        c1.faction_id = "f-0"
        c2.faction_id = "f-0"
        fs = FactionState()
        fs.factions["f-0"] = Faction(
            id="f-0", name="Test", founded_year=1,
            ideology="militarist", member_ids=["a", "b"],
            influence=0.5)
        events = cleanup_dead([c1, c2], fs, year=10)
        # Faction dissolved since only 1 member left (<= DISSOLUTION_SIZE)
        assert len(events) == 1
        assert events[0]["type"] == "faction_dissolved"
        assert "f-0" not in fs.factions
        assert c1.faction_id is None

    def test_exiled_removed(self):
        """Exiled colonists removed from faction."""
        c1 = _make_colonist("a", "resolve")
        c2 = _make_colonist("b", "resolve")
        c3 = _make_colonist("c", "resolve")
        c2.exiled = True
        for c in [c1, c2, c3]:
            c.faction_id = "f-0"
        fs = FactionState()
        fs.factions["f-0"] = Faction(
            id="f-0", name="Test", founded_year=1,
            ideology="militarist", member_ids=["a", "b", "c"])
        cleanup_dead([c1, c2, c3], fs, year=10)
        assert "b" not in fs.factions["f-0"].member_ids
        assert len(fs.factions["f-0"].member_ids) == 2


class TestInfluence:
    def test_influence_proportional(self):
        colonists = [_make_colonist(f"c{i}", "resolve") for i in range(10)]
        for c in colonists[:4]:
            c.faction_id = "f-0"
        fs = FactionState()
        fs.factions["f-0"] = Faction(
            id="f-0", name="Big", founded_year=1,
            ideology="militarist", member_ids=[c.id for c in colonists[:4]])
        update_influence(colonists, fs)
        assert abs(fs.factions["f-0"].influence - 0.4) < 0.01


class TestBlocVote:
    def test_faction_member_votes_with_bloc(self):
        fs = FactionState()
        fs.factions["f-0"] = Faction(
            id="f-0", name="Hawks", founded_year=1,
            ideology="militarist", member_ids=["voter"])
        # Militarists favor dictator (0.7 affinity)
        votes = [faction_bloc_vote("voter", "dictator", fs, random.Random(s))
                 for s in range(100)]
        # Should vote True roughly 70% of the time
        true_count = sum(1 for v in votes if v is True)
        assert 50 < true_count < 90

    def test_no_faction_returns_none(self):
        fs = FactionState()
        assert faction_bloc_vote("nobody", "council", fs, random.Random(42)) is None


class TestActionWeightModifier:
    def test_faction_boosts_action(self):
        fs = FactionState()
        fs.factions["f-0"] = Faction(
            id="f-0", name="Builders", founded_year=1,
            ideology="innovator", member_ids=["coder"])
        assert action_weight_modifier("coder", "code", fs) > 0
        assert action_weight_modifier("coder", "pray", fs) == 0.0

    def test_no_faction_no_boost(self):
        fs = FactionState()
        assert action_weight_modifier("nobody", "code", fs) == 0.0


class TestFactionTick:
    def test_full_cycle(self):
        """Run faction_tick for multiple years and verify dynamics."""
        colonists = create_founding_ten(seed=42)
        social = SocialGraph()
        social.initialize([c.id for c in colonists], random.Random(42))
        # Boost trust between first 3 colonists to trigger formation
        for a in colonists[:3]:
            for b in colonists[:3]:
                if a.id != b.id:
                    rel = social.get(a.id, b.id)
                    rel.trust = 0.85
        fs = FactionState()
        rng = random.Random(42)
        all_events: list[dict] = []
        for year in range(1, 51):
            events = faction_tick(colonists, social, fs, year, rng)
            all_events.extend(events)
        # Factions should have formed at some point
        assert len(all_events) > 0
        # Faction state should be serializable
        d = fs.to_dict()
        assert isinstance(d, dict)

    def test_deterministic(self):
        """Same seed produces same faction history."""
        def run_once(seed):
            colonists = create_founding_ten(seed=seed)
            social = SocialGraph()
            social.initialize([c.id for c in colonists], random.Random(seed))
            for a in colonists[:3]:
                for b in colonists[:3]:
                    if a.id != b.id:
                        social.get(a.id, b.id).trust = 0.85
            fs = FactionState()
            rng = random.Random(seed)
            events = []
            for year in range(1, 26):
                events.extend(faction_tick(colonists, social, fs, year, rng))
            return [e["type"] for e in events]
        assert run_once(99) == run_once(99)

    def test_no_phantom_members(self):
        """After death cleanup, no faction references a dead colonist."""
        colonists = [_make_colonist(f"c{i}", "resolve") for i in range(5)]
        for c in colonists[:3]:
            c.faction_id = "f-0"
        colonists[1].alive = False
        social = _make_social(colonists, trust=0.8)
        fs = FactionState()
        fs.factions["f-0"] = Faction(
            id="f-0", name="Test", founded_year=1,
            ideology="militarist",
            member_ids=[c.id for c in colonists[:3]])
        faction_tick(colonists, social, fs, year=10, rng=random.Random(42))
        for f in fs.factions.values():
            for mid in f.member_ids:
                c = next((c for c in colonists if c.id == mid), None)
                assert c is not None and c.is_active()


class TestEngineIntegration:
    def test_factions_appear_in_year_result(self):
        """Engine tick produces faction_events and faction_state."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=50)
        result = engine.run()
        d = result.to_dict()
        # At least some years should have faction_state
        has_factions = any(yr.get("faction_state", {}).get("factions")
                          for yr in d["years"])
        # faction_events key exists on every year
        assert all("faction_events" in yr for yr in d["years"])
        assert all("faction_state" in yr for yr in d["years"])

    def test_colonist_faction_id_serialized(self):
        """Colonist snapshots include faction_id."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        d = result.to_dict()
        last_year = d["years"][-1]
        for snap in last_year["colonist_snapshots"]:
            assert "faction_id" in snap

    def test_50_year_with_factions_stable(self):
        """50-year sim with factions doesn't crash or produce unbounded values."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=77, total_years=50)
        result = engine.run()
        d = result.to_dict()
        assert len(d["years"]) > 0
        for yr in d["years"]:
            for name, val in yr.get("resources_after", {}).items():
                assert 0.0 <= val <= 1.0, f"resource {name} out of bounds: {val}"
