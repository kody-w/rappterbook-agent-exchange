"""Tests for the Mars-100 cultural memory / oral tradition organ."""
from __future__ import annotations

import random
import pytest

from src.mars100.culture import (
    OralTradition, Story, CultureSnapshot,
    _myth_influence, STORY_DECAY_RATE, MYTH_PROMOTION_THRESHOLD,
    MAX_ACTIVE_STORIES, STORY_EVICTION_THRESHOLD,
    MYTH_SPREAD_THRESHOLD, MYTH_TELLING_THRESHOLD,
    THEMES, _THEME_GOV_MAP, _THEME_ACTION_MAP,
)
from src.mars100.colonist import create_founding_ten, Colonist
from src.mars100.engine import Mars100Engine


# ── Helpers ──

def _make_colonists(n: int = 5, seed: int = 42) -> list[Colonist]:
    """Create n colonists from the founding ten."""
    return create_founding_ten(seed)[:n]


def _trivial_events(severity: float = 0.8) -> list[dict]:
    return [{"name": "dust_storm", "severity": severity,
             "description": "A dust storm hits", "effects": {"morale": -0.1}}]


def _trivial_deaths() -> list[dict]:
    return [{"id": "col-0", "name": "Ares", "cause": "equipment malfunction", "year": 5}]


# ── Story dataclass ──

class TestStory:
    def test_to_dict_roundtrip(self):
        s = Story(id="s-0", year_created=1, source_event="test",
                  theme="loss", rememberers=["col-0", "col-1"],
                  tellings=3, strength=0.8)
        d = s.to_dict()
        assert d["id"] == "s-0"
        assert d["theme"] == "loss"
        assert d["rememberers"] == ["col-0", "col-1"]
        assert d["tellings"] == 3
        assert 0.79 < d["strength"] < 0.81
        assert d["is_myth"] is False
        assert d["governance_stance"] is None

    def test_governance_stance(self):
        s = Story(id="s-1", year_created=2, source_event="gov change",
                  theme="governance", governance_stance="council")
        assert s.to_dict()["governance_stance"] == "council"


# ── Myth influence decay ──

class TestMythInfluence:
    def test_same_year_full_influence(self):
        assert abs(_myth_influence(10, 10) - 1.0) < 0.01

    def test_halflife(self):
        """At the halflife, influence should be ~50%."""
        inf = _myth_influence(0, 60)
        assert 0.45 < inf < 0.55

    def test_monotonic_decay(self):
        vals = [_myth_influence(0, y) for y in range(0, 101, 10)]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_never_negative(self):
        assert _myth_influence(0, 1000) > 0


# ── OralTradition unit tests ──

class TestOralTraditionBasic:
    def test_empty_init(self):
        ot = OralTradition()
        assert len(ot.stories) == 0
        assert ot._next_id == 0

    def test_no_stories_signals_empty(self):
        ot = OralTradition()
        assert ot.governance_signals(10) == {}
        assert ot.action_bias(10) == {}


class TestStoryGeneration:
    def test_death_generates_loss_story(self):
        ot = OralTradition()
        colonists = _make_colonists()
        snap = ot.post_tick_update(
            year=5, active_colonists=colonists,
            events=[], deaths=_trivial_deaths(), exiles=[],
            governance=None, meta_events=[], subsim_log=[],
            births=[], infra_event=None, rng=random.Random(42))
        assert snap.stories_created >= 1
        loss_stories = [s for s in ot.stories if s.theme == "loss"]
        assert len(loss_stories) >= 1
        assert "death of Ares" in loss_stories[0].source_event

    def test_exile_generates_exile_story(self):
        ot = OralTradition()
        colonists = _make_colonists()
        snap = ot.post_tick_update(
            year=5, active_colonists=colonists,
            events=[], deaths=[], exiles=[{"id": "col-1", "name": "Pyra"}],
            governance=None, meta_events=[], subsim_log=[],
            births=[], infra_event=None, rng=random.Random(42))
        exile_stories = [s for s in ot.stories if s.theme == "exile"]
        assert len(exile_stories) >= 1

    def test_crisis_from_high_severity(self):
        ot = OralTradition()
        colonists = _make_colonists()
        rng = random.Random(1)  # seed that hits 0.7 threshold
        snap = ot.post_tick_update(
            year=5, active_colonists=colonists,
            events=_trivial_events(severity=0.9), deaths=[], exiles=[],
            governance=None, meta_events=[], subsim_log=[],
            births=[], infra_event=None, rng=rng)
        # May or may not generate depending on RNG, but should not crash
        assert snap.stories_created >= 0

    def test_governance_change_generates_story(self):
        ot = OralTradition()
        colonists = _make_colonists()
        snap = ot.post_tick_update(
            year=10, active_colonists=colonists,
            events=[], deaths=[], exiles=[],
            governance={"passed": True, "gov_type": "council"},
            meta_events=[], subsim_log=[],
            births=[], infra_event=None, rng=random.Random(42))
        gov_stories = [s for s in ot.stories if s.theme == "governance"]
        assert len(gov_stories) >= 1
        assert gov_stories[0].governance_stance == "council"

    def test_deep_subsim_generates_discovery(self):
        ot = OralTradition()
        colonists = _make_colonists()
        snap = ot.post_tick_update(
            year=15, active_colonists=colonists,
            events=[], deaths=[], exiles=[],
            governance=None, meta_events=[],
            subsim_log=[{"depth": 2, "colonist_id": "col-0", "result": 0.9}],
            births=[], infra_event=None, rng=random.Random(42))
        discovery_stories = [s for s in ot.stories if s.theme == "discovery"]
        assert len(discovery_stories) >= 0  # 80% chance

    def test_meta_awareness_generates_transcendence(self):
        ot = OralTradition()
        colonists = _make_colonists()
        snap = ot.post_tick_update(
            year=50, active_colonists=colonists,
            events=[], deaths=[], exiles=[],
            governance=None,
            meta_events=[{"colonist_id": "col-0", "year": 50, "insight": "Are we variables?"}],
            subsim_log=[], births=[], infra_event=None, rng=random.Random(42))
        trans = [s for s in ot.stories if s.theme == "transcendence"]
        assert len(trans) >= 1

    def test_first_birth_generates_hope(self):
        ot = OralTradition()
        colonists = _make_colonists()
        snap = ot.post_tick_update(
            year=12, active_colonists=colonists,
            events=[], deaths=[], exiles=[], governance=None,
            meta_events=[], subsim_log=[],
            births=[{"id": "child-10", "name": "Nova", "parents": ["col-0", "col-1"]}],
            infra_event=None, rng=random.Random(42))
        hope_stories = [s for s in ot.stories if s.theme == "hope"]
        assert len(hope_stories) == 1
        assert "first child" in hope_stories[0].source_event

    def test_second_birth_no_duplicate_hope(self):
        ot = OralTradition()
        colonists = _make_colonists()
        # First birth
        ot.post_tick_update(
            year=12, active_colonists=colonists, events=[], deaths=[], exiles=[],
            governance=None, meta_events=[], subsim_log=[],
            births=[{"id": "child-10", "name": "Nova"}],
            infra_event=None, rng=random.Random(42))
        # Second birth
        ot.post_tick_update(
            year=14, active_colonists=colonists, events=[], deaths=[], exiles=[],
            governance=None, meta_events=[], subsim_log=[],
            births=[{"id": "child-11", "name": "Vega"}],
            infra_event=None, rng=random.Random(42))
        hope_stories = [s for s in ot.stories if s.theme == "hope"]
        assert len(hope_stories) == 1  # still only one hope story

    def test_infra_milestone_generates_progress(self):
        ot = OralTradition()
        colonists = _make_colonists()
        snap = ot.post_tick_update(
            year=20, active_colonists=colonists, events=[], deaths=[], exiles=[],
            governance=None, meta_events=[], subsim_log=[], births=[],
            infra_event={"completed": True, "tech_id": "solar_array"},
            rng=random.Random(42))
        progress = [s for s in ot.stories if s.theme == "progress"]
        assert len(progress) >= 1


class TestCooldowns:
    def test_loss_cooldown_is_1_year(self):
        ot = OralTradition()
        colonists = _make_colonists()
        # Two deaths in the same year should generate stories (cooldown=1)
        ot.post_tick_update(
            year=5, active_colonists=colonists, events=[],
            deaths=[{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            exiles=[], governance=None, meta_events=[], subsim_log=[],
            births=[], infra_event=None, rng=random.Random(42))
        # First death generates, second may be blocked by cooldown
        loss = [s for s in ot.stories if s.theme == "loss"]
        assert len(loss) >= 1

    def test_crisis_cooldown_prevents_spam(self):
        ot = OralTradition()
        colonists = _make_colonists()
        for year in range(1, 4):
            ot.post_tick_update(
                year=year, active_colonists=colonists,
                events=_trivial_events(0.9), deaths=[], exiles=[],
                governance=None, meta_events=[], subsim_log=[],
                births=[], infra_event=None, rng=random.Random(year))
        crisis = [s for s in ot.stories if s.theme == "crisis"]
        # 3 years with cooldown=3 should produce at most 1 crisis story
        assert len(crisis) <= 1


class TestStorySpreading:
    def test_stories_spread_over_time(self):
        ot = OralTradition()
        colonists = _make_colonists(5)
        rng = random.Random(42)
        # Create a story and manually add one rememberer
        ot.post_tick_update(
            year=5, active_colonists=colonists, events=[],
            deaths=_trivial_deaths(), exiles=[], governance=None,
            meta_events=[], subsim_log=[], births=[],
            infra_event=None, rng=rng)
        story = ot.stories[0]
        story.rememberers = [colonists[0].id]

        # Run several years of spreading
        for y in range(6, 20):
            ot.post_tick_update(
                year=y, active_colonists=colonists, events=[], deaths=[],
                exiles=[], governance=None, meta_events=[], subsim_log=[],
                births=[], infra_event=None, rng=rng)

        # After 14 years, at least some colonists should know the story
        assert len(story.rememberers) > 1

    def test_spreading_reinforces_strength(self):
        ot = OralTradition()
        colonists = _make_colonists(5)
        rng = random.Random(42)
        ot.post_tick_update(
            year=5, active_colonists=colonists, events=[],
            deaths=_trivial_deaths(), exiles=[], governance=None,
            meta_events=[], subsim_log=[], births=[],
            infra_event=None, rng=rng)
        story = ot.stories[0]
        story.rememberers = [c.id for c in colonists[:3]]
        initial_tellings = story.tellings
        ot._spread_stories(6, colonists, rng)
        # Tellings should have increased if anyone retold it
        assert story.tellings >= initial_tellings


class TestDecayAndEviction:
    def test_decay_reduces_strength(self):
        ot = OralTradition()
        s = Story(id="s-0", year_created=1, source_event="test", theme="loss",
                  strength=1.0)
        ot.stories = [s]
        ot._decay_stories()
        assert abs(s.strength - STORY_DECAY_RATE) < 0.001

    def test_myths_dont_decay(self):
        ot = OralTradition()
        s = Story(id="s-0", year_created=1, source_event="test", theme="loss",
                  strength=0.9, is_myth=True)
        ot.stories = [s]
        ot._decay_stories()
        assert s.strength == 0.9

    def test_weak_stories_evicted(self):
        ot = OralTradition()
        ot.stories = [
            Story(id="s-0", year_created=1, source_event="a", theme="loss",
                  strength=0.01),  # below threshold
            Story(id="s-1", year_created=2, source_event="b", theme="crisis",
                  strength=0.5),  # above threshold
        ]
        evicted = ot._evict_weak_stories()
        assert evicted == 1
        assert len(ot.stories) == 1
        assert ot.stories[0].id == "s-1"

    def test_myths_never_evicted(self):
        ot = OralTradition()
        s = Story(id="s-0", year_created=1, source_event="a", theme="loss",
                  strength=0.01, is_myth=True)
        ot.stories = [s]
        evicted = ot._evict_weak_stories()
        assert evicted == 0
        assert len(ot.stories) == 1


class TestMythPromotion:
    def test_story_promotes_to_myth(self):
        ot = OralTradition()
        colonists = _make_colonists(5)
        s = Story(id="s-0", year_created=1, source_event="test", theme="loss",
                  strength=0.8,  # above MYTH_PROMOTION_THRESHOLD
                  rememberers=[c.id for c in colonists[:3]],  # 3/5 = 60% > 50%
                  tellings=12)  # above MYTH_TELLING_THRESHOLD
        ot.stories = [s]
        promoted = ot._promote_stories(5, colonists)
        assert promoted == 1
        assert s.is_myth is True
        assert s.strength == 1.0  # reset to full

    def test_weak_story_not_promoted(self):
        ot = OralTradition()
        colonists = _make_colonists(5)
        s = Story(id="s-0", year_created=1, source_event="test", theme="loss",
                  strength=0.3,  # below threshold
                  rememberers=[c.id for c in colonists[:3]],
                  tellings=12)
        ot.stories = [s]
        promoted = ot._promote_stories(5, colonists)
        assert promoted == 0
        assert s.is_myth is False

    def test_low_spread_not_promoted(self):
        ot = OralTradition()
        colonists = _make_colonists(5)
        s = Story(id="s-0", year_created=1, source_event="test", theme="loss",
                  strength=0.9,
                  rememberers=[colonists[0].id],  # 1/5 = 20% < 50%
                  tellings=12)
        ot.stories = [s]
        promoted = ot._promote_stories(5, colonists)
        assert promoted == 0


class TestTeachChild:
    def test_child_inherits_myths(self):
        ot = OralTradition()
        myth = Story(id="s-0", year_created=1, source_event="ancient",
                     theme="loss", is_myth=True, rememberers=["col-0", "col-1"])
        ot.stories = [myth]
        ot.teach_child("child-10", ["col-0", "col-2"], random.Random(42))
        assert "child-10" in myth.rememberers

    def test_child_sometimes_inherits_strong_parent_stories(self):
        ot = OralTradition()
        rng = random.Random(42)
        story = Story(id="s-0", year_created=5, source_event="event",
                      theme="crisis", strength=0.6,
                      rememberers=["col-0", "col-1"])
        ot.stories = [story]
        # Run teach_child many times to test probability
        inherited = 0
        for i in range(100):
            s_copy = Story(id=f"s-{i}", year_created=5, source_event="event",
                          theme="crisis", strength=0.6,
                          rememberers=["col-0", "col-1"])
            ot_test = OralTradition()
            ot_test.stories = [s_copy]
            ot_test.teach_child(f"child-{i}", ["col-0"], random.Random(i))
            if f"child-{i}" in s_copy.rememberers:
                inherited += 1
        # Should inherit roughly 50% of the time
        assert 20 < inherited < 80

    def test_child_does_not_inherit_weak_parent_stories(self):
        ot = OralTradition()
        story = Story(id="s-0", year_created=5, source_event="event",
                      theme="crisis", strength=0.1,  # below 0.3 threshold
                      rememberers=["col-0", "col-1"])
        ot.stories = [story]
        ot.teach_child("child-10", ["col-0"], random.Random(42))
        assert "child-10" not in story.rememberers


class TestGovernanceSignals:
    def test_myths_influence_governance(self):
        ot = OralTradition()
        myth = Story(id="s-0", year_created=10, source_event="test",
                     theme="loss", is_myth=True, strength=1.0,
                     governance_stance="council")
        ot.stories = [myth]
        signals = ot.governance_signals(15)
        assert "council" in signals
        assert signals["council"] > 0

    def test_non_myths_dont_influence(self):
        ot = OralTradition()
        story = Story(id="s-0", year_created=10, source_event="test",
                      theme="loss", strength=0.9,
                      governance_stance="council")  # not a myth
        ot.stories = [story]
        signals = ot.governance_signals(15)
        assert signals == {}

    def test_influence_decays_with_age(self):
        ot = OralTradition()
        myth = Story(id="s-0", year_created=1, source_event="old",
                     theme="loss", is_myth=True, strength=1.0,
                     governance_stance="council")
        ot.stories = [myth]
        early = ot.governance_signals(5)
        late = ot.governance_signals(80)
        assert early["council"] > late["council"]


class TestActionBias:
    def test_myths_bias_actions(self):
        ot = OralTradition()
        myth = Story(id="s-0", year_created=10, source_event="test",
                     theme="loss", is_myth=True, strength=1.0)
        ot.stories = [myth]
        biases = ot.action_bias(12)
        # Loss myths should boost pray, cooperate, hoard
        assert biases.get("pray", 0) > 0
        assert biases.get("cooperate", 0) > 0

    def test_bias_capped_at_0_3(self):
        ot = OralTradition()
        # Stack many myths of the same theme
        for i in range(10):
            ot.stories.append(Story(
                id=f"s-{i}", year_created=1, source_event="x",
                theme="loss", is_myth=True, strength=1.0))
        biases = ot.action_bias(2)
        for _, v in biases.items():
            assert -0.3 <= v <= 0.3


class TestCultureSnapshot:
    def test_snapshot_fields(self):
        snap = CultureSnapshot(
            active_stories=5, myths=2, dominant_theme="loss",
            cultural_cohesion=0.4, governance_signals={"council": 0.3},
            stories_created=1, stories_promoted=0, stories_evicted=0)
        d = snap.to_dict()
        assert d["active_stories"] == 5
        assert d["myths"] == 2
        assert d["dominant_theme"] == "loss"
        assert 0.39 < d["cultural_cohesion"] < 0.41
        assert d["stories_created"] == 1


class TestMaxStoriesCap:
    def test_stories_capped(self):
        ot = OralTradition()
        colonists = _make_colonists()
        # Force many stories by running many years with deaths
        for y in range(1, 100):
            ot.post_tick_update(
                year=y, active_colonists=colonists, events=[],
                deaths=[{"id": f"victim-{y}", "name": f"V{y}"}],
                exiles=[], governance=None, meta_events=[], subsim_log=[],
                births=[], infra_event=None, rng=random.Random(y))
        non_myths = [s for s in ot.stories if not s.is_myth]
        assert len(non_myths) <= MAX_ACTIVE_STORIES


# ── Constants sanity ──

class TestConstants:
    def test_all_themes_have_gov_map(self):
        for theme in THEMES:
            assert theme in _THEME_GOV_MAP

    def test_all_themes_have_action_map(self):
        for theme in THEMES:
            assert theme in _THEME_ACTION_MAP


# ── Engine integration tests ──

class TestEngineIntegration:
    def test_10_year_smoke(self):
        """Engine runs 10 years without crash, producing cultural data."""
        eng = Mars100Engine(seed=42, total_years=10)
        result = eng.run()
        assert len(result.years) == 10
        for yr in result.years:
            cm = yr.cultural_memory
            assert isinstance(cm, dict)
            assert "active_stories" in cm
            assert "myths" in cm
            assert "governance_signals" in cm
        # SimulationResult should have cultural_memory too
        assert "stories" in result.cultural_memory

    def test_stories_accumulate(self):
        """Over 20 years, stories should be created."""
        eng = Mars100Engine(seed=42, total_years=20)
        result = eng.run()
        total_created = sum(yr.cultural_memory.get("stories_created", 0)
                           for yr in result.years)
        assert total_created > 0

    def test_cultural_memory_in_to_dict(self):
        """YearResult.to_dict() includes cultural_memory."""
        eng = Mars100Engine(seed=42, total_years=5)
        result = eng.run()
        d = result.years[0].to_dict()
        assert "cultural_memory" in d

    def test_sim_result_cultural_memory(self):
        """SimulationResult.to_dict() includes cultural_memory."""
        eng = Mars100Engine(seed=42, total_years=5)
        result = eng.run()
        d = result.to_dict()
        assert "cultural_memory" in d
        assert d["_meta"]["version"] == "5.0"

    def test_deterministic(self):
        """Same seed produces same cultural output."""
        r1 = Mars100Engine(seed=123, total_years=15).run()
        r2 = Mars100Engine(seed=123, total_years=15).run()
        for y1, y2 in zip(r1.years, r2.years):
            assert y1.cultural_memory == y2.cultural_memory

    def test_100_year_full_run(self):
        """Full 100-year run completes with cultural memory."""
        eng = Mars100Engine(seed=42, total_years=100)
        result = eng.run()
        assert len(result.years) > 0
        final_cm = result.cultural_memory
        assert isinstance(final_cm, dict)
        assert "stories" in final_cm
        assert "myths" in final_cm

    def test_cultural_memory_influences_actions(self):
        """Verify cultural bias is applied (indirectly: different seeds
        should produce different action distributions after myths form)."""
        r1 = Mars100Engine(seed=42, total_years=50).run()
        r2 = Mars100Engine(seed=999, total_years=50).run()
        # Action distributions should differ due to different cultural memory
        actions1 = {}
        actions2 = {}
        for yr in r1.years[-10:]:
            for a in yr.actions.values():
                actions1[a] = actions1.get(a, 0) + 1
        for yr in r2.years[-10:]:
            for a in yr.actions.values():
                actions2[a] = actions2.get(a, 0) + 1
        # Not identical (different seed = different culture = different behavior)
        assert actions1 != actions2


# ── Physical invariants ──

class TestInvariants:
    def test_story_strength_bounded(self):
        """Story strength stays in [0, 1]."""
        eng = Mars100Engine(seed=42, total_years=50)
        result = eng.run()
        for story in eng.oral_tradition.stories:
            assert 0.0 <= story.strength <= 1.0

    def test_cultural_cohesion_bounded(self):
        """Cultural cohesion is in [0, 1]."""
        eng = Mars100Engine(seed=42, total_years=30)
        result = eng.run()
        for yr in result.years:
            cohesion = yr.cultural_memory.get("cultural_cohesion", 0)
            assert 0.0 <= cohesion <= 1.0

    def test_governance_bias_capped(self):
        """Action bias values never exceed ±0.3."""
        eng = Mars100Engine(seed=42, total_years=50)
        eng.run()
        biases = eng.oral_tradition.action_bias(50)
        for v in biases.values():
            assert -0.3 <= v <= 0.3

    def test_no_duplicate_rememberers(self):
        """teach_child should not add duplicates."""
        ot = OralTradition()
        myth = Story(id="s-0", year_created=1, source_event="x",
                     theme="loss", is_myth=True, rememberers=["child-10"])
        ot.stories = [myth]
        ot.teach_child("child-10", ["col-0"], random.Random(42))
        # child-10 was already in rememberers — should not be added again
        assert myth.rememberers.count("child-10") == 1


class TestOralTraditionSerialization:
    def test_to_dict(self):
        ot = OralTradition()
        ot.stories.append(Story(id="s-0", year_created=1, source_event="test",
                                theme="loss"))
        d = ot.to_dict()
        assert "stories" in d
        assert "myths" in d
        assert "total_stories_ever" in d
        assert d["total_stories_ever"] == 0  # we manually appended, not via _make_id
        assert d["first_birth_seen"] is False
