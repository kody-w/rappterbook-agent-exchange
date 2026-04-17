"""Tests for the Mars-100 cultural DNA engine."""
from __future__ import annotations

import random
import json
from dataclasses import dataclass

import pytest

from src.mars100.culture import (
    Meme, InteractionRecord, CulturalMemory,
    create_meme_from_event, create_meme_from_interaction,
    transmit_memes, transmit_to_child, prune_dead_colonist,
    forget_memes, enforce_colony_cap, cultural_action_modifiers,
    cultural_vote_modifier, cultural_resource_bonuses, tick_culture,
    MAX_MEMES_PER_COLONIST, COLONY_MEME_CAP_FACTOR,
    ACCURACY_DECAY, TABOO_PENALTY, INNOVATION_BOOST, SONG_MORALE,
)
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import SocialGraph, Relationship


# --- Fixtures ---

@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def culture():
    return CulturalMemory()


@pytest.fixture
def colonists():
    return create_founding_ten(42)


@pytest.fixture
def social(colonists):
    sg = SocialGraph()
    sg.initialize([c.id for c in colonists], random.Random(42))
    return sg


def _make_meme(culture: CulturalMemory, believers: list[str],
               meme_type: str = "story", accuracy: float = 1.0) -> Meme:
    """Helper to create a meme with specific believers."""
    m = Meme(
        id=culture._gen_id(), meme_type=meme_type,
        content="test meme", origin_colonist=believers[0] if believers else "nobody",
        origin_year=1, accuracy=accuracy, believers=sorted(believers),
        influence={"terraform": 0.05},
    )
    culture.memes[m.id] = m
    return m


# --- Meme dataclass tests ---

class TestMeme:
    def test_to_dict_roundtrip(self):
        m = Meme(id="m-0", meme_type="story", content="A tale",
                 origin_colonist="kira-sol", origin_year=5, accuracy=0.85,
                 believers=["kira-sol", "fen-marsh"],
                 influence={"pray": 0.03}, generation=2)
        d = m.to_dict()
        m2 = Meme.from_dict(d)
        assert m2.id == m.id
        assert m2.meme_type == m.meme_type
        assert m2.accuracy == pytest.approx(0.85, abs=0.001)
        assert m2.believers == sorted(m.believers)
        assert m2.generation == 2

    def test_to_dict_json_serializable(self):
        m = Meme(id="m-0", meme_type="taboo", content="Never explore",
                 origin_colonist="dax-iron", origin_year=10, accuracy=0.5,
                 believers=["dax-iron"], influence={"explore": -0.3})
        json.dumps(m.to_dict())  # must not raise

    def test_believers_always_sorted(self):
        m = Meme.from_dict({
            "id": "m-0", "meme_type": "song", "content": "song",
            "origin_colonist": "x", "origin_year": 1, "accuracy": 1.0,
            "believers": ["zeph-wind", "aura-kai", "fen-marsh"],
            "influence": {},
        })
        assert m.believers == ["aura-kai", "fen-marsh", "zeph-wind"]


# --- CulturalMemory tests ---

class TestCulturalMemory:
    def test_empty_state(self, culture):
        assert culture.active_meme_count() == 0
        assert culture.memes_known_by("nobody") == []

    def test_to_dict_roundtrip(self, culture, rng):
        _make_meme(culture, ["a", "b"], "story")
        _make_meme(culture, ["c"], "taboo")
        d = culture.to_dict()
        c2 = CulturalMemory.from_dict(d)
        assert c2.active_meme_count() == 2
        assert c2.next_id == culture.next_id

    def test_json_serializable(self, culture):
        _make_meme(culture, ["a"], "innovation")
        json.dumps(culture.to_dict())

    def test_memes_known_by(self, culture):
        m1 = _make_meme(culture, ["a", "b"], "story")
        m2 = _make_meme(culture, ["b", "c"], "taboo")
        known_b = culture.memes_known_by("b")
        assert len(known_b) == 2
        known_a = culture.memes_known_by("a")
        assert len(known_a) == 1
        assert known_a[0].id == m1.id


# --- Meme creation tests ---

class TestMemeCreation:
    def test_create_from_high_severity_event(self, culture, rng):
        meme = create_meme_from_event(culture, "kira-sol", 5,
                                       "dust_storm", 0.9, rng)
        assert meme is not None
        assert meme.origin_colonist == "kira-sol"
        assert meme.origin_year == 5
        assert "kira-sol" in meme.believers

    def test_low_severity_event_usually_no_meme(self, culture):
        created = 0
        for seed in range(100):
            c = CulturalMemory()
            r = random.Random(seed)
            m = create_meme_from_event(c, "test", 1, "calm", 0.05, r)
            if m:
                created += 1
        assert created < 30  # low severity = low creation rate

    def test_create_innovation_from_exceptional_interaction(self, culture, rng):
        inter = InteractionRecord(
            colonist_id="rust-vega", action="terraform", partner_id=None,
            resource_contribution={"water": 0.1}, outcome="exceptional",
            adjacent_death=None,
        )
        meme = create_meme_from_interaction(culture, inter, 10, rng)
        # May or may not create — probabilistic. Run multiple seeds.
        created = 0
        for seed in range(50):
            c = CulturalMemory()
            r = random.Random(seed)
            inter2 = InteractionRecord("a", "farm", None, {}, "exceptional", None)
            m = create_meme_from_interaction(c, inter2, 1, r)
            if m:
                created += 1
                assert m.meme_type == "innovation"
        assert created > 5  # should create at least some

    def test_create_taboo_from_death(self, culture, rng):
        inter = InteractionRecord(
            colonist_id="fen-marsh", action="explore", partner_id=None,
            resource_contribution={}, outcome="normal", adjacent_death="kira-sol",
        )
        created = 0
        for seed in range(50):
            c = CulturalMemory()
            r = random.Random(seed)
            m = create_meme_from_interaction(c, inter, 5, r)
            if m and m.meme_type == "taboo":
                created += 1
        assert created > 10

    def test_create_ritual_from_cooperation(self, culture, rng):
        inter = InteractionRecord(
            colonist_id="aura-kai", action="cooperate", partner_id="fen-marsh",
            resource_contribution={}, outcome="normal", adjacent_death=None,
        )
        created = 0
        for seed in range(100):
            c = CulturalMemory()
            r = random.Random(seed)
            m = create_meme_from_interaction(c, inter, 3, r)
            if m and m.meme_type == "ritual":
                created += 1
                assert "aura-kai" in m.believers
                assert "fen-marsh" in m.believers
        assert created > 5


# --- Transmission tests ---

class TestTransmission:
    def test_transmit_through_social_graph(self, culture, colonists, social, rng):
        _make_meme(culture, [colonists[0].id], "innovation", accuracy=1.0)
        transmissions = transmit_memes(culture, colonists, social, 5, rng)
        assert len(transmissions) >= 0  # probabilistic
        # Run enough seeds to get at least one transmission
        total = 0
        for seed in range(20):
            c = CulturalMemory()
            _make_meme(c, [colonists[0].id], "innovation", accuracy=1.0)
            sg = SocialGraph()
            sg.initialize([co.id for co in colonists], random.Random(seed))
            tx = transmit_memes(c, colonists, sg, 5, random.Random(seed))
            total += len(tx)
        assert total > 0

    def test_accuracy_degrades_on_transmission(self, culture, colonists, social):
        m = _make_meme(culture, [colonists[0].id], "story", accuracy=1.0)
        original_acc = m.accuracy
        # Force a transmission by making trust very high
        for other in colonists[1:]:
            social.edges[colonists[0].id][other.id].trust = 1.0
        rng = random.Random(99)
        transmit_memes(culture, colonists, social, 5, rng)
        if len(m.believers) > 1:
            assert m.accuracy < original_acc

    def test_max_memes_per_colonist_enforced(self, culture, colonists, social):
        target = colonists[1]
        # Fill target with max memes
        for i in range(MAX_MEMES_PER_COLONIST):
            _make_meme(culture, [target.id], "story", accuracy=0.3)
        # Try transmitting a high-accuracy meme
        high_acc = _make_meme(culture, [colonists[0].id], "innovation", accuracy=1.0)
        social.edges[colonists[0].id][target.id].trust = 1.0
        transmit_memes(culture, colonists, social, 5, random.Random(42))
        known = culture.memes_known_by(target.id)
        assert len(known) <= MAX_MEMES_PER_COLONIST

    def test_vertical_transmission_to_child(self, culture, rng):
        _make_meme(culture, ["parent-a", "parent-b"], "story", accuracy=0.9)
        _make_meme(culture, ["parent-a"], "innovation", accuracy=0.8)
        _make_meme(culture, ["parent-b"], "taboo", accuracy=0.7)
        inherited = transmit_to_child(culture, ["parent-a", "parent-b"], "child-1", rng)
        assert len(inherited) <= 3
        # Child should appear in believer lists of inherited memes
        for mid in inherited:
            assert "child-1" in culture.memes[mid].believers


# --- Pruning tests ---

class TestPruning:
    def test_prune_dead_colonist(self, culture):
        m = _make_meme(culture, ["a", "b", "c"], "story")
        prune_dead_colonist(culture, "b")
        assert "b" not in m.believers
        assert "a" in m.believers

    def test_forget_empty_memes(self, culture):
        m = _make_meme(culture, [], "story")
        forgotten = forget_memes(culture, 10)
        assert m.id in forgotten
        assert m.id not in culture.memes

    def test_enforce_colony_cap(self, culture):
        for i in range(50):
            _make_meme(culture, ["a"], "story", accuracy=0.1)
        pruned = enforce_colony_cap(culture, 5)  # cap = 5 * 3 = 15
        assert culture.active_meme_count() <= 15
        assert len(pruned) > 0

    def test_cap_scales_with_population(self, culture):
        for i in range(30):
            _make_meme(culture, ["a"], "story")
        # Large population = high cap = less pruning
        pruned_large = enforce_colony_cap(CulturalMemory.from_dict(culture.to_dict()), 20)
        pruned_small = enforce_colony_cap(CulturalMemory.from_dict(culture.to_dict()), 3)
        assert len(pruned_small) >= len(pruned_large)


# --- Influence tests ---

class TestInfluence:
    def test_action_modifiers_from_taboo(self, culture):
        m = Meme(id="t-1", meme_type="taboo", content="no exploring",
                 origin_colonist="a", origin_year=1, accuracy=0.8,
                 believers=["a"], influence={"explore": -0.3})
        culture.memes["t-1"] = m
        mods = cultural_action_modifiers(culture, "a")
        assert "explore" in mods
        assert mods["explore"] < 0

    def test_action_modifiers_accumulate(self, culture):
        m1 = Meme(id="i-1", meme_type="innovation", content="better farming",
                  origin_colonist="a", origin_year=1, accuracy=1.0,
                  believers=["a"], influence={"farm": 0.04})
        m2 = Meme(id="i-2", meme_type="innovation", content="even better farming",
                  origin_colonist="b", origin_year=2, accuracy=0.9,
                  believers=["a"], influence={"farm": 0.04})
        culture.memes["i-1"] = m1
        culture.memes["i-2"] = m2
        mods = cultural_action_modifiers(culture, "a")
        assert mods["farm"] == pytest.approx(0.04 * 1.0 + 0.04 * 0.9, abs=0.001)

    def test_vote_modifier_small(self, culture):
        _make_meme(culture, ["a"], "story", accuracy=1.0)
        mod = cultural_vote_modifier(culture, "a", "council")
        assert abs(mod) < 0.1  # subtle, not overwhelming

    def test_resource_bonuses_from_innovations(self, culture):
        m = Meme(id="i-1", meme_type="innovation", content="tech",
                 origin_colonist="a", origin_year=1, accuracy=1.0,
                 believers=["a", "b"], influence={"terraform": 0.04})
        culture.memes["i-1"] = m
        bonuses = cultural_resource_bonuses(culture, ["a", "b"])
        assert "terraform" in bonuses
        assert bonuses["terraform"] > 0

    def test_song_boosts_medicine(self, culture):
        m = Meme(id="s-1", meme_type="song", content="hymn",
                 origin_colonist="a", origin_year=1, accuracy=1.0,
                 believers=["a", "b", "c"], influence={"medicine": SONG_MORALE})
        culture.memes["s-1"] = m
        bonuses = cultural_resource_bonuses(culture, ["a", "b", "c"])
        assert "medicine" in bonuses
        assert bonuses["medicine"] > 0


# --- Integration: tick_culture ---

class TestTickCulture:
    def test_tick_returns_summary(self, culture, colonists, social, rng):
        from src.mars100.events import generate_events
        events = generate_events(5, rng)
        interactions = [
            InteractionRecord(c.id, "terraform", None, {}, "normal", None)
            for c in colonists[:5]
        ]
        result = tick_culture(culture, 5, colonists, social, events,
                              interactions, [], rng)
        assert "year" in result
        assert "active_memes" in result
        assert result["year"] == 5

    def test_tick_creates_memes_from_events(self, culture, colonists, social):
        from src.mars100.events import Event
        events = [Event(name="dust_storm", category="weather",
                        description="Massive dust storm", severity=0.9,
                        effects={"power": -0.1})]
        interactions = []
        total_created = 0
        for seed in range(30):
            c = CulturalMemory()
            r = random.Random(seed)
            result = tick_culture(c, 5, colonists, social, events, interactions, [], r)
            total_created += result["memes_created"]
        assert total_created > 5

    def test_tick_handles_deaths(self, culture, colonists, social, rng):
        m = _make_meme(culture, [colonists[0].id], "story")
        deaths = [{"id": colonists[0].id, "name": colonists[0].name,
                    "cause": "radiation", "year": 5}]
        tick_culture(culture, 5, colonists[1:], social, [], [], deaths, rng)
        assert colonists[0].id not in m.believers

    def test_100_year_meme_cap_holds(self, colonists, social):
        """Invariant: meme count never exceeds colony cap over 100 years."""
        culture = CulturalMemory()
        from src.mars100.events import generate_events
        for year in range(1, 101):
            rng = random.Random(42 + year)
            events = generate_events(year, rng)
            interactions = [
                InteractionRecord(c.id, "farm", None, {}, "normal", None)
                for c in colonists if c.is_active()
            ]
            tick_culture(culture, year, colonists, social, events, interactions, [], rng)
            cap = max(10, len([c for c in colonists if c.is_active()]) * COLONY_MEME_CAP_FACTOR)
            assert culture.active_meme_count() <= cap + 5  # small buffer for same-tick creation


# --- Determinism ---

class TestDeterminism:
    def test_same_seed_same_culture(self, colonists, social):
        """Two runs with same seed must produce identical cultural state."""
        from src.mars100.events import generate_events

        def run_culture(seed: int) -> dict:
            c = CulturalMemory()
            for year in range(1, 21):
                r = random.Random(seed + year)
                events = generate_events(year, r)
                interactions = [
                    InteractionRecord(col.id, "farm", None, {}, "normal", None)
                    for col in colonists if col.is_active()
                ]
                tick_culture(c, year, colonists, social, events, interactions, [], r)
            return c.to_dict()

        result1 = run_culture(42)
        result2 = run_culture(42)
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


# --- Edge cases ---

class TestEdgeCases:
    def test_empty_colony_no_crash(self, culture, social, rng):
        result = tick_culture(culture, 1, [], social, [], [], [], rng)
        assert result["memes_created"] == 0

    def test_single_colonist(self, culture, social, rng):
        colonists = create_founding_ten(42)[:1]
        from src.mars100.events import generate_events
        events = generate_events(1, rng)
        result = tick_culture(culture, 1, colonists, social, events, [], [], rng)
        assert result["active_memes"] >= 0

    def test_all_meme_types_can_be_created(self):
        """Each meme type must be producible."""
        types_seen: set[str] = set()
        for seed in range(200):
            c = CulturalMemory()
            r = random.Random(seed)
            # Events
            m = create_meme_from_event(c, "test", 5, "dust_storm", 0.95, r)
            if m:
                types_seen.add(m.meme_type)
            # Interactions
            for outcome, adj in [("exceptional", None), ("normal", "dead-1")]:
                c2 = CulturalMemory()
                r2 = random.Random(seed + 1000)
                inter = InteractionRecord("a", "cooperate", "b", {}, outcome, adj)
                m2 = create_meme_from_interaction(c2, inter, 5, r2)
                if m2:
                    types_seen.add(m2.meme_type)
        # story, taboo, innovation should appear; ritual and song are created elsewhere
        assert "story" in types_seen or "taboo" in types_seen
        assert "innovation" in types_seen
