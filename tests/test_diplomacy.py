"""Tests for the Mars-100 diplomacy engine."""
from __future__ import annotations

import json
import random

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.diplomacy import (
    ActionOutcome,
    DiplomacyEvent,
    DiplomacyState,
    Faction,
    Treaty,
    check_betrayals,
    detect_factions,
    expire_treaties,
    propose_treaties,
    tick_diplomacy,
    update_factions,
    apply_betrayal_consequences,
    _mutual_trust,
    _dominant_stat,
    FACTION_TRUST_THRESHOLD,
)
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import SocialGraph, Relationship


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_colonists(n: int, seed: int = 42) -> list[Colonist]:
    """Create n simple colonists for testing."""
    rng = random.Random(seed)
    colonists = []
    for i in range(n):
        c = Colonist(
            id=f"c-{i}", name=f"Colonist {i}",
            element=["fire", "water", "earth", "air"][i % 4],
            archetype="test",
            stats=ColonistStats(
                resolve=rng.uniform(0.3, 0.8),
                improvisation=rng.uniform(0.3, 0.8),
                empathy=rng.uniform(0.3, 0.8),
                hoarding=rng.uniform(0.3, 0.8),
                faith=rng.uniform(0.3, 0.8),
                paranoia=rng.uniform(0.3, 0.8),
            ),
            skills=ColonistSkills(),
            decision_expr="(+ resolve empathy)",
        )
        colonists.append(c)
    return colonists


def _make_graph_with_clusters(ids: list[str], cluster_a: list[str],
                              cluster_b: list[str],
                              high_trust: float = 0.8,
                              low_trust: float = 0.3) -> SocialGraph:
    """Create a social graph with two high-trust clusters and low inter-cluster trust."""
    graph = SocialGraph()
    graph.edges = {}
    for a in ids:
        graph.edges[a] = {}
        for b in ids:
            if a != b:
                if (a in cluster_a and b in cluster_a) or (a in cluster_b and b in cluster_b):
                    graph.edges[a][b] = Relationship(trust=high_trust, affection=0.5, respect=0.5)
                else:
                    graph.edges[a][b] = Relationship(trust=low_trust, affection=0.3, respect=0.3)
    return graph


def _make_uniform_graph(ids: list[str], trust: float = 0.5) -> SocialGraph:
    """Create a social graph with uniform trust."""
    graph = SocialGraph()
    graph.edges = {}
    for a in ids:
        graph.edges[a] = {}
        for b in ids:
            if a != b:
                graph.edges[a][b] = Relationship(trust=trust, affection=0.5, respect=0.5)
    return graph


# ---------------------------------------------------------------------------
# Faction detection
# ---------------------------------------------------------------------------

class TestFactionDetection:
    def test_finds_clusters(self):
        """High-trust clusters should produce factions."""
        ids = ["c-0", "c-1", "c-2", "c-3", "c-4"]
        cluster_a = ["c-0", "c-1", "c-2"]
        cluster_b = ["c-3", "c-4"]
        colonists = _make_colonists(5)
        graph = _make_graph_with_clusters(ids, cluster_a, cluster_b)
        factions = detect_factions(ids, graph, colonists)
        assert len(factions) >= 1
        # At least one cluster should be detected
        sizes = sorted([len(f) for f in factions], reverse=True)
        assert sizes[0] >= 2

    def test_no_factions_with_low_trust(self):
        """Uniform low trust should produce no factions."""
        ids = ["c-0", "c-1", "c-2", "c-3"]
        colonists = _make_colonists(4)
        graph = _make_uniform_graph(ids, trust=0.3)
        factions = detect_factions(ids, graph, colonists)
        assert len(factions) == 0

    def test_single_large_faction(self):
        """All-high trust should produce one faction with everyone."""
        ids = ["c-0", "c-1", "c-2"]
        colonists = _make_colonists(3)
        graph = _make_uniform_graph(ids, trust=0.9)
        factions = detect_factions(ids, graph, colonists)
        assert len(factions) == 1
        assert len(factions[0]) == 3

    def test_empty_ids(self):
        """Empty ID list should produce no factions."""
        colonists = _make_colonists(0)
        graph = SocialGraph()
        factions = detect_factions([], graph, colonists)
        assert factions == []

    def test_minimum_size(self):
        """Single colonist should not form a faction."""
        ids = ["c-0"]
        colonists = _make_colonists(1)
        graph = _make_uniform_graph(ids, trust=0.9)
        factions = detect_factions(ids, graph, colonists)
        assert len(factions) == 0


# ---------------------------------------------------------------------------
# Faction hysteresis
# ---------------------------------------------------------------------------

class TestFactionHysteresis:
    def test_faction_persists_with_overlap(self):
        """Existing faction should continue if membership overlaps."""
        ids = ["c-0", "c-1", "c-2", "c-3"]
        colonists = _make_colonists(4)
        graph = _make_uniform_graph(ids, trust=0.9)
        rng = random.Random(42)
        state = DiplomacyState()

        events = update_factions(state, ids, graph, colonists, year=1, rng=rng)
        assert len(state.active_factions()) >= 1
        faction_id = state.active_factions()[0].id

        events2 = update_factions(state, ids, graph, colonists, year=2, rng=rng)
        # Same faction should persist (not dissolved + recreated)
        active = state.active_factions()
        assert len(active) >= 1
        assert any(f.id == faction_id for f in active)

    def test_faction_dissolves_when_members_leave(self):
        """Faction should dissolve when active members drop out."""
        colonists = _make_colonists(4)
        all_ids = [c.id for c in colonists]
        graph = _make_uniform_graph(all_ids, trust=0.9)
        rng = random.Random(42)
        state = DiplomacyState()

        update_factions(state, all_ids, graph, colonists, year=1, rng=rng)
        assert len(state.active_factions()) >= 1

        # Kill most members — only 1 remains active
        for c in colonists[1:]:
            c.die(2, "test")
        remaining = [c.id for c in colonists if c.is_active()]
        events = update_factions(state, remaining, graph, colonists, year=2, rng=rng)

        dissolved = [e for e in events if e.event_type == "faction_dissolved"]
        assert len(dissolved) >= 1


# ---------------------------------------------------------------------------
# Treaties
# ---------------------------------------------------------------------------

class TestTreaties:
    def test_propose_between_high_trust(self):
        """High-trust pairs should generate treaty proposals."""
        ids = ["c-0", "c-1"]
        graph = _make_uniform_graph(ids, trust=0.85)
        rng = random.Random(42)
        state = DiplomacyState()

        # Run multiple times — probabilistic
        signed = False
        for _ in range(20):
            events = propose_treaties(state, ids, graph, year=5, rng=rng)
            if any(e.event_type == "treaty_signed" for e in events):
                signed = True
                break
        assert signed, "Expected at least one treaty to be signed in 20 attempts"

    def test_no_duplicate_treaty(self):
        """Should not propose duplicate treaty between same pair."""
        ids = ["c-0", "c-1"]
        graph = _make_uniform_graph(ids, trust=0.85)
        rng = random.Random(42)
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=1, duration=10,
        ))

        events = propose_treaties(state, ids, graph, year=5, rng=rng)
        new_treaties = [e for e in events if e.event_type == "treaty_signed"
                        and set(e.details["parties"]) == {"c-0", "c-1"}]
        assert len(new_treaties) == 0

    def test_no_treaty_with_low_trust(self):
        """Low trust pairs should not form treaties."""
        ids = ["c-0", "c-1"]
        graph = _make_uniform_graph(ids, trust=0.3)
        rng = random.Random(42)
        state = DiplomacyState()

        for _ in range(50):
            events = propose_treaties(state, ids, graph, year=5, rng=rng)
            signed = [e for e in events if e.event_type == "treaty_signed"]
            assert len(signed) == 0

    def test_treaty_expires(self):
        """Treaty should expire after its duration."""
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=5, duration=10,
        ))
        assert len(state.active_treaties()) == 1

        events = expire_treaties(state, year=15)
        assert len(events) == 1
        assert events[0].event_type == "treaty_expired"
        assert len(state.active_treaties()) == 0

    def test_treaty_not_expired_early(self):
        """Treaty should remain active before its duration ends."""
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=5, duration=10,
        ))

        events = expire_treaties(state, year=10)
        assert len(events) == 0
        assert len(state.active_treaties()) == 1


# ---------------------------------------------------------------------------
# Betrayals
# ---------------------------------------------------------------------------

class TestBetrayals:
    def test_sabotage_violates_non_aggression(self):
        """Sabotaging a treaty partner should count as betrayal."""
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=1, duration=20,
        ))
        outcomes = [
            ActionOutcome("c-0", "sabotage", target_id="c-1"),
            ActionOutcome("c-1", "farm"),
        ]
        events = check_betrayals(state, outcomes, year=5)
        assert len(events) == 1
        assert events[0].event_type == "betrayal"
        assert events[0].details["violator"] == "c-0"

    def test_sabotage_non_partner_is_not_betrayal(self):
        """Sabotaging someone NOT in the treaty is not a violation."""
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=1, duration=20,
        ))
        outcomes = [
            ActionOutcome("c-0", "sabotage", target_id="c-2"),
            ActionOutcome("c-1", "farm"),
        ]
        events = check_betrayals(state, outcomes, year=5)
        assert len(events) == 0

    def test_hoarding_violates_cooperation(self):
        """Hoarding while partner cooperates violates cooperation treaty."""
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="cooperation", year_signed=1, duration=20,
        ))
        outcomes = [
            ActionOutcome("c-0", "hoard"),
            ActionOutcome("c-1", "cooperate"),
        ]
        events = check_betrayals(state, outcomes, year=5)
        assert len(events) == 1
        assert events[0].details["reason"] == "hoarding_while_partner_cooperates"

    def test_resting_violates_mutual_defense(self):
        """Resting while partner labors violates mutual defense."""
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="mutual_defense", year_signed=1, duration=20,
        ))
        outcomes = [
            ActionOutcome("c-0", "rest"),
            ActionOutcome("c-1", "terraform"),
        ]
        events = check_betrayals(state, outcomes, year=5)
        assert len(events) == 1
        assert events[0].details["reason"] == "resting_during_partner_labor"

    def test_no_betrayal_on_inactive_treaty(self):
        """Inactive treaties should not trigger betrayals."""
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=1, duration=20,
            active=False,
        ))
        outcomes = [
            ActionOutcome("c-0", "sabotage", target_id="c-1"),
            ActionOutcome("c-1", "farm"),
        ]
        events = check_betrayals(state, outcomes, year=5)
        assert len(events) == 0

    def test_mutual_betrayal(self):
        """Both parties sabotaging each other = two betrayals."""
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=1, duration=20,
        ))
        outcomes = [
            ActionOutcome("c-0", "sabotage", target_id="c-1"),
            ActionOutcome("c-1", "sabotage", target_id="c-0"),
        ]
        events = check_betrayals(state, outcomes, year=5)
        assert len(events) == 2


# ---------------------------------------------------------------------------
# Betrayal consequences
# ---------------------------------------------------------------------------

class TestBetrayalConsequences:
    def test_trust_drops_on_betrayal(self):
        """Betrayal should collapse trust between parties."""
        ids = ["c-0", "c-1", "c-2"]
        graph = _make_uniform_graph(ids, trust=0.8)
        rng = random.Random(42)
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=1, duration=20,
        ))
        state.betrayals.append({
            "treaty_id": "treaty-0", "violator_id": "c-0",
            "year": 5, "reason": "sabotage_against_partner",
            "treaty_type": "non_aggression",
        })
        trust_before = graph.get("c-1", "c-0").trust
        apply_betrayal_consequences(state, graph, year=5, rng=rng)
        trust_after = graph.get("c-1", "c-0").trust
        assert trust_after < trust_before


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_diplomacy_state_round_trip(self):
        """DiplomacyState should survive serialization round-trip."""
        state = DiplomacyState()
        state.factions.append(Faction(
            id="faction-0", member_ids=["c-0", "c-1"],
            coherence=0.75, dominant_value="empathy",
            formed_year=3, name="Heart Circle",
        ))
        state.treaties.append(Treaty(
            id="treaty-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=5, duration=10,
        ))
        state.betrayals.append({"treaty_id": "treaty-0", "year": 8})

        d = state.to_dict()
        json_str = json.dumps(d)
        restored = DiplomacyState.from_dict(json.loads(json_str))

        assert len(restored.factions) == 1
        assert restored.factions[0].name == "Heart Circle"
        assert len(restored.treaties) == 1
        assert restored.treaties[0].treaty_type == "non_aggression"
        assert len(restored.betrayals) == 1

    def test_faction_to_dict(self):
        """Faction serialization should be JSON-safe."""
        f = Faction(id="f-0", member_ids=["a", "b"], coherence=0.5,
                    dominant_value="empathy", formed_year=1, name="Test")
        d = f.to_dict()
        assert isinstance(json.dumps(d), str)
        assert "dissolved_year" not in d

    def test_treaty_to_dict(self):
        """Treaty serialization should be JSON-safe."""
        t = Treaty(id="t-0", party_a="a", party_b="b",
                   treaty_type="non_aggression", year_signed=1, duration=5)
        d = t.to_dict()
        assert isinstance(json.dumps(d), str)

    def test_action_outcome_to_dict(self):
        """ActionOutcome serialization."""
        ao = ActionOutcome("c-0", "sabotage", "c-1")
        d = ao.to_dict()
        assert d["target_id"] == "c-1"

        ao2 = ActionOutcome("c-0", "farm")
        d2 = ao2.to_dict()
        assert "target_id" not in d2


# ---------------------------------------------------------------------------
# Integration: tick_diplomacy
# ---------------------------------------------------------------------------

class TestTickDiplomacy:
    def test_tick_returns_events(self):
        """tick_diplomacy should return a list of events."""
        colonists = _make_colonists(4)
        ids = [c.id for c in colonists]
        graph = _make_uniform_graph(ids, trust=0.7)
        rng = random.Random(42)
        state = DiplomacyState()
        outcomes = [ActionOutcome(c.id, "farm") for c in colonists]

        events = tick_diplomacy(state, colonists, outcomes, graph, year=5, rng=rng)
        assert isinstance(events, list)
        for e in events:
            assert isinstance(e, DiplomacyEvent)

    def test_full_cycle(self):
        """Run diplomacy for 20 years and verify invariants."""
        colonists = _make_colonists(6, seed=99)
        ids = [c.id for c in colonists]
        graph = _make_uniform_graph(ids, trust=0.65)
        rng = random.Random(99)
        state = DiplomacyState()

        all_events: list[DiplomacyEvent] = []
        for year in range(1, 21):
            actions = [rng.choice(["farm", "code", "sabotage", "hoard", "cooperate", "rest"])
                       for _ in colonists]
            outcomes = []
            for c, action in zip(colonists, actions):
                target = rng.choice([o.id for o in colonists if o.id != c.id])
                outcomes.append(ActionOutcome(c.id, action,
                                              target_id=target if action == "sabotage" else None))
            events = tick_diplomacy(state, colonists, outcomes, graph, year=year, rng=rng)
            all_events.extend(events)

        # Invariant: no faction member overlaps between active factions
        active_factions = state.active_factions()
        all_members: set[str] = set()
        for f in active_factions:
            overlap = all_members & set(f.member_ids)
            assert len(overlap) == 0, f"Faction member overlap: {overlap}"
            all_members.update(f.member_ids)

        # Invariant: every betrayal references a valid treaty
        treaty_ids = {t.id for t in state.treaties}
        for b in state.betrayals:
            assert b["treaty_id"] in treaty_ids

        # Invariant: history is non-decreasing in year
        years = [h["year"] for h in state.history]
        assert years == sorted(years)


# ---------------------------------------------------------------------------
# Engine integration: 20-year smoke test
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_short_sim_with_diplomacy(self):
        """Run 20-year sim and verify diplomacy data appears in results."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        d = result.to_dict()

        assert d["_meta"]["version"] == "5.0"
        assert "diplomacy" in d
        assert "factions" in d["diplomacy"]
        assert "treaties" in d["diplomacy"]
        assert "betrayals" in d["diplomacy"]

        # Year results should have diplomacy_events
        for yr in d["years"]:
            assert "diplomacy_events" in yr

    def test_full_100_year_sim(self):
        """Run full 100-year sim — must not crash."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=123, total_years=100)
        result = engine.run()
        d = result.to_dict()

        assert len(d["years"]) > 0
        summary = d["summary"]
        assert "total_factions" in summary
        assert "total_treaties" in summary
        assert "total_betrayals" in summary

    def test_diplomacy_produces_events_over_time(self):
        """Over 50 years, some diplomatic activity should emerge."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=77, total_years=50)
        result = engine.run()
        d = result.to_dict()

        total_events = sum(len(yr["diplomacy_events"]) for yr in d["years"])
        assert total_events > 0, "Expected some diplomatic events in 50 years"


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_faction_coherence_bounded(self):
        """Faction coherence should be in [0, 1]."""
        ids = ["c-0", "c-1", "c-2"]
        colonists = _make_colonists(3)
        graph = _make_uniform_graph(ids, trust=0.9)
        rng = random.Random(42)
        state = DiplomacyState()
        update_factions(state, ids, graph, colonists, year=1, rng=rng)

        for f in state.factions:
            assert 0.0 <= f.coherence <= 1.0

    def test_treaty_type_is_valid(self):
        """All treaty types should be from the allowed set."""
        from src.mars100.diplomacy import TREATY_TYPES
        state = DiplomacyState()
        ids = ["c-0", "c-1"]
        graph = _make_uniform_graph(ids, trust=0.9)
        rng = random.Random(42)

        for _ in range(50):
            events = propose_treaties(state, ids, graph, year=5, rng=rng)
            for e in events:
                if e.event_type == "treaty_signed":
                    assert e.details["type"] in TREATY_TYPES

    def test_dissolved_faction_not_active(self):
        """A dissolved faction should not appear in active_factions()."""
        state = DiplomacyState()
        state.factions.append(Faction(
            id="f-0", member_ids=["c-0", "c-1"], coherence=0.7,
            dominant_value="empathy", formed_year=1, dissolved_year=5,
        ))
        assert len(state.active_factions()) == 0

    def test_expired_treaty_not_active(self):
        """An expired treaty should not appear in active_treaties()."""
        state = DiplomacyState()
        state.treaties.append(Treaty(
            id="t-0", party_a="c-0", party_b="c-1",
            treaty_type="non_aggression", year_signed=1, duration=5,
            active=False, year_expired=6,
        ))
        assert len(state.active_treaties()) == 0

    def test_mutual_trust_symmetric_ish(self):
        """Mutual trust should be the average of both directions."""
        ids = ["c-0", "c-1"]
        graph = SocialGraph()
        graph.edges = {
            "c-0": {"c-1": Relationship(trust=0.8, affection=0.5, respect=0.5)},
            "c-1": {"c-0": Relationship(trust=0.6, affection=0.5, respect=0.5)},
        }
        mt = _mutual_trust(graph, "c-0", "c-1")
        assert abs(mt - 0.7) < 1e-6
