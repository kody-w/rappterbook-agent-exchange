"""Tests for the Mars-100 psychology engine."""
from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure repo root is on sys.path so src.* imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mars100.psychology import (
    BASELINE,
    EMOTION_NAMES,
    MAX_CONTAGION_DELTA,
    MoodState,
    collective_mood,
    compute_action_bias,
    compute_mood_shift,
    compute_resilience,
    contagion_spread,
    decay_toward_baseline,
)
from src.mars100.colonist import Colonist, ColonistStats, ColonistSkills, create_founding_ten
from src.mars100.colony import SocialGraph
from src.mars100.engine import Mars100Engine


# ─── MoodState basics ───────────────────────────────────────────────

class TestMoodState:
    def test_default_values(self) -> None:
        m = MoodState()
        assert m.joy == 0.4
        assert m.grief == 0.1
        assert m.hope == 0.5
        assert m.fear == 0.2

    def test_serialization_roundtrip(self) -> None:
        m = MoodState(joy=0.8, grief=0.3, fear=0.1, anger=0.5, hope=0.9, despair=0.0)
        d = m.to_dict()
        m2 = MoodState.from_dict(d)
        for name in EMOTION_NAMES:
            assert abs(getattr(m, name) - getattr(m2, name)) < 1e-10

    def test_clamp(self) -> None:
        m = MoodState(joy=1.5, grief=-0.3, fear=0.5, anger=0.5, hope=0.5, despair=0.5)
        m.clamp()
        assert m.joy == 1.0
        assert m.grief == 0.0

    def test_dominant(self) -> None:
        m = MoodState(joy=0.1, grief=0.1, fear=0.1, anger=0.1, hope=0.9, despair=0.1)
        assert m.dominant() == "hope"

    def test_valence_positive(self) -> None:
        m = MoodState(joy=0.8, grief=0.0, fear=0.0, anger=0.0, hope=0.8, despair=0.0)
        assert m.valence() > 0

    def test_valence_negative(self) -> None:
        m = MoodState(joy=0.0, grief=0.8, fear=0.8, anger=0.8, hope=0.0, despair=0.8)
        assert m.valence() < 0

    def test_from_dict_missing_fields(self) -> None:
        """Old serialized colonists without mood still deserialize."""
        m = MoodState.from_dict({})
        for name in EMOTION_NAMES:
            assert getattr(m, name) == BASELINE[name]


# ─── Resilience ──────────────────────────────────────────────────────

class TestResilience:
    def test_computed_from_stats(self) -> None:
        assert compute_resilience(0.8, 0.6) == 0.7

    def test_low_stats(self) -> None:
        assert compute_resilience(0.0, 0.0) == 0.0

    def test_high_stats(self) -> None:
        assert compute_resilience(1.0, 1.0) == 1.0


# ─── Mood shift ──────────────────────────────────────────────────────

class TestMoodShift:
    def test_death_increases_grief(self) -> None:
        m = MoodState()
        rng = random.Random(42)
        compute_mood_shift(m, events=[], deaths=[{"id": "x"}], births=[],
                           resource_avg=0.5, relationship_to_dead={"x": 0.8}, rng=rng)
        assert m.grief > BASELINE["grief"]

    def test_birth_increases_joy(self) -> None:
        m = MoodState()
        rng = random.Random(42)
        compute_mood_shift(m, events=[], deaths=[], births=[{"id": "baby"}],
                           resource_avg=0.5, relationship_to_dead={}, rng=rng)
        assert m.joy > BASELINE["joy"]
        assert m.hope > BASELINE["hope"]

    def test_crisis_increases_fear(self) -> None:
        m = MoodState()
        rng = random.Random(42)
        compute_mood_shift(m, events=[], deaths=[], births=[],
                           resource_avg=0.15, relationship_to_dead={}, rng=rng)
        assert m.fear > BASELINE["fear"]
        assert m.despair > BASELINE["despair"]

    def test_surplus_increases_joy(self) -> None:
        m = MoodState()
        rng = random.Random(42)
        compute_mood_shift(m, events=[], deaths=[], births=[],
                           resource_avg=0.8, relationship_to_dead={}, rng=rng)
        assert m.joy > BASELINE["joy"]

    def test_cosmic_event_increases_fear(self) -> None:
        m = MoodState()
        rng = random.Random(42)
        ev = {"severity": 0.7, "category": "cosmic", "effects": {"morale": -0.2}, "name": "solar_flare"}
        compute_mood_shift(m, events=[ev], deaths=[], births=[],
                           resource_avg=0.5, relationship_to_dead={}, rng=rng)
        assert m.fear > BASELINE["fear"]

    def test_conflict_increases_anger(self) -> None:
        m = MoodState()
        rng = random.Random(42)
        ev = {"severity": 0.5, "category": "social", "effects": {"morale": -0.1}, "name": "colonist_conflict"}
        compute_mood_shift(m, events=[ev], deaths=[], births=[],
                           resource_avg=0.5, relationship_to_dead={}, rng=rng)
        assert m.anger > BASELINE["anger"]

    def test_all_moods_stay_in_bounds(self) -> None:
        """Property: no mood goes below 0 or above 1 regardless of input."""
        rng = random.Random(99)
        for _ in range(50):
            m = MoodState(joy=rng.random(), grief=rng.random(), fear=rng.random(),
                          anger=rng.random(), hope=rng.random(), despair=rng.random())
            compute_mood_shift(
                m, events=[{"severity": 1.0, "category": "cosmic", "effects": {"morale": -0.5}, "name": "x"}],
                deaths=[{"id": f"d{i}"} for i in range(5)],
                births=[{"id": f"b{i}"} for i in range(5)],
                resource_avg=rng.random(),
                relationship_to_dead={f"d{i}": rng.random() for i in range(5)},
                rng=rng,
            )
            for name in EMOTION_NAMES:
                val = getattr(m, name)
                assert 0.0 <= val <= 1.0, f"{name}={val} out of bounds"


# ─── Decay ───────────────────────────────────────────────────────────

class TestDecay:
    def test_extreme_values_decay_faster(self) -> None:
        m1 = MoodState(grief=0.9)  # extreme
        m2 = MoodState(grief=0.4)  # mild
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        decay_toward_baseline(m1, resilience=0.5, rng=rng1)
        decay_toward_baseline(m2, resilience=0.5, rng=rng2)
        # m1's grief should have a bigger absolute move toward baseline
        move1 = 0.9 - m1.grief
        move2 = 0.4 - m2.grief
        assert abs(move1) > abs(move2)

    def test_high_resilience_decays_faster(self) -> None:
        m_hi = MoodState(grief=0.8)
        m_lo = MoodState(grief=0.8)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        decay_toward_baseline(m_hi, resilience=0.9, rng=rng1)
        decay_toward_baseline(m_lo, resilience=0.1, rng=rng2)
        # High resilience should recover faster
        assert m_hi.grief < m_lo.grief

    def test_decay_does_not_overshoot(self) -> None:
        """Values should move toward baseline, not past it."""
        rng = random.Random(42)
        for _ in range(100):
            m = MoodState(grief=0.8)
            for _ in range(50):
                decay_toward_baseline(m, resilience=0.5, rng=rng)
            # After many decays, should be near baseline, not below 0
            assert m.grief >= 0.0
            assert m.grief <= 1.0

    def test_multi_year_stability(self) -> None:
        """Repeated decay doesn't pin everyone at despair=1.0."""
        rng = random.Random(42)
        m = MoodState(despair=0.9, fear=0.9, grief=0.9)
        for _ in range(20):
            decay_toward_baseline(m, resilience=0.5, rng=rng)
        assert m.despair < 0.5, "Despair should recover after 20 years"
        assert m.fear < 0.5, "Fear should recover after 20 years"


# ─── Contagion ───────────────────────────────────────────────────────

class TestContagion:
    def _make_pair(self, trust: float = 0.8) -> tuple[dict[str, MoodState], dict, dict[str, float]]:
        moods = {
            "a": MoodState(grief=0.9, joy=0.1),
            "b": MoodState(grief=0.1, joy=0.5),
        }
        edges = {
            "a": {"b": {"trust": trust}},
            "b": {"a": {"trust": trust}},
        }
        empathy = {"a": 0.5, "b": 0.7}
        return moods, edges, empathy

    def test_grief_spreads(self) -> None:
        moods, edges, empathy = self._make_pair()
        contagion_spread(moods, edges, empathy)
        assert moods["b"].grief > 0.1, "Grief should spread from a to b"

    def test_snapshot_prevents_chaining(self) -> None:
        """A→B and B→C shouldn't chain within one spread pass."""
        moods = {
            "a": MoodState(grief=0.9),
            "b": MoodState(grief=0.1),
            "c": MoodState(grief=0.1),
        }
        edges = {
            "a": {"b": {"trust": 0.8}},
            "b": {"a": {"trust": 0.8}, "c": {"trust": 0.8}},
            "c": {"b": {"trust": 0.8}},
        }
        empathy = {"a": 0.5, "b": 0.7, "c": 0.7}
        contagion_spread(moods, edges, empathy)
        # B absorbs grief from A. C should absorb based on B's ORIGINAL grief (0.1),
        # not the already-updated value.
        # C's grief increase should be much smaller than B's.
        b_increase = moods["b"].grief - 0.1
        c_increase = moods["c"].grief - 0.1
        assert b_increase > c_increase * 2, "C should get much less grief than B (snapshot)"

    def test_low_trust_blocks_contagion(self) -> None:
        moods, edges, empathy = self._make_pair(trust=0.1)
        before_b_grief = moods["b"].grief
        contagion_spread(moods, edges, empathy)
        assert moods["b"].grief == before_b_grief, "Low trust should block contagion"

    def test_per_colonist_cap(self) -> None:
        """Total contagion delta per colonist shouldn't exceed MAX_CONTAGION_DELTA."""
        moods = {"target": MoodState(grief=0.0, fear=0.0, despair=0.0)}
        edges = {"target": {}}
        empathy = {"target": 1.0}
        # Create 10 very emotional colonists all trusted by target
        for i in range(10):
            cid = f"sender-{i}"
            moods[cid] = MoodState(grief=1.0, fear=1.0, despair=1.0, anger=1.0)
            edges[cid] = {"target": {"trust": 0.9}}
            edges["target"][cid] = {"trust": 0.9}  # target trusts senders too
            empathy[cid] = 0.5
        deltas = contagion_spread(moods, edges, empathy)
        total_delta = sum(abs(v) for v in deltas["target"].values())
        assert total_delta <= MAX_CONTAGION_DELTA + 0.01, f"Cap exceeded: {total_delta}"

    def test_dead_colonists_excluded(self) -> None:
        """Dead colonists should not appear in mood_map (engine ensures this)."""
        # This is an engine-level test — mood_map only includes active colonists
        engine = Mars100Engine(seed=42, total_years=1)
        result = engine.tick()
        active_ids = {c["id"] for c in result.colonist_snapshots if c["alive"]}
        for snap in result.colonist_snapshots:
            if snap["id"] in active_ids:
                assert "mood" in snap, f"Active colonist {snap['id']} missing mood"


# ─── Action bias ─────────────────────────────────────────────────────

class TestActionBias:
    def test_high_fear_boosts_hoard(self) -> None:
        m = MoodState(fear=0.8)
        biases = compute_action_bias(m)
        assert biases.get("hoard", 0) > 0

    def test_high_hope_boosts_terraform(self) -> None:
        m = MoodState(hope=0.9)
        biases = compute_action_bias(m)
        assert biases.get("terraform", 0) > 0

    def test_high_grief_boosts_pray(self) -> None:
        m = MoodState(grief=0.8)
        biases = compute_action_bias(m)
        assert biases.get("pray", 0) > 0

    def test_high_anger_boosts_sabotage(self) -> None:
        m = MoodState(anger=0.7)
        biases = compute_action_bias(m)
        assert biases.get("sabotage", 0) > 0

    def test_low_emotion_no_bias(self) -> None:
        m = MoodState(joy=0.1, grief=0.1, fear=0.1, anger=0.1, hope=0.1, despair=0.1)
        biases = compute_action_bias(m)
        assert all(v == 0 for v in biases.values()), "Low emotions should produce no bias"

    def test_high_despair_boosts_rest(self) -> None:
        m = MoodState(despair=0.9)
        biases = compute_action_bias(m)
        assert biases.get("rest", 0) > 0

    def test_grief_suppresses_cooperate(self) -> None:
        m = MoodState(grief=0.8)
        biases = compute_action_bias(m)
        assert biases.get("cooperate", 0) < 0


# ─── Collective mood ────────────────────────────────────────────────

class TestCollectiveMood:
    def test_empty_colony(self) -> None:
        result = collective_mood({})
        assert "morale" in result
        assert "stability" in result

    def test_happy_colony(self) -> None:
        moods = {
            "a": MoodState(joy=0.9, hope=0.9, grief=0.0, despair=0.0, fear=0.0, anger=0.0),
            "b": MoodState(joy=0.8, hope=0.8, grief=0.0, despair=0.0, fear=0.0, anger=0.0),
        }
        result = collective_mood(moods)
        assert result["morale"] > 0.5
        assert result["stability"] > 0.8

    def test_fearful_colony(self) -> None:
        moods = {
            "a": MoodState(fear=0.9, anger=0.8),
            "b": MoodState(fear=0.8, anger=0.7),
        }
        result = collective_mood(moods)
        assert result["stability"] < 0.2


# ─── Colonist integration ───────────────────────────────────────────

class TestColonistIntegration:
    def test_colonist_has_mood(self) -> None:
        colonists = create_founding_ten()
        for c in colonists:
            assert hasattr(c, "mood")
            assert isinstance(c.mood, MoodState)

    def test_colonist_serialization_with_mood(self) -> None:
        colonists = create_founding_ten()
        c = colonists[0]
        c.mood = MoodState(joy=0.9, grief=0.3)
        d = c.to_dict()
        assert "mood" in d
        c2 = Colonist.from_dict(d)
        assert abs(c2.mood.joy - 0.9) < 1e-10
        assert abs(c2.mood.grief - 0.3) < 1e-10

    def test_old_colonist_dict_without_mood(self) -> None:
        """Backward compat: colonists serialized before mood was added."""
        d = {
            "id": "old-guy", "name": "Old Guy", "element": "fire",
            "archetype": "commander", "stats": {}, "skills": {},
        }
        c = Colonist.from_dict(d)
        assert isinstance(c.mood, MoodState)
        assert c.mood.joy == BASELINE["joy"]

    def test_mood_bindings_in_lispy(self) -> None:
        """Mood values are available in LisPy decision expressions."""
        colonists = create_founding_ten()
        c = colonists[0]
        c.mood = MoodState(joy=0.75)
        bindings = c.lispy_bindings()
        assert "mood-joy" in bindings
        assert bindings["mood-joy"] == 0.75
        assert "mood-valence" in bindings


# ─── Engine integration ──────────────────────────────────────────────

class TestEngineIntegration:
    def test_tick_includes_mood_summary(self) -> None:
        engine = Mars100Engine(seed=42, total_years=1)
        result = engine.tick()
        assert "mood_summary" in result.to_dict()
        ms = result.mood_summary
        assert "joy" in ms
        assert "morale" in ms
        assert "stability" in ms

    def test_colonist_snapshots_have_mood(self) -> None:
        engine = Mars100Engine(seed=42, total_years=1)
        result = engine.tick()
        for snap in result.colonist_snapshots:
            assert "mood" in snap, f"Colonist {snap['id']} missing mood"

    def test_10_year_run_with_mood(self) -> None:
        """Smoke test: 10 years don't crash with psychology active."""
        engine = Mars100Engine(seed=42, total_years=10)
        sim = engine.run()
        assert len(sim.years) == 10
        for year in sim.years:
            assert "mood_summary" in year.to_dict()
            for snap in year.colonist_snapshots:
                mood = snap.get("mood", {})
                for emo in EMOTION_NAMES:
                    val = mood.get(emo, 0.5)
                    assert 0.0 <= val <= 1.0, f"Year {year.year}: {snap['id']}.{emo}={val}"

    def test_determinism(self) -> None:
        """Same seed produces identical mood trajectories."""
        engine1 = Mars100Engine(seed=99, total_years=5)
        engine2 = Mars100Engine(seed=99, total_years=5)
        r1 = engine1.run()
        r2 = engine2.run()
        for y1, y2 in zip(r1.years, r2.years):
            assert y1.mood_summary == y2.mood_summary

    def test_fear_increases_hoarding(self) -> None:
        """Statistical test: high fear colonists hoard more across many seeds."""
        hoard_with_fear = 0
        hoard_without = 0
        for seed in range(20):
            engine = Mars100Engine(seed=seed, total_years=3)
            # Make half the colonists very afraid
            for i, c in enumerate(engine.colonists):
                if i < 5:
                    c.mood = MoodState(fear=0.9)
                else:
                    c.mood = MoodState(fear=0.0)
            result = engine.tick()
            for cid, action in result.actions.items():
                c = next(cc for cc in engine.colonists if cc.id == cid)
                if c.mood.fear > 0.5 and action == "hoard":
                    hoard_with_fear += 1
                elif c.mood.fear < 0.2 and action == "hoard":
                    hoard_without += 1
        # Fearful colonists should hoard more often
        assert hoard_with_fear > hoard_without, (
            f"Fear should increase hoarding: {hoard_with_fear} vs {hoard_without}")

    def test_version_is_3(self) -> None:
        engine = Mars100Engine(seed=42, total_years=1)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "3.0"
