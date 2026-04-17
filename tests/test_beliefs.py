"""Tests for the belief systems organ (engine v9.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.beliefs import (
    BeliefState, BeliefYearContext, BeliefTickResult,
    MartyrdomEffect, Faction,
    BELIEF_AXES, BELIEF_CAP_DELTA, MARTYRDOM_HORIZON,
    ACTION_BELIEF_WEIGHT, GOV_BELIEF_BIAS,
    FACTION_DISTANCE_THRESHOLD, MIN_FACTION_SIZE,
    _clamp_belief, _cap_delta,
    init_beliefs_from_stats, inherit_beliefs,
    compute_social_influence, compute_experience_shift,
    create_martyrdom_effect, apply_martyrdom,
    detect_factions, compute_polarization,
    compute_belief_action_weights, compute_governance_vote_bias,
    tick_beliefs,
)


# -- BeliefState tests -------------------------------------------------------

class TestBeliefState:
    def test_default_values(self):
        b = BeliefState()
        for axis in BELIEF_AXES:
            assert getattr(b, axis) == 0.0

    def test_round_trip(self):
        b = BeliefState(collectivism=0.5, authority=-0.3,
                        spiritualism=0.8, risk_appetite=-0.7)
        d = b.to_dict()
        b2 = BeliefState.from_dict(d)
        for axis in BELIEF_AXES:
            assert getattr(b2, axis) == pytest.approx(getattr(b, axis), abs=0.001)

    def test_from_dict_clamps(self):
        b = BeliefState.from_dict({"collectivism": 5.0, "authority": -3.0})
        assert b.collectivism == 1.0
        assert b.authority == -1.0

    def test_from_dict_defaults(self):
        b = BeliefState.from_dict({})
        for axis in BELIEF_AXES:
            assert getattr(b, axis) == 0.0

    def test_distance_same(self):
        b = BeliefState(collectivism=0.5, authority=0.3)
        assert b.distance(b) == pytest.approx(0.0)

    def test_distance_symmetry(self):
        a = BeliefState(collectivism=0.5, authority=-0.3)
        b = BeliefState(collectivism=-0.2, authority=0.8)
        assert a.distance(b) == pytest.approx(b.distance(a))

    def test_distance_bounded(self):
        a = BeliefState(collectivism=-1, authority=-1,
                        spiritualism=-1, risk_appetite=-1)
        b = BeliefState(collectivism=1, authority=1,
                        spiritualism=1, risk_appetite=1)
        d = a.distance(b)
        assert 0.0 <= d <= 1.0

    def test_copy_independent(self):
        a = BeliefState(collectivism=0.5)
        b = a.copy()
        b.collectivism = -0.5
        assert a.collectivism == 0.5


# -- Initialization tests ----------------------------------------------------

class TestInitialization:
    def test_init_from_stats_bounded(self):
        rng = random.Random(42)
        for _ in range(100):
            b = init_beliefs_from_stats(
                resolve=rng.random(), empathy=rng.random(),
                faith=rng.random(), paranoia=rng.random(),
                improvisation=rng.random(), hoarding=rng.random(),
                rng=rng,
            )
            for axis in BELIEF_AXES:
                v = getattr(b, axis)
                assert -1.0 <= v <= 1.0, f"{axis}={v} out of bounds"

    def test_init_deterministic(self):
        a = init_beliefs_from_stats(0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                    random.Random(42))
        b = init_beliefs_from_stats(0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                    random.Random(42))
        for axis in BELIEF_AXES:
            assert getattr(a, axis) == getattr(b, axis)

    def test_inherit_beliefs_bounded(self):
        rng = random.Random(42)
        pa = BeliefState(collectivism=0.8, authority=-0.9,
                         spiritualism=0.7, risk_appetite=-0.6)
        pb = BeliefState(collectivism=-0.5, authority=0.3,
                         spiritualism=-0.2, risk_appetite=0.9)
        for _ in range(100):
            child = inherit_beliefs(pa, pb, rng)
            for axis in BELIEF_AXES:
                assert -1.0 <= getattr(child, axis) <= 1.0

    def test_inherit_near_midpoint(self):
        rng = random.Random(42)
        pa = BeliefState(collectivism=0.8, authority=0.8,
                         spiritualism=0.8, risk_appetite=0.8)
        pb = BeliefState(collectivism=-0.8, authority=-0.8,
                         spiritualism=-0.8, risk_appetite=-0.8)
        children = [inherit_beliefs(pa, pb, rng) for _ in range(200)]
        for axis in BELIEF_AXES:
            mean = sum(getattr(c, axis) for c in children) / len(children)
            assert abs(mean) < 0.2, f"{axis} mean={mean} too far from 0"


# -- Social propagation tests ------------------------------------------------

class TestSocialPropagation:
    def _trust_func(self, a, b):
        return 0.6  # uniform trust

    def test_no_self_influence(self):
        snapshot = {"a": BeliefState(collectivism=0.5)}
        deltas = compute_social_influence("a", snapshot, self._trust_func, ["a"])
        for axis in BELIEF_AXES:
            assert deltas[axis] == pytest.approx(0.0)

    def test_pulled_toward_peers(self):
        snapshot = {
            "a": BeliefState(collectivism=-0.5),
            "b": BeliefState(collectivism=0.5),
            "c": BeliefState(collectivism=0.5),
        }
        deltas = compute_social_influence(
            "a", snapshot, self._trust_func, ["a", "b", "c"])
        assert deltas["collectivism"] > 0  # pulled positive

    def test_deltas_capped(self):
        snapshot = {
            "a": BeliefState(collectivism=-1.0),
            "b": BeliefState(collectivism=1.0),
        }
        high_trust = lambda a, b: 1.0
        deltas = compute_social_influence(
            "a", snapshot, high_trust, ["a", "b"])
        for axis in BELIEF_AXES:
            assert abs(deltas[axis]) <= BELIEF_CAP_DELTA


# -- Experience shift tests --------------------------------------------------

class TestExperienceShift:
    def test_scarcity_increases_collectivism(self):
        shifts = compute_experience_shift(
            "dust_storm", 0.3, 0.1, "farm", 0, False)
        assert shifts["collectivism"] > 0

    def test_abundance_decreases_collectivism(self):
        shifts = compute_experience_shift(
            "calm", 0.1, 0.9, "farm", 0, False)
        assert shifts["collectivism"] < 0

    def test_severe_event_caution(self):
        shifts = compute_experience_shift(
            "solar_flare", 0.9, 0.5, "rest", 0, False)
        assert shifts["risk_appetite"] < 0

    def test_death_spiritualism(self):
        shifts = compute_experience_shift(
            "none", 0.0, 0.5, "rest", 2, False)
        assert shifts["spiritualism"] > 0

    def test_all_deltas_capped(self):
        shifts = compute_experience_shift(
            "apocalypse", 1.0, 0.0, "sabotage", 10, True)
        for axis in BELIEF_AXES:
            assert abs(shifts[axis]) <= BELIEF_CAP_DELTA


# -- Martyrdom tests ---------------------------------------------------------

class TestMartyrdom:
    def test_create_martyrdom(self):
        beliefs = {"dead": BeliefState(collectivism=0.8, spiritualism=0.9)}
        effect = create_martyrdom_effect("dead", beliefs, ["a", "b"])
        assert effect is not None
        assert effect.source_id == "dead"
        assert "a" in effect.affected_ids

    def test_create_martyrdom_missing(self):
        effect = create_martyrdom_effect("dead", {}, ["a"])
        assert effect is None

    def test_strength_decays(self):
        effect = MartyrdomEffect(
            source_id="x", belief_snapshot={}, year_of_death=10)
        s10 = effect.current_strength(10)
        s11 = effect.current_strength(11)
        s12 = effect.current_strength(12)
        assert s10 > s11 > s12

    def test_strength_zero_after_horizon(self):
        effect = MartyrdomEffect(
            source_id="x", belief_snapshot={}, year_of_death=10)
        assert effect.current_strength(10 + MARTYRDOM_HORIZON + 1) == 0.0

    def test_apply_martyrdom_shifts(self):
        beliefs = {
            "a": BeliefState(collectivism=-0.5),
        }
        effect = MartyrdomEffect(
            source_id="dead",
            belief_snapshot={"collectivism": 0.8, "authority": 0.0,
                             "spiritualism": 0.0, "risk_appetite": 0.0},
            year_of_death=5,
            affected_ids=["a"],
        )
        before = beliefs["a"].collectivism
        apply_martyrdom(beliefs, [effect], 5)
        after = beliefs["a"].collectivism
        assert after > before  # shifted toward martyr's 0.8


# -- Faction detection tests -------------------------------------------------

class TestFactions:
    def test_identical_beliefs_one_faction(self):
        beliefs = {
            "a": BeliefState(collectivism=0.5, authority=0.5),
            "b": BeliefState(collectivism=0.5, authority=0.5),
            "c": BeliefState(collectivism=0.5, authority=0.5),
        }
        factions = detect_factions(beliefs, ["a", "b", "c"])
        assert len(factions) == 1
        assert set(factions[0].member_ids) == {"a", "b", "c"}

    def test_opposite_beliefs_two_factions(self):
        beliefs = {
            "a": BeliefState(collectivism=1.0, authority=1.0,
                             spiritualism=1.0, risk_appetite=1.0),
            "b": BeliefState(collectivism=0.9, authority=0.9,
                             spiritualism=0.9, risk_appetite=0.9),
            "c": BeliefState(collectivism=-1.0, authority=-1.0,
                             spiritualism=-1.0, risk_appetite=-1.0),
            "d": BeliefState(collectivism=-0.9, authority=-0.9,
                             spiritualism=-0.9, risk_appetite=-0.9),
        }
        factions = detect_factions(beliefs, ["a", "b", "c", "d"])
        assert len(factions) == 2

    def test_min_faction_size(self):
        beliefs = {
            "loner": BeliefState(collectivism=1.0),
            "a": BeliefState(collectivism=-1.0),
            "b": BeliefState(collectivism=-0.9),
        }
        factions = detect_factions(beliefs, ["loner", "a", "b"])
        # Loner shouldn't form their own faction
        for f in factions:
            assert len(f.member_ids) >= MIN_FACTION_SIZE

    def test_empty_colony(self):
        assert detect_factions({}, []) == []

    def test_faction_cohesion_bounded(self):
        beliefs = {
            f"c-{i}": BeliefState(collectivism=0.5 + i * 0.01)
            for i in range(5)
        }
        factions = detect_factions(beliefs, list(beliefs.keys()))
        for f in factions:
            assert 0.0 <= f.cohesion <= 1.0


# -- Polarization tests ------------------------------------------------------

class TestPolarization:
    def test_identical_zero_polarization(self):
        beliefs = {
            "a": BeliefState(), "b": BeliefState(), "c": BeliefState()
        }
        assert compute_polarization(beliefs, ["a", "b", "c"]) == pytest.approx(0.0)

    def test_opposing_high_polarization(self):
        beliefs = {
            "a": BeliefState(collectivism=1, authority=1,
                             spiritualism=1, risk_appetite=1),
            "b": BeliefState(collectivism=-1, authority=-1,
                             spiritualism=-1, risk_appetite=-1),
        }
        p = compute_polarization(beliefs, ["a", "b"])
        assert p > 0.5

    def test_bounded(self):
        rng = random.Random(42)
        beliefs = {
            f"c-{i}": BeliefState(
                collectivism=rng.uniform(-1, 1),
                authority=rng.uniform(-1, 1),
                spiritualism=rng.uniform(-1, 1),
                risk_appetite=rng.uniform(-1, 1),
            )
            for i in range(20)
        }
        p = compute_polarization(beliefs, list(beliefs.keys()))
        assert 0.0 <= p <= 1.0


# -- Action weight tests -----------------------------------------------------

class TestActionWeights:
    def test_neutral_beliefs_no_effect(self):
        w = compute_belief_action_weights(BeliefState())
        total = sum(abs(v) for v in w.values())
        assert total == pytest.approx(0.0, abs=0.01)

    def test_collectivist_boosts_cooperate(self):
        w = compute_belief_action_weights(
            BeliefState(collectivism=1.0))
        assert w.get("cooperate", 0) > 0
        assert w.get("hoard", 0) < 0

    def test_total_bounded(self):
        extremes = [
            BeliefState(collectivism=1, authority=1,
                        spiritualism=1, risk_appetite=1),
            BeliefState(collectivism=-1, authority=-1,
                        spiritualism=-1, risk_appetite=-1),
        ]
        for b in extremes:
            w = compute_belief_action_weights(b)
            total = sum(abs(v) for v in w.values())
            assert total <= ACTION_BELIEF_WEIGHT + 0.01


# -- Governance vote bias tests ----------------------------------------------

class TestGovernanceVoteBias:
    def test_bounded(self):
        rng = random.Random(42)
        from src.mars100.governance import GOVERNANCE_TYPES
        for _ in range(100):
            b = BeliefState(
                collectivism=rng.uniform(-1, 1),
                authority=rng.uniform(-1, 1),
                spiritualism=rng.uniform(-1, 1),
                risk_appetite=rng.uniform(-1, 1),
            )
            for gov in GOVERNANCE_TYPES:
                bias = compute_governance_vote_bias(b, gov)
                assert -GOV_BELIEF_BIAS <= bias <= GOV_BELIEF_BIAS

    def test_authoritarian_favors_dictator(self):
        b = BeliefState(authority=1.0)
        assert compute_governance_vote_bias(b, "dictator") > 0
        assert compute_governance_vote_bias(b, "anarchy") < 0


# -- tick_beliefs integration tests ------------------------------------------

class TestTickBeliefs:
    def _make_contexts(self, ids, action="farm"):
        return [
            BeliefYearContext(
                colonist_id=cid, action=action, event_type="calm",
                event_severity=0.1, resource_avg=0.6,
                death_count=0, gov_changed=False,
            )
            for cid in ids
        ]

    def _trust_func(self, a, b):
        return 0.5

    def test_basic_tick(self):
        beliefs = {
            "a": BeliefState(collectivism=0.3),
            "b": BeliefState(collectivism=-0.3),
        }
        ctxs = self._make_contexts(["a", "b"])
        result = tick_beliefs(
            beliefs, ctxs, self._trust_func, ["a", "b"],
            year=5, martyrdom_effects=[], rng=random.Random(42))
        assert isinstance(result, BeliefTickResult)
        assert "a" in result.snapshots
        assert "b" in result.snapshots

    def test_beliefs_bounded_after_tick(self):
        rng = random.Random(42)
        beliefs = {
            f"c-{i}": BeliefState(
                collectivism=rng.uniform(-1, 1),
                authority=rng.uniform(-1, 1),
                spiritualism=rng.uniform(-1, 1),
                risk_appetite=rng.uniform(-1, 1),
            )
            for i in range(10)
        }
        ids = list(beliefs.keys())
        ctxs = self._make_contexts(ids)
        for year in range(50):
            tick_beliefs(beliefs, ctxs, self._trust_func, ids,
                         year=year, martyrdom_effects=[], rng=rng)
        for cid in ids:
            for axis in BELIEF_AXES:
                v = getattr(beliefs[cid], axis)
                assert -1.0 <= v <= 1.0, f"{cid}.{axis}={v}"

    def test_synchronous_order_independence(self):
        """Same beliefs and contexts should produce same result regardless of order."""
        beliefs_a = {
            "x": BeliefState(collectivism=0.5),
            "y": BeliefState(collectivism=-0.5),
        }
        beliefs_b = {
            "x": BeliefState(collectivism=0.5),
            "y": BeliefState(collectivism=-0.5),
        }
        ctxs_fwd = self._make_contexts(["x", "y"])
        ctxs_rev = list(reversed(self._make_contexts(["x", "y"])))
        tick_beliefs(beliefs_a, ctxs_fwd, self._trust_func, ["x", "y"],
                     year=1, martyrdom_effects=[], rng=random.Random(42))
        tick_beliefs(beliefs_b, ctxs_rev, self._trust_func, ["x", "y"],
                     year=1, martyrdom_effects=[], rng=random.Random(42))
        for axis in BELIEF_AXES:
            assert getattr(beliefs_a["x"], axis) == pytest.approx(
                getattr(beliefs_b["x"], axis), abs=0.001)

    def test_missing_colonist_gets_default(self):
        beliefs: dict[str, BeliefState] = {}
        ctxs = self._make_contexts(["new"])
        tick_beliefs(beliefs, ctxs, self._trust_func, ["new"],
                     year=1, martyrdom_effects=[], rng=random.Random(42))
        assert "new" in beliefs

    def test_martyrdom_pruning(self):
        beliefs = {"a": BeliefState()}
        effects = [
            MartyrdomEffect(
                source_id="dead", belief_snapshot=BeliefState().to_dict(),
                year_of_death=0, affected_ids=["a"]),
        ]
        ctxs = self._make_contexts(["a"])
        tick_beliefs(beliefs, ctxs, self._trust_func, ["a"],
                     year=MARTYRDOM_HORIZON + 2,
                     martyrdom_effects=effects, rng=random.Random(42))
        assert len(effects) == 0  # expired, should be pruned

    def test_factions_in_result(self):
        beliefs = {
            "a": BeliefState(collectivism=0.9, authority=0.9),
            "b": BeliefState(collectivism=0.8, authority=0.8),
            "c": BeliefState(collectivism=-0.9, authority=-0.9),
            "d": BeliefState(collectivism=-0.8, authority=-0.8),
        }
        ids = list(beliefs.keys())
        ctxs = self._make_contexts(ids)
        result = tick_beliefs(
            beliefs, ctxs, self._trust_func, ids,
            year=1, martyrdom_effects=[], rng=random.Random(42))
        assert result.faction_count >= 1
        assert len(result.factions) == result.faction_count


# -- Psychology action perturbation tests ------------------------------------

class TestPsychActionWeights:
    def test_import(self):
        from src.mars100.psychology import compute_psych_action_weights, PsychState
        w = compute_psych_action_weights(PsychState())
        assert isinstance(w, dict)

    def test_calm_colonist_no_effect(self):
        from src.mars100.psychology import compute_psych_action_weights, PsychState
        w = compute_psych_action_weights(PsychState())
        total = sum(abs(v) for v in w.values())
        assert total == pytest.approx(0.0, abs=0.01)

    def test_stressed_prefers_rest(self):
        from src.mars100.psychology import compute_psych_action_weights, PsychState
        p = PsychState(stress=0.9, loneliness=0.5, purpose=0.3)
        w = compute_psych_action_weights(p)
        assert w.get("rest", 0) > 0
        assert w.get("explore", 0) < 0

    def test_driven_prefers_research(self):
        from src.mars100.psychology import compute_psych_action_weights, PsychState
        p = PsychState(stress=0.1, loneliness=0.1, purpose=0.9)
        w = compute_psych_action_weights(p)
        assert w.get("research", 0) > 0

    def test_total_bounded(self):
        from src.mars100.psychology import (
            compute_psych_action_weights, PsychState, PSYCH_ACTION_WEIGHT)
        p = PsychState(stress=1.0, loneliness=1.0, purpose=0.0)
        w = compute_psych_action_weights(p)
        total = sum(abs(v) for v in w.values())
        assert total <= PSYCH_ACTION_WEIGHT + 0.01
