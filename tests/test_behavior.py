"""Tests for the behavior organ (v10.0)."""
from __future__ import annotations

import random
import pytest
from src.mars100.behavior import (
    BehaviorProfile, BehaviorTickResult, ContagionDelta,
    compute_action_perturbation, compute_social_contagion,
    update_learned_preferences, compute_risk_tolerance,
    STRESS_WEIGHT_CAP, MORALE_WEIGHT_CAP, PURPOSE_WEIGHT_CAP,
    LEARNED_WEIGHT_CAP, STRESS_CONTAGION_CAP, LONELINESS_CONTAGION_CAP,
    PURPOSE_CONTAGION_CAP, TRUST_THRESHOLD, LEARNING_RATE, PREF_CAP,
    SMALL_PREF_THRESHOLD, ACTION_RESOURCE_MAP,
    STRESS_BOOST_ACTIONS, STRESS_REDUCE_ACTIONS,
)

ACTIONS = ["terraform", "farm", "mediate", "code", "pray",
           "sabotage", "cooperate", "hoard", "explore", "rest", "research"]


# --- BehaviorProfile tests ---

class TestBehaviorProfile:

    def test_defaults(self) -> None:
        p = BehaviorProfile()
        assert p.action_preferences == {}
        assert p.total_actions == 0

    def test_to_dict_roundtrip(self) -> None:
        p = BehaviorProfile(action_preferences={"farm": 0.1, "code": -0.05},
                            total_actions=10)
        d = p.to_dict()
        p2 = BehaviorProfile.from_dict(d)
        assert p2.action_preferences == p.action_preferences
        assert p2.total_actions == p.total_actions

    def test_from_dict_empty(self) -> None:
        p = BehaviorProfile.from_dict({})
        assert p.action_preferences == {}
        assert p.total_actions == 0


# --- ContagionDelta tests ---

class TestContagionDelta:

    def test_defaults(self) -> None:
        cd = ContagionDelta(colonist_id="c1")
        assert cd.stress_delta == 0.0
        assert cd.loneliness_delta == 0.0
        assert cd.purpose_delta == 0.0

    def test_to_dict(self) -> None:
        cd = ContagionDelta("c1", 0.01, -0.02, 0.005)
        d = cd.to_dict()
        assert d["colonist_id"] == "c1"
        assert abs(d["stress_delta"] - 0.01) < 1e-5
        assert abs(d["loneliness_delta"] - (-0.02)) < 1e-5


# --- BehaviorTickResult tests ---

class TestBehaviorTickResult:

    def test_defaults(self) -> None:
        r = BehaviorTickResult()
        assert r.contagion == []
        assert r.perturbations == {}
        assert r.learned_updates == {}

    def test_to_dict(self) -> None:
        r = BehaviorTickResult(
            contagion=[{"colonist_id": "c1", "stress_delta": 0.01}],
            perturbations={"c1": {"farm": 0.1}},
            learned_updates={"c1": {"farm": 0.05}},
        )
        d = r.to_dict()
        assert len(d["contagion"]) == 1
        assert "c1" in d["perturbations"]


# --- compute_action_perturbation tests ---

class TestPerturbation:

    def test_zero_state_zero_profile(self) -> None:
        p = BehaviorProfile()
        deltas = compute_action_perturbation(0.0, 0.5, 0.5, p, ACTIONS)
        for v in deltas.values():
            assert abs(v) < 1e-9

    def test_high_stress_boosts_comfort(self) -> None:
        p = BehaviorProfile()
        deltas = compute_action_perturbation(1.0, 0.5, 0.5, p, ACTIONS)
        for action in STRESS_BOOST_ACTIONS:
            if action in deltas:
                assert deltas[action] > 0

    def test_high_stress_reduces_ambitious(self) -> None:
        p = BehaviorProfile()
        deltas = compute_action_perturbation(1.0, 0.5, 0.5, p, ACTIONS)
        for action in STRESS_REDUCE_ACTIONS:
            if action in deltas:
                assert deltas[action] < 0

    def test_learned_prefs_applied(self) -> None:
        p = BehaviorProfile(action_preferences={"farm": 0.2})
        deltas = compute_action_perturbation(0.0, 0.5, 0.5, p, ACTIONS)
        assert deltas["farm"] == pytest.approx(0.2, abs=1e-6)

    def test_learned_prefs_capped(self) -> None:
        p = BehaviorProfile(action_preferences={"farm": 5.0})
        deltas = compute_action_perturbation(0.0, 0.5, 0.5, p, ACTIONS)
        assert deltas["farm"] <= LEARNED_WEIGHT_CAP + 1e-9

    def test_all_actions_present(self) -> None:
        p = BehaviorProfile()
        deltas = compute_action_perturbation(0.5, 0.5, 0.5, p, ACTIONS)
        for action in ACTIONS:
            assert action in deltas


# --- compute_social_contagion tests ---

class TestSocialContagion:

    def _make_snapshot(self, colonists: dict[str, dict]) -> dict:
        return colonists

    def test_no_neighbors(self) -> None:
        snap = {"c1": {"stress": 0.5, "loneliness": 0.3, "purpose": 0.5}}
        cd = compute_social_contagion("c1", snap, [])
        assert cd.stress_delta == 0.0

    def test_low_trust_ignored(self) -> None:
        snap = {
            "c1": {"stress": 0.1, "loneliness": 0.1, "purpose": 0.5},
            "c2": {"stress": 0.9, "loneliness": 0.9, "purpose": 0.1},
        }
        cd = compute_social_contagion("c1", snap,
                                       [("c2", TRUST_THRESHOLD - 0.01)])
        assert cd.stress_delta == 0.0

    def test_high_trust_propagates_stress(self) -> None:
        snap = {
            "c1": {"stress": 0.1, "loneliness": 0.1, "purpose": 0.5},
            "c2": {"stress": 0.9, "loneliness": 0.1, "purpose": 0.5},
        }
        cd = compute_social_contagion("c1", snap, [("c2", 0.9)])
        assert cd.stress_delta > 0  # c2's high stress should pull c1 up

    def test_contagion_capped(self) -> None:
        snap = {
            "c1": {"stress": 0.0, "loneliness": 0.0, "purpose": 0.0},
            "c2": {"stress": 1.0, "loneliness": 1.0, "purpose": 1.0},
        }
        cd = compute_social_contagion("c1", snap, [("c2", 1.0)])
        assert cd.stress_delta <= STRESS_CONTAGION_CAP + 1e-9
        assert cd.loneliness_delta <= LONELINESS_CONTAGION_CAP + 1e-9
        assert cd.purpose_delta <= PURPOSE_CONTAGION_CAP + 1e-9

    def test_missing_colonist_id(self) -> None:
        cd = compute_social_contagion("missing", {}, [("c2", 0.9)])
        assert cd.stress_delta == 0.0

    def test_simultaneous_symmetry(self) -> None:
        """Two colonists with same trust should get symmetric contagion."""
        snap = {
            "c1": {"stress": 0.0, "loneliness": 0.5, "purpose": 0.5},
            "c2": {"stress": 1.0, "loneliness": 0.5, "purpose": 0.5},
        }
        cd1 = compute_social_contagion("c1", snap, [("c2", 0.8)])
        cd2 = compute_social_contagion("c2", snap, [("c1", 0.8)])
        assert cd1.stress_delta > 0
        assert cd2.stress_delta < 0
        assert abs(cd1.stress_delta + cd2.stress_delta) < 1e-9


# --- update_learned_preferences tests ---

class TestLearnedPreferences:

    def test_positive_reward(self) -> None:
        p = BehaviorProfile()
        result = update_learned_preferences(
            p, "farm", {"food": 0.1, "water": 0.05})
        assert p.action_preferences["farm"] > 0
        assert p.total_actions == 1

    def test_negative_reward(self) -> None:
        p = BehaviorProfile()
        update_learned_preferences(p, "farm", {"food": -0.2})
        assert p.action_preferences["farm"] < 0

    def test_unmapped_action_no_learning(self) -> None:
        p = BehaviorProfile()
        update_learned_preferences(p, "pray", {"food": 0.5})
        assert p.action_preferences.get("pray", 0.0) == 0.0

    def test_preference_capped(self) -> None:
        p = BehaviorProfile(action_preferences={"farm": PREF_CAP - 0.01})
        update_learned_preferences(p, "farm", {"food": 100.0})
        assert p.action_preferences["farm"] <= PREF_CAP

    def test_decay_over_time(self) -> None:
        p = BehaviorProfile(action_preferences={"farm": 0.5, "code": 0.3})
        update_learned_preferences(p, "farm", {"food": 0.0})
        assert p.action_preferences["code"] < 0.3

    def test_small_values_pruned(self) -> None:
        p = BehaviorProfile(action_preferences={
            "farm": 0.5, "code": SMALL_PREF_THRESHOLD * 0.5})
        update_learned_preferences(p, "farm", {"food": 0.0})
        assert "code" not in p.action_preferences

    def test_returns_updated_dict(self) -> None:
        p = BehaviorProfile()
        result = update_learned_preferences(p, "farm", {"food": 0.1})
        assert isinstance(result, dict)
        assert "farm" in result


# --- compute_risk_tolerance tests ---

class TestRiskTolerance:

    def test_neutral_state(self) -> None:
        r = compute_risk_tolerance(0.0, 0.5, 0.5)
        assert 0.0 <= r <= 1.0

    def test_high_stress_lowers_risk(self) -> None:
        r_low = compute_risk_tolerance(0.0, 0.5, 0.5)
        r_high = compute_risk_tolerance(1.0, 0.5, 0.5)
        assert r_high < r_low

    def test_high_purpose_raises_risk(self) -> None:
        r_low = compute_risk_tolerance(0.0, 0.5, 0.0)
        r_high = compute_risk_tolerance(0.0, 0.5, 1.0)
        assert r_high > r_low

    def test_always_bounded(self) -> None:
        for s in [0.0, 0.5, 1.0]:
            for m in [0.0, 0.5, 1.0]:
                for p in [0.0, 0.5, 1.0]:
                    r = compute_risk_tolerance(s, m, p)
                    assert 0.0 <= r <= 1.0


# --- property-based fuzz tests ---

class TestPropertyBased:

    @pytest.mark.parametrize("seed", range(20))
    def test_perturbation_bounded(self, seed: int) -> None:
        rng = random.Random(seed)
        stress = rng.random()
        morale = rng.random()
        purpose = rng.random()
        prefs = {a: rng.uniform(-1, 1) for a in ACTIONS if rng.random() > 0.5}
        p = BehaviorProfile(action_preferences=prefs)
        deltas = compute_action_perturbation(stress, morale, purpose, p, ACTIONS)
        for d in deltas.values():
            assert abs(d) < STRESS_WEIGHT_CAP + MORALE_WEIGHT_CAP + PURPOSE_WEIGHT_CAP + LEARNED_WEIGHT_CAP + 0.01

    @pytest.mark.parametrize("seed", range(10))
    def test_contagion_bounded(self, seed: int) -> None:
        rng = random.Random(seed)
        n = rng.randint(2, 8)
        cids = [f"c{i}" for i in range(n)]
        snap = {cid: {"stress": rng.random(), "loneliness": rng.random(),
                       "purpose": rng.random()} for cid in cids}
        for cid in cids:
            pairs = [(oid, rng.random()) for oid in cids if oid != cid]
            cd = compute_social_contagion(cid, snap, pairs)
            assert abs(cd.stress_delta) <= STRESS_CONTAGION_CAP + 1e-9
            assert abs(cd.loneliness_delta) <= LONELINESS_CONTAGION_CAP + 1e-9
            assert abs(cd.purpose_delta) <= PURPOSE_CONTAGION_CAP + 1e-9
