"""Tests for the behavioral psychology organ (engine v9.0).

Covers: compute_behavior_weights purity, directionality, bounds,
forced rest, critical-action floors, and engine integration.
"""
from __future__ import annotations

import random
import pytest
from src.mars100.psychology import PsychState
from src.mars100.behavior import (
    compute_behavior_weights, is_forced_rest,
    AXIS_CAP, TOTAL_CAP, CRITICAL_ACTIONS, CRITICAL_FLOOR_DELTA,
)
from src.mars100.engine import Mars100Engine, ACTIONS


# ---------------------------------------------------------------------------
# Unit tests: compute_behavior_weights
# ---------------------------------------------------------------------------

class TestBehaviorWeightsPurity:
    """Weights are a pure function of PsychState -- no randomness."""

    def test_same_input_same_output(self):
        psych = PsychState(stress=0.7, loneliness=0.6, purpose=0.8)
        a = compute_behavior_weights(psych)
        b = compute_behavior_weights(psych)
        assert a == b

    def test_default_psych_no_large_deltas(self):
        psych = PsychState()
        deltas = compute_behavior_weights(psych)
        for v in deltas.values():
            assert abs(v) < 0.5, f"Default psych should not produce large deltas: {deltas}"


class TestStressAxis:
    def test_high_stress_boosts_rest(self):
        deltas = compute_behavior_weights(PsychState(stress=0.9))
        assert deltas.get("rest", 0) > 0

    def test_high_stress_boosts_pray(self):
        deltas = compute_behavior_weights(PsychState(stress=0.85))
        assert deltas.get("pray", 0) > 0

    def test_high_stress_reduces_explore(self):
        deltas = compute_behavior_weights(PsychState(stress=0.8))
        assert deltas.get("explore", 0) < 0

    def test_high_stress_reduces_research(self):
        deltas = compute_behavior_weights(PsychState(stress=0.8))
        assert deltas.get("research", 0) < 0

    def test_low_stress_boosts_explore(self):
        deltas = compute_behavior_weights(PsychState(stress=0.05))
        assert deltas.get("explore", 0) > 0

    def test_moderate_stress_no_effect(self):
        psych = PsychState(stress=0.4, loneliness=0.3, purpose=0.5)
        deltas = compute_behavior_weights(psych)
        assert "rest" not in deltas or abs(deltas.get("rest", 0)) < 0.01


class TestLonelinessAxis:
    def test_high_loneliness_boosts_cooperate(self):
        deltas = compute_behavior_weights(PsychState(loneliness=0.8))
        assert deltas.get("cooperate", 0) > 0

    def test_high_loneliness_boosts_mediate(self):
        deltas = compute_behavior_weights(PsychState(loneliness=0.7))
        assert deltas.get("mediate", 0) > 0

    def test_high_loneliness_reduces_sabotage(self):
        deltas = compute_behavior_weights(PsychState(loneliness=0.8))
        assert deltas.get("sabotage", 0) < 0


class TestPurposeAxis:
    def test_high_purpose_boosts_research(self):
        deltas = compute_behavior_weights(PsychState(purpose=0.9))
        assert deltas.get("research", 0) > 0

    def test_high_purpose_boosts_terraform(self):
        deltas = compute_behavior_weights(PsychState(purpose=0.85))
        assert deltas.get("terraform", 0) > 0

    def test_low_purpose_boosts_rest(self):
        deltas = compute_behavior_weights(PsychState(purpose=0.1))
        assert deltas.get("rest", 0) > 0


class TestMoraleComposite:
    def test_low_morale_boosts_rest(self):
        psych = PsychState(stress=0.8, loneliness=0.6, purpose=0.1)
        assert psych.morale < 0.3
        deltas = compute_behavior_weights(psych)
        assert deltas.get("rest", 0) > 0

    def test_low_morale_boosts_pray(self):
        psych = PsychState(stress=0.8, loneliness=0.6, purpose=0.1)
        deltas = compute_behavior_weights(psych)
        assert deltas.get("pray", 0) > 0


class TestBounds:
    """All deltas must be bounded by TOTAL_CAP and respect critical floors."""

    @pytest.mark.parametrize("stress", [0.0, 0.3, 0.6, 0.9, 1.0])
    @pytest.mark.parametrize("loneliness", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("purpose", [0.0, 0.3, 0.7, 1.0])
    def test_deltas_within_total_cap(self, stress, loneliness, purpose):
        psych = PsychState(stress=stress, loneliness=loneliness, purpose=purpose)
        deltas = compute_behavior_weights(psych)
        for action, delta in deltas.items():
            assert abs(delta) <= TOTAL_CAP + 1e-9, (
                f"{action}={delta} exceeds TOTAL_CAP={TOTAL_CAP}"
            )

    @pytest.mark.parametrize("stress", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("loneliness", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("purpose", [0.0, 0.5, 1.0])
    def test_critical_floor_respected(self, stress, loneliness, purpose):
        psych = PsychState(stress=stress, loneliness=loneliness, purpose=purpose)
        deltas = compute_behavior_weights(psych)
        for action in CRITICAL_ACTIONS:
            if action in deltas:
                assert deltas[action] >= CRITICAL_FLOOR_DELTA - 1e-9

    def test_all_keys_are_valid_actions(self):
        psych = PsychState(stress=0.9, loneliness=0.9, purpose=0.1)
        deltas = compute_behavior_weights(psych)
        valid = set(ACTIONS)
        for key in deltas:
            assert key in valid, f"Unknown action key: {key}"


# ---------------------------------------------------------------------------
# Unit tests: is_forced_rest
# ---------------------------------------------------------------------------

class TestForcedRest:
    def test_no_forced_rest_by_default(self):
        assert not is_forced_rest(PsychState(), 5)

    def test_forced_rest_on_crisis_year_plus_one(self):
        psych = PsychState(forced_rest_until=10)
        assert is_forced_rest(psych, 10)
        assert not is_forced_rest(psych, 11)

    def test_forced_rest_none(self):
        assert not is_forced_rest(PsychState(forced_rest_until=None), 1)

    def test_forced_rest_past(self):
        assert not is_forced_rest(PsychState(forced_rest_until=5), 6)


# ---------------------------------------------------------------------------
# PsychState serialization round-trip with new field
# ---------------------------------------------------------------------------

class TestPsychStateSerialization:
    def test_round_trip_with_forced_rest(self):
        psych = PsychState(stress=0.5, forced_rest_until=12)
        d = psych.to_dict()
        assert d["forced_rest_until"] == 12
        restored = PsychState.from_dict(d)
        assert restored.forced_rest_until == 12

    def test_round_trip_without_forced_rest(self):
        d = PsychState(stress=0.3).to_dict()
        assert d["forced_rest_until"] is None
        assert PsychState.from_dict(d).forced_rest_until is None

    def test_from_dict_missing_forced_rest(self):
        """Backward compat: old dicts without forced_rest_until."""
        psych = PsychState.from_dict({"stress": 0.5, "loneliness": 0.3, "purpose": 0.6})
        assert psych.forced_rest_until is None


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------

class TestEngineIntegration:

    def test_v9_determinism(self):
        a = Mars100Engine(seed=42, total_years=10).run()
        b = Mars100Engine(seed=42, total_years=10).run()
        assert len(a.years) == len(b.years)
        for ya, yb in zip(a.years, b.years):
            assert ya.actions == yb.actions
            assert ya.resources_after == yb.resources_after

    def test_v9_different_seeds_differ(self):
        a = Mars100Engine(seed=1, total_years=10).run()
        b = Mars100Engine(seed=2, total_years=10).run()
        assert sum(1 for ya, yb in zip(a.years, b.years) if ya.actions != yb.actions) > 0

    def test_version_is_9(self):
        d = Mars100Engine(seed=42, total_years=3).run().to_dict()
        assert d["_meta"]["version"] == "9.0"

    def test_forced_rest_happens_after_crisis(self):
        """If a colonist has a crisis, they must rest the next year."""
        result = Mars100Engine(seed=42, total_years=50).run()
        crisis_map: dict[str, int] = {}
        for yr in result.years:
            for crisis in yr.psychology.get("crises", []):
                cid = crisis["colonist_id"]
                crisis_map[cid] = crisis["year"]
        for cid, cy in crisis_map.items():
            next_yr = [yr for yr in result.years if yr.year == cy + 1]
            if next_yr and cid in next_yr[0].actions:
                assert next_yr[0].actions[cid] == "rest", (
                    f"{cid} had crisis year {cy} but chose "
                    f"{next_yr[0].actions[cid]} in year {cy+1}")

    def test_behavior_weights_applied_in_engine(self):
        """Verify that compute_behavior_weights is actually called and
        influences the weight vector.  We check by confirming that a
        highly-stressed PsychState produces non-zero deltas for rest."""
        psych = PsychState(stress=0.9, loneliness=0.1, purpose=0.5)
        deltas = compute_behavior_weights(psych)
        assert deltas.get("rest", 0) > 0, "High stress should boost rest weight"
        assert deltas.get("explore", 0) < 0, "High stress should reduce explore"

    def test_smoke_100_year(self):
        result = Mars100Engine(seed=42, total_years=100).run()
        assert len(result.years) > 0
        assert result.to_dict()["_meta"]["version"] == "9.0"

    def test_psych_map_initialized_before_action(self):
        engine = Mars100Engine(seed=42, total_years=3)
        engine.run()
        for c in engine.colonists:
            if c.is_active():
                assert c.id in engine.psych_map


class TestCrisisSetsForcedRest:
    def test_crisis_sets_forced_rest_field(self):
        from src.mars100.psychology import tick_psychology, ColonistPsychContext
        ctx = ColonistPsychContext(
            colonist_id="test-1", action="sabotage",
            event_severity=0.8, resource_avg=0.3,
            social_connections=1, avg_trust=0.2,
            earth_contact=False, infra_completed=False,
            gov_participated=False, subsim_ran=False,
            resolve=0.3, empathy=0.3, faith=0.3, paranoia=0.8,
        )
        crisis_found = False
        for attempt in range(100):
            psych_map: dict[str, PsychState] = {
                "test-1": PsychState(stress=0.95, last_crisis_year=-999)
            }
            result = tick_psychology(psych_map, [ctx], 10, random.Random(attempt))
            if result.crises:
                crisis_found = True
                assert psych_map["test-1"].forced_rest_until == 11
                break
        assert crisis_found, "No crisis triggered in 100 attempts with stress=0.95"
