"""Tests for the psychology organ (engine v8.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.psychology import (
    PsychState, CrisisEvent, PsychTickResult,
    ColonistPsychContext, tick_psychology,
    compute_stress_delta, compute_loneliness_delta,
    compute_purpose_delta, check_crisis,
    compute_colony_morale, compute_bottom_quartile_morale,
    death_rate_modifier,
    STRESS_CAP_DELTA, LONELINESS_CAP_DELTA, PURPOSE_CAP_DELTA,
    CRISIS_THRESHOLD, CRISIS_COOLDOWN,
    MORALE_DEATH_THRESHOLD, MORALE_DEATH_MULTIPLIER,
    _clamp, _cap_delta,
)


def _make_context(
    cid="c-0", action="farm", event_severity=0.3,
    resource_avg=0.6, social_connections=4, avg_trust=0.5,
    earth_contact=False, infra_completed=False,
    gov_participated=False, subsim_ran=False,
    resolve=0.5, empathy=0.5, faith=0.3, paranoia=0.3,
):
    return ColonistPsychContext(
        colonist_id=cid, action=action, event_severity=event_severity,
        resource_avg=resource_avg, social_connections=social_connections,
        avg_trust=avg_trust, earth_contact=earth_contact,
        infra_completed=infra_completed, gov_participated=gov_participated,
        subsim_ran=subsim_ran, resolve=resolve, empathy=empathy,
        faith=faith, paranoia=paranoia,
    )


def _make_contexts(n=5):
    return [_make_context(cid=f"c-{i}") for i in range(n)]


# -- PsychState tests -------------------------------------------------------

class TestPsychState:
    def test_default_morale(self):
        p = PsychState()
        assert 0.0 <= p.morale <= 1.0
        assert p.morale == pytest.approx(0.665, abs=0.01)

    def test_high_stress_low_morale(self):
        p = PsychState(stress=0.95, loneliness=0.8, purpose=0.1)
        assert p.morale < 0.2

    def test_ideal_high_morale(self):
        p = PsychState(stress=0.0, loneliness=0.0, purpose=1.0)
        assert p.morale == pytest.approx(1.0, abs=0.01)

    def test_round_trip(self):
        p = PsychState(stress=0.4, loneliness=0.3, purpose=0.7, last_crisis_year=42)
        d = p.to_dict()
        p2 = PsychState.from_dict(d)
        assert p2.stress == pytest.approx(0.4, abs=0.001)
        assert p2.loneliness == pytest.approx(0.3, abs=0.001)
        assert p2.purpose == pytest.approx(0.7, abs=0.001)
        assert p2.last_crisis_year == 42

    def test_from_dict_defaults(self):
        p = PsychState.from_dict({})
        assert p.stress == 0.15
        assert p.purpose == 0.50

    def test_morale_in_dict(self):
        p = PsychState()
        d = p.to_dict()
        assert "morale" in d
        assert d["morale"] == pytest.approx(p.morale, abs=0.001)


# -- Stress delta tests ------------------------------------------------------

class TestStressDelta:
    def test_rest_reduces_stress(self):
        d = compute_stress_delta("rest", 0.0, 0.6, 0.5)
        assert d < 0

    def test_sabotage_increases_stress(self):
        d = compute_stress_delta("sabotage", 0.5, 0.6, 0.5)
        assert d > 0

    def test_high_resolve_aids_recovery(self):
        d_low = compute_stress_delta("rest", 0.0, 0.6, 0.1)
        d_high = compute_stress_delta("rest", 0.0, 0.6, 0.9)
        assert d_high < d_low

    def test_capped(self):
        d = compute_stress_delta("sabotage", 1.0, 0.0, 0.0)
        assert abs(d) <= STRESS_CAP_DELTA

    def test_resource_scarcity_stressful(self):
        d_scarce = compute_stress_delta("farm", 0.0, 0.1, 0.5)
        d_plenty = compute_stress_delta("farm", 0.0, 0.6, 0.5)
        assert d_scarce > d_plenty


# -- Loneliness delta tests --------------------------------------------------

class TestLonelinessDelta:
    def test_many_connections_reduce(self):
        d = compute_loneliness_delta(5, 0.7, True, 0.5)
        assert d < 0

    def test_isolation_increases(self):
        d_alone = compute_loneliness_delta(0, 0.0, False, 0.0)
        d_conn = compute_loneliness_delta(5, 0.7, True, 0.8)
        assert d_alone > d_conn

    def test_earth_contact_helps(self):
        d_contact = compute_loneliness_delta(3, 0.5, True, 0.5)
        d_no = compute_loneliness_delta(3, 0.5, False, 0.5)
        assert d_contact < d_no

    def test_capped(self):
        d = compute_loneliness_delta(0, 0.0, False, 0.0)
        assert abs(d) <= LONELINESS_CAP_DELTA


# -- Purpose delta tests -----------------------------------------------------

class TestPurposeDelta:
    def test_research_increases(self):
        d = compute_purpose_delta("research", False, False, False, 0.5)
        assert d > 0

    def test_sabotage_decreases(self):
        d = compute_purpose_delta("sabotage", False, False, False, 0.0)
        assert d < 0

    def test_infra_bonus(self):
        d_no = compute_purpose_delta("farm", False, False, False, 0.3)
        d_yes = compute_purpose_delta("farm", True, False, False, 0.3)
        assert d_yes > d_no

    def test_faith_provides_floor(self):
        d_no_faith = compute_purpose_delta("rest", False, False, False, 0.0)
        d_faith = compute_purpose_delta("rest", False, False, False, 0.9)
        assert d_faith > d_no_faith

    def test_capped(self):
        d = compute_purpose_delta("research", True, True, True, 1.0)
        assert abs(d) <= PURPOSE_CAP_DELTA


# -- Crisis check tests ------------------------------------------------------

class TestCrisis:
    def test_no_crisis_below_threshold(self):
        p = PsychState(stress=0.5)
        assert not check_crisis(p, 10, random.Random(42))

    def test_crisis_possible_above_threshold(self):
        p = PsychState(stress=0.95)
        triggered = any(check_crisis(p, 10, random.Random(s)) for s in range(100))
        assert triggered

    def test_cooldown_prevents_repeat(self):
        p = PsychState(stress=0.95, last_crisis_year=8)
        assert not check_crisis(p, 9, random.Random(0))
        assert not check_crisis(p, 10, random.Random(0))

    def test_cooldown_expires(self):
        p = PsychState(stress=0.95, last_crisis_year=5)
        triggered = any(check_crisis(p, 10, random.Random(s)) for s in range(100))
        assert triggered


# -- Colony aggregates tests -------------------------------------------------

class TestColonyAggregates:
    def test_colony_morale_empty(self):
        assert compute_colony_morale([]) == 0.5

    def test_colony_morale_average(self):
        states = [PsychState(stress=0.0, loneliness=0.0, purpose=1.0),
                  PsychState(stress=1.0, loneliness=1.0, purpose=0.0)]
        m = compute_colony_morale(states)
        assert 0.0 <= m <= 1.0

    def test_bottom_quartile_empty(self):
        assert compute_bottom_quartile_morale([]) == 0.5

    def test_bottom_quartile_surfaces_minority(self):
        happy = [PsychState(stress=0.0, loneliness=0.0, purpose=0.9)] * 8
        miserable = [PsychState(stress=0.9, loneliness=0.9, purpose=0.1)] * 2
        bq = compute_bottom_quartile_morale(happy + miserable)
        colony_avg = compute_colony_morale(happy + miserable)
        assert bq < colony_avg


# -- Death rate modifier tests -----------------------------------------------

class TestDeathRateModifier:
    def test_healthy_morale_no_change(self):
        assert death_rate_modifier(0.5) == 1.0

    def test_threshold_boundary(self):
        assert death_rate_modifier(MORALE_DEATH_THRESHOLD) == pytest.approx(1.0, abs=0.01)

    def test_zero_morale_max_modifier(self):
        assert death_rate_modifier(0.0) == pytest.approx(MORALE_DEATH_MULTIPLIER, abs=0.01)

    def test_monotonic(self):
        for m in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
            assert death_rate_modifier(m) >= death_rate_modifier(min(1.0, m + 0.05))


# -- Clamp / cap tests -------------------------------------------------------

class TestClampCap:
    def test_clamp_normal(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_floor(self):
        assert _clamp(-1.0) == 0.0

    def test_clamp_ceil(self):
        assert _clamp(2.0) == 1.0

    def test_cap_delta(self):
        assert _cap_delta(0.5, 0.2) == 0.2
        assert _cap_delta(-0.5, 0.2) == -0.2
        assert _cap_delta(0.1, 0.2) == 0.1


# -- Full tick tests ---------------------------------------------------------

class TestTickPsychology:
    def test_basic_tick(self):
        psych_map = {}
        contexts = _make_contexts(5)
        result = tick_psychology(psych_map, contexts, 10, random.Random(42))
        assert len(result.snapshots) == 5
        assert result.colony_morale > 0
        assert 0.0 <= result.colony_stress <= 1.0

    def test_creates_missing_psych(self):
        psych_map = {}
        contexts = [_make_context(cid="new-1")]
        tick_psychology(psych_map, contexts, 1, random.Random(0))
        assert "new-1" in psych_map

    def test_crisis_recorded(self):
        ctx = _make_context(cid="c-0", action="sabotage", event_severity=1.0,
                            resource_avg=0.1, resolve=0.0, paranoia=0.9)
        found_crisis = False
        for seed in range(200):
            pm = {"c-0": PsychState(stress=0.90)}
            result = tick_psychology(pm, [ctx], 10, random.Random(seed))
            if result.crises:
                found_crisis = True
                assert result.crises[0].colonist_id == "c-0"
                assert pm["c-0"].stress < 0.90
                break
        assert found_crisis

    def test_empty_colony(self):
        psych_map = {}
        result = tick_psychology(psych_map, [], 1, random.Random(0))
        assert result.snapshots == {}
        assert result.colony_morale == 0.5

    def test_result_serialisation(self):
        psych_map = {}
        contexts = _make_contexts(3)
        result = tick_psychology(psych_map, contexts, 10, random.Random(42))
        d = result.to_dict()
        assert "snapshots" in d
        assert "crises" in d
        assert "colony_morale" in d
        assert "bottom_quartile_morale" in d


# -- Property invariants (parametrized) --------------------------------------

@pytest.mark.parametrize("seed", list(range(20)))
class TestPropertyInvariants:
    def test_stress_bounded(self, seed):
        psych_map = {}
        contexts = _make_contexts(8)
        rng = random.Random(seed)
        for year in range(1, 11):
            tick_psychology(psych_map, contexts, year, rng)
        for p in psych_map.values():
            assert 0.0 <= p.stress <= 1.0

    def test_loneliness_bounded(self, seed):
        psych_map = {}
        contexts = _make_contexts(8)
        rng = random.Random(seed)
        for year in range(1, 11):
            tick_psychology(psych_map, contexts, year, rng)
        for p in psych_map.values():
            assert 0.0 <= p.loneliness <= 1.0

    def test_purpose_bounded(self, seed):
        psych_map = {}
        contexts = _make_contexts(8)
        rng = random.Random(seed)
        for year in range(1, 11):
            tick_psychology(psych_map, contexts, year, rng)
        for p in psych_map.values():
            assert 0.0 <= p.purpose <= 1.0

    def test_morale_bounded(self, seed):
        psych_map = {}
        contexts = _make_contexts(8)
        rng = random.Random(seed)
        for year in range(1, 11):
            tick_psychology(psych_map, contexts, year, rng)
        for p in psych_map.values():
            assert 0.0 <= p.morale <= 1.0

    def test_death_modifier_bounded(self, seed):
        psych_map = {}
        contexts = _make_contexts(5)
        rng = random.Random(seed)
        for year in range(1, 11):
            tick_psychology(psych_map, contexts, year, rng)
        for p in psych_map.values():
            m = death_rate_modifier(p.morale)
            assert 1.0 <= m <= MORALE_DEATH_MULTIPLIER
