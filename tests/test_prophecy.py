"""Tests for the Mars-100 prophecy engine."""
from __future__ import annotations

import random

import pytest

from src.mars100.prophecy import (
    AVERTED_CREDIT,
    INFLUENCE_BOUND,
    MAX_PROPHECIES,
    PREDICTION_TYPES,
    PROPHECY_INTERVAL,
    PROPHECY_LOOKAHEAD_MAX,
    Prophecy,
    ProphecyState,
    compute_prophecy_influence,
    generate_prophecy,
    resolve_prophecy,
    select_prophet,
    _analyse_trend,
    _build_counter_expr,
    _build_projection_expr,
    _classify_prediction,
    _compute_confidence,
    _is_crisis,
    _is_prosperity,
    _is_death_wave,
    _is_governance_shift,
    _is_breakthrough,
    _was_trending_toward,
)
from src.mars100.colonist import create_founding_ten, Colonist
from src.mars100.subsim import SubSimResult, SubSimBudget


# ---- Fixtures ----------------------------------------------------------------

@pytest.fixture
def colonists():
    return create_founding_ten(seed=42)


@pytest.fixture
def rng():
    return random.Random(42)


@pytest.fixture
def sample_history():
    """10 years of fake history summaries."""
    return [
        {
            "year": y,
            "resources_after": {
                "food": 0.6 - y * 0.02,
                "water": 0.7 - y * 0.01,
                "power": 0.5,
                "air": 0.8,
                "medicine": 0.4,
            },
            "deaths": [],
            "governance": None,
            "infrastructure": {},
        }
        for y in range(1, 11)
    ]


# ---- Resolution predicates ---------------------------------------------------

class TestResolutionPredicates:
    def test_crisis_below_threshold(self):
        assert _is_crisis({"resources_after": {"food": 0.1, "water": 0.5}})

    def test_crisis_above_threshold(self):
        assert not _is_crisis({"resources_after": {"food": 0.3, "water": 0.5}})

    def test_crisis_empty(self):
        assert not _is_crisis({})

    def test_prosperity_all_high(self):
        assert _is_prosperity({"resources_after": {"food": 0.7, "water": 0.8}})

    def test_prosperity_one_low(self):
        assert not _is_prosperity({"resources_after": {"food": 0.7, "water": 0.3}})

    def test_death_wave_two_deaths(self):
        assert _is_death_wave({"deaths": [{"id": "a"}, {"id": "b"}]})

    def test_death_wave_one_death(self):
        assert not _is_death_wave({"deaths": [{"id": "a"}]})

    def test_governance_shift_passed(self):
        assert _is_governance_shift({"governance": {"passed": True}})

    def test_governance_shift_rejected(self):
        assert not _is_governance_shift({"governance": {"passed": False}})

    def test_breakthrough_completed(self):
        assert _is_breakthrough({"infrastructure": {"just_completed": "greenhouse_dome"}})

    def test_breakthrough_none(self):
        assert not _is_breakthrough({"infrastructure": {}})


# ---- Prophecy dataclass ------------------------------------------------------

class TestProphecy:
    def test_to_dict_roundtrip(self):
        p = Prophecy(
            prophet_id="0", year_made=10, year_target=15,
            prediction_type="crisis", confidence=0.7,
            subsim_depth=2, evidence_expr="(+ 1 2)",
            evidence_result={"primary": 3, "counter": 0.5},
        )
        d = p.to_dict()
        assert d["prophet_id"] == "0"
        assert d["year_target"] == 15
        assert d["prediction_type"] == "crisis"
        assert d["subsim_depth"] == 2
        assert isinstance(d["confidence"], float)

    def test_to_dict_serializable(self):
        """Prophecy.to_dict() must be JSON-serializable."""
        import json
        p = Prophecy(
            prophet_id="1", year_made=20, year_target=25,
            prediction_type="prosperity", confidence=0.5,
            evidence_result=[1, 2, {"nested": True}],
        )
        serialized = json.dumps(p.to_dict())
        assert isinstance(serialized, str)


# ---- ProphecyState -----------------------------------------------------------

class TestProphecyState:
    def test_empty_state(self):
        s = ProphecyState()
        assert s.active_prophecies(1) == []
        assert s.prophet_accuracy("nobody") == 0.5

    def test_active_prophecies_filters(self):
        s = ProphecyState(prophecies=[
            Prophecy("0", 1, 10, "crisis", 0.5),
            Prophecy("1", 1, 5, "prosperity", 0.5, resolved=True, outcome="hit"),
        ])
        active = s.active_prophecies(5)
        assert len(active) == 1
        assert active[0].year_target == 10

    def test_pending_for_year(self):
        s = ProphecyState(prophecies=[
            Prophecy("0", 1, 10, "crisis", 0.5),
            Prophecy("1", 1, 15, "crisis", 0.5),
        ])
        pending = s.pending_for_year(10)
        assert len(pending) == 1

    def test_record_outcome_hit(self):
        s = ProphecyState()
        s.record_outcome("0", "hit")
        assert s.track_record["0"]["hits"] == 1
        assert s.track_record["0"]["total"] == 1
        assert s.prophet_accuracy("0") == 1.0

    def test_record_outcome_averted(self):
        s = ProphecyState()
        s.record_outcome("0", "averted")
        assert s.track_record["0"]["averted"] == 1
        assert s.prophet_accuracy("0") == AVERTED_CREDIT

    def test_record_outcome_miss(self):
        s = ProphecyState()
        s.record_outcome("0", "miss")
        assert s.track_record["0"]["misses"] == 1
        assert s.prophet_accuracy("0") == 0.0

    def test_update_influence(self):
        s = ProphecyState()
        s.record_outcome("0", "hit")
        s.record_outcome("0", "hit")
        s._update_influence()
        assert s.current_influence > 0
        assert s.current_influence <= INFLUENCE_BOUND

    def test_influence_zero_when_no_records(self):
        s = ProphecyState()
        s._update_influence()
        assert s.current_influence == 0.0

    def test_to_dict_serializable(self):
        import json
        s = ProphecyState(prophecies=[
            Prophecy("0", 1, 10, "crisis", 0.5),
        ])
        s.record_outcome("0", "hit")
        serialized = json.dumps(s.to_dict())
        assert isinstance(serialized, str)

    def test_summary(self):
        s = ProphecyState(prophecies=[
            Prophecy("0", 1, 10, "crisis", 0.5),
        ])
        summary = s.summary()
        assert summary["total"] == 1
        assert summary["active"] == 1
        assert summary["resolved"] == 0


# ---- Prophet selection -------------------------------------------------------

class TestSelectProphet:
    def test_returns_colonist(self, colonists, rng):
        prophet = select_prophet(colonists, {}, rng)
        assert prophet is not None
        assert prophet.is_active()

    def test_prefers_high_coding(self):
        """Prophet with higher coding skill should be selected more often."""
        from src.mars100.colonist import create_founding_ten
        colonists = create_founding_ten(seed=1)
        # Boost one colonist's coding
        colonists[0].skills.coding = 1.0
        colonists[0].stats.improvisation = 1.0
        for c in colonists[1:]:
            c.skills.coding = 0.0
            c.stats.improvisation = 0.0

        counts: dict[str, int] = {}
        for seed in range(100):
            rng = random.Random(seed)
            p = select_prophet(colonists, {}, rng)
            counts[p.id] = counts.get(p.id, 0) + 1
        # The boosted colonist should be selected most often
        assert counts.get(colonists[0].id, 0) > 30

    def test_no_active_colonists(self, rng):
        assert select_prophet([], {}, rng) is None

    def test_track_record_boosts(self, colonists, rng):
        track = {colonists[3].id: {"hits": 5, "averted": 0, "misses": 0, "total": 5}}
        # With high track record the experienced prophet should appear more
        counts: dict[str, int] = {}
        for seed in range(100):
            r = random.Random(seed)
            p = select_prophet(colonists, track, r)
            counts[p.id] = counts.get(p.id, 0) + 1
        assert counts.get(colonists[3].id, 0) >= 5


# ---- Trend analysis ----------------------------------------------------------

class TestTrendAnalysis:
    def test_flat_trend(self):
        history = [{"resources_after": {"food": 0.5}} for _ in range(5)]
        assert _analyse_trend(history, "food") == 0.0

    def test_declining_trend(self, sample_history):
        trend = _analyse_trend(sample_history, "food")
        assert trend < 0

    def test_missing_resource(self, sample_history):
        assert _analyse_trend(sample_history, "unobtanium") == 0.0

    def test_short_history(self):
        assert _analyse_trend([{"resources_after": {"food": 0.5}}], "food") == 0.0


# ---- Expression building -----------------------------------------------------

class TestExpressionBuilding:
    def test_projection_expr_evaluable(self):
        from src.mars100.lispy_vm import run as lispy_run
        expr = _build_projection_expr(
            {"food": -0.02, "water": 0.01},
            {"food": 0.5, "water": 0.7},
        )
        result = lispy_run(expr)
        assert isinstance(result, (int, float))

    def test_counter_expr_evaluable(self):
        from src.mars100.lispy_vm import run as lispy_run
        expr = _build_counter_expr(0.3, 0.7)
        result = lispy_run(expr)
        assert isinstance(result, (int, float))

    def test_empty_trends_fallback(self):
        expr = _build_projection_expr({}, {})
        from src.mars100.lispy_vm import run as lispy_run
        result = lispy_run(expr)
        assert result == 0.5


# ---- Prediction classification -----------------------------------------------

class TestClassification:
    def test_crisis_on_low_primary(self):
        rng = random.Random(1)
        assert _classify_prediction(0.1, 0.3, {"food": -0.01}, rng) == "crisis"

    def test_prosperity_on_high(self):
        rng = random.Random(1)
        assert _classify_prediction(0.8, 0.7, {"food": 0.01}, rng) == "prosperity"

    def test_death_wave_on_negative_trends(self):
        rng = random.Random(1)
        trends = {"food": -0.03, "water": -0.04, "power": -0.05}
        assert _classify_prediction(0.4, 0.3, trends, rng) == "death_wave"

    def test_governance_shift_on_divergence(self):
        rng = random.Random(1)
        assert _classify_prediction(0.3, 0.7, {"food": 0.0}, rng) == "governance_shift"

    def test_fallback_random(self):
        rng = random.Random(42)
        ptype = _classify_prediction(0.5, 0.5, {"food": 0.0}, rng)
        assert ptype in PREDICTION_TYPES


# ---- Prophecy generation -----------------------------------------------------

class TestGenerateProphecy:
    def test_generates_valid_prophecy(self, colonists, sample_history, rng):
        prophet = colonists[0]
        resources = {"food": 0.5, "water": 0.7, "power": 0.5,
                     "air": 0.8, "medicine": 0.4}
        subsim_log: list = []
        p = generate_prophecy(
            prophet, 10, sample_history, resources, 100, subsim_log, rng)
        assert p is not None
        assert p.prophet_id == prophet.id
        assert p.year_made == 10
        assert 15 <= p.year_target <= 20
        assert p.prediction_type in PREDICTION_TYPES
        assert 0 < p.confidence <= 1.0
        assert p.subsim_depth >= 1

    def test_depth_2_subsim_spawned(self, colonists, sample_history, rng):
        """Prophecy should spawn at least depth-2 sub-sim."""
        subsim_log: list = []
        p = generate_prophecy(
            colonists[0], 10, sample_history,
            {"food": 0.5, "water": 0.7, "power": 0.5, "air": 0.8, "medicine": 0.4},
            100, subsim_log, rng)
        depths = [s.depth for s in subsim_log]
        assert 2 in depths, f"Expected depth-2 sub-sim, got depths: {depths}"

    def test_none_when_target_exceeds_total(self, colonists, rng):
        """Should return None if year_target > total_years."""
        p = generate_prophecy(
            colonists[0], 96, [],
            {"food": 0.5}, 100, [], rng)
        assert p is None

    def test_subsim_log_populated(self, colonists, sample_history, rng):
        log: list = []
        generate_prophecy(
            colonists[0], 10, sample_history,
            {"food": 0.5, "water": 0.7, "power": 0.5, "air": 0.8, "medicine": 0.4},
            100, log, rng)
        assert len(log) >= 2  # primary + counter

    def test_prophecy_serializable(self, colonists, sample_history, rng):
        import json
        p = generate_prophecy(
            colonists[0], 10, sample_history,
            {"food": 0.5, "water": 0.7, "power": 0.5, "air": 0.8, "medicine": 0.4},
            100, [], rng)
        assert p is not None
        serialized = json.dumps(p.to_dict())
        assert isinstance(serialized, str)


# ---- Resolution ---------------------------------------------------------------

class TestResolveResolution:
    def test_hit_on_matching_crisis(self):
        p = Prophecy("0", 1, 10, "crisis", 0.7)
        outcome = resolve_prophecy(
            p, {"resources_after": {"food": 0.1, "water": 0.5}})
        assert outcome == "hit"
        assert p.resolved
        assert p.outcome == "hit"

    def test_miss_on_no_match(self):
        p = Prophecy("0", 1, 10, "crisis", 0.7)
        outcome = resolve_prophecy(
            p, {"resources_after": {"food": 0.8, "water": 0.9}})
        assert outcome == "miss"

    def test_averted_crisis(self):
        p = Prophecy("0", 1, 10, "crisis", 0.7)
        prev = {"resources_after": {"food": 0.25, "water": 0.5}}
        current = {"resources_after": {"food": 0.4, "water": 0.6}}
        outcome = resolve_prophecy(p, current, prev)
        assert outcome == "averted"

    def test_unknown_prediction_type(self):
        p = Prophecy("0", 1, 10, "alien_invasion", 0.7)
        outcome = resolve_prophecy(p, {})
        assert outcome == "miss"

    def test_resolved_only_once(self):
        p = Prophecy("0", 1, 10, "crisis", 0.7)
        resolve_prophecy(p, {"resources_after": {"food": 0.1}})
        assert p.resolved


# ---- Averted detection -------------------------------------------------------

class TestAvertedDetection:
    def test_crisis_averted(self):
        prev = {"resources_after": {"food": 0.25}}
        cur = {"resources_after": {"food": 0.4}}
        assert _was_trending_toward("crisis", prev, cur)

    def test_crisis_not_averted(self):
        prev = {"resources_after": {"food": 0.6}}
        cur = {"resources_after": {"food": 0.7}}
        assert not _was_trending_toward("crisis", prev, cur)

    def test_death_wave_trending(self):
        prev = {"deaths": [{"id": "a"}]}
        cur = {"deaths": []}
        assert _was_trending_toward("death_wave", prev, cur)


# ---- Prophecy influence -------------------------------------------------------

class TestProphecyInfluence:
    def test_empty_state_no_influence(self):
        s = ProphecyState()
        assert compute_prophecy_influence(s, 10) == {}

    def test_crisis_boosts_farm(self):
        s = ProphecyState(
            prophecies=[Prophecy("0", 5, 15, "crisis", 0.8)],
            current_influence=0.2,
        )
        influence = compute_prophecy_influence(s, 10)
        assert influence.get("farm", 0) > 0

    def test_influence_bounded(self):
        s = ProphecyState(
            prophecies=[Prophecy("0", 5, 15, "crisis", 1.0)],
            current_influence=0.3,
        )
        influence = compute_prophecy_influence(s, 14)
        for v in influence.values():
            assert -INFLUENCE_BOUND <= v <= INFLUENCE_BOUND

    def test_proximity_increases_strength(self):
        s = ProphecyState(
            prophecies=[Prophecy("0", 1, 10, "crisis", 0.8)],
            current_influence=0.2,
        )
        far = compute_prophecy_influence(s, 2)
        near = compute_prophecy_influence(s, 9)
        assert near.get("farm", 0) >= far.get("farm", 0)

    def test_death_wave_suppresses_sabotage(self):
        s = ProphecyState(
            prophecies=[Prophecy("0", 5, 15, "death_wave", 0.8)],
            current_influence=0.2,
        )
        influence = compute_prophecy_influence(s, 10)
        assert influence.get("sabotage", 0) < 0


# ---- Determinism --------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_prophecy(self, colonists, sample_history):
        """Two runs with same seed must produce identical prophecy."""
        def run_once(seed):
            rng = random.Random(seed)
            return generate_prophecy(
                colonists[0], 10, sample_history,
                {"food": 0.5, "water": 0.7, "power": 0.5,
                 "air": 0.8, "medicine": 0.4},
                100, [], rng)

        p1 = run_once(99)
        p2 = run_once(99)
        assert p1 is not None and p2 is not None
        assert p1.prediction_type == p2.prediction_type
        assert p1.year_target == p2.year_target
        assert p1.confidence == p2.confidence


# ---- Integration with engine --------------------------------------------------

class TestEngineIntegration:
    def test_engine_with_prophecy(self):
        """Run 20 years and verify prophecy is generated at year 10."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        years_with_prophecy = [
            y for y in result.years if y.prophecy
        ]
        assert len(years_with_prophecy) >= 1, \
            "Expected at least one prophecy in 20 years"

    def test_prophecy_in_year_result(self):
        """Year result should contain prophecy summary."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=20)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "prophecy" in d

    def test_full_100_year_with_prophecy(self):
        """Run 100 years and verify prophecy data in simulation result."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        rd = result.to_dict()
        assert "prophecy" in rd
        prophecy_data = rd["prophecy"]
        assert len(prophecy_data.get("prophecies", [])) > 0

    def test_prophecy_subsims_reach_depth_2(self):
        """At least one prophecy sub-sim should reach depth 2."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=30)
        result = engine.run()
        all_subsims = []
        for yr in result.years:
            all_subsims.extend(yr.subsim_log)
        depths = [s.get("depth", 1) for s in all_subsims]
        assert 2 in depths, f"No depth-2 sub-sims found. Depths: {set(depths)}"

    def test_prophecy_resolution_happens(self):
        """At least one prophecy should be resolved in a 100-year run."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        prophecy_data = result.to_dict().get("prophecy", {})
        prophecies = prophecy_data.get("prophecies", [])
        resolved_count = sum(1 for p in prophecies if p.get("resolved"))
        assert resolved_count > 0, \
            f"Expected resolved prophecies in 100-year run, got {len(prophecies)} total"
