"""Tests for Earth Protocol module."""
from __future__ import annotations

import random
import pytest

from src.mars100.earth import (
    EarthRelations, Treaty, TREATY_TEMPLATES,
    INDEPENDENCE_THRESHOLD, SUPPLY_BASE, CONTACT_DECAY, MAX_TREATIES,
    compute_supply_effects, compute_colonist_autonomy_stance,
    tick_earth_pre, tick_earth_post,
    _tick_contact, _tick_trust, _tick_supply_pipeline, _tick_autonomy,
    _expire_treaties, _propose_treaty, _should_colony_accept_treaty,
    _accept_treaty, _reject_treaty, _check_independence_vote,
    _clamp, _compute_resource_avg,
)


@pytest.fixture
def earth() -> EarthRelations:
    return EarthRelations()


@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def sample_colonist_dicts() -> list[dict]:
    rng = random.Random(99)
    return [
        {"id": f"colonist-{i}", "name": f"Test Colonist {i}",
         "stats": {s: rng.uniform(0.2, 0.8)
                   for s in ("resolve", "improvisation", "empathy",
                             "hoarding", "faith", "paranoia")}}
        for i in range(10)
    ]


@pytest.fixture
def year_summary() -> dict:
    return {"resources_after": {
        "food": 0.6, "water": 0.5, "power": 0.7,
        "air": 0.8, "medicine": 0.4}}


class TestTreaty:
    def test_roundtrip(self):
        t = Treaty(kind="resource_pact", start_year=5, duration=10,
                   terms={"supply_bonus": 0.02})
        t2 = Treaty.from_dict(t.to_dict())
        assert t2.kind == "resource_pact"
        assert t2.duration == 10
        assert t2.terms["supply_bonus"] == 0.02

    def test_defaults(self):
        t = Treaty.from_dict({"kind": "tech_transfer", "start_year": 1, "duration": 5})
        assert t.active is True and t.breached is False and t.terms == {}


class TestEarthRelations:
    def test_initial_state(self, earth: EarthRelations):
        assert earth.contact_quality == 1.0
        assert earth.earth_trust == 0.8
        assert earth.supply_pipeline == 1.0
        assert earth.autonomy_desire == 0.0
        assert not earth.independence_declared

    def test_roundtrip(self, earth: EarthRelations):
        earth.treaties.append(Treaty(kind="resource_pact", start_year=3, duration=10))
        earth.autonomy_desire = 0.35
        e2 = EarthRelations.from_dict(earth.to_dict())
        assert abs(e2.autonomy_desire - 0.35) < 0.001
        assert len(e2.treaties) == 1

    def test_active_treaties(self, earth: EarthRelations):
        earth.treaties = [
            Treaty(kind="resource_pact", start_year=1, duration=10),
            Treaty(kind="tech_transfer", start_year=1, duration=5, active=False),
            Treaty(kind="emergency_aid", start_year=1, duration=2, breached=True),
        ]
        assert len(earth.active_treaties()) == 1

    def test_history_capped(self, earth: EarthRelations):
        earth.history = [{"type": f"e{i}"} for i in range(30)]
        assert len(earth.to_dict()["history"]) == 20


class TestTickContact:
    def test_decays(self, earth: EarthRelations):
        _tick_contact(earth, [], 10)
        assert earth.contact_quality == pytest.approx(1.0 - CONTACT_DECAY, abs=0.01)

    def test_earth_contact_boosts(self, earth: EarthRelations):
        _tick_contact(earth, [{"name": "earth_contact", "severity": 0.1}], 10)
        assert earth.contact_quality > 1.0 - CONTACT_DECAY

    def test_solar_flare_hurts(self, earth: EarthRelations):
        _tick_contact(earth, [{"name": "solar_flare", "severity": 0.8}], 10)
        assert earth.contact_quality < 1.0 - CONTACT_DECAY

    def test_bounded(self, earth: EarthRelations):
        earth.contact_quality = 0.01
        _tick_contact(earth, [], 80)
        assert 0.0 <= earth.contact_quality <= 1.0


class TestTickTrust:
    def test_decays(self, earth: EarthRelations, rng: random.Random):
        _tick_trust(earth, "survival", 0.5, rng)
        assert earth.earth_trust < 0.8

    def test_democratic_boosts(self, earth: EarthRelations, rng: random.Random):
        initial = earth.earth_trust
        _tick_trust(earth, "democratic", 0.5, rng)
        assert earth.earth_trust > initial - 0.02

    def test_low_resources_erode(self, earth: EarthRelations, rng: random.Random):
        e_good = EarthRelations()
        _tick_trust(e_good, "council", 0.6, rng)
        e_bad = EarthRelations()
        _tick_trust(e_bad, "council", 0.15, rng)
        assert e_bad.earth_trust < e_good.earth_trust

    def test_breached_treaty_erodes(self, earth: EarthRelations, rng: random.Random):
        e_clean = EarthRelations()
        _tick_trust(e_clean, "council", 0.5, random.Random(42))
        e_breach = EarthRelations()
        e_breach.treaties.append(
            Treaty(kind="resource_pact", start_year=1, duration=10, breached=True))
        _tick_trust(e_breach, "council", 0.5, random.Random(42))
        assert e_breach.earth_trust < e_clean.earth_trust

    def test_bounded(self, earth: EarthRelations, rng: random.Random):
        earth.earth_trust = 0.01
        _tick_trust(earth, "autocratic", 0.1, rng)
        assert 0.0 <= earth.earth_trust <= 1.0


class TestTickSupplyPipeline:
    def test_decays(self, earth: EarthRelations):
        _tick_supply_pipeline(earth, 10)
        assert earth.supply_pipeline < 1.0

    def test_independence_accelerates(self, earth: EarthRelations):
        earth.independence_declared = True
        _tick_supply_pipeline(earth, 10)
        assert earth.supply_pipeline < 1.0 - 0.05

    def test_pact_slows_decay(self, earth: EarthRelations):
        earth.treaties.append(Treaty(kind="resource_pact", start_year=1, duration=10))
        _tick_supply_pipeline(earth, 10)
        assert earth.supply_pipeline > 1.0 - 0.01

    def test_bounded(self, earth: EarthRelations):
        earth.supply_pipeline = 0.01
        earth.independence_declared = True
        _tick_supply_pipeline(earth, 50)
        assert earth.supply_pipeline >= 0.0


class TestTickAutonomy:
    def test_grows(self, earth: EarthRelations, sample_colonist_dicts, rng):
        _tick_autonomy(earth, 10, sample_colonist_dicts, "council", rng)
        assert earth.autonomy_desire > 0.0

    def test_no_growth_after_independence(self, earth: EarthRelations,
                                          sample_colonist_dicts, rng):
        earth.independence_declared = True
        earth.autonomy_desire = 0.5
        _tick_autonomy(earth, 50, sample_colonist_dicts, "council", rng)
        assert earth.autonomy_desire == 0.5

    def test_emergency_aid_suppresses(self, earth: EarthRelations,
                                       sample_colonist_dicts, rng):
        earth.treaties.append(Treaty(kind="emergency_aid", start_year=1, duration=5))
        _tick_autonomy(earth, 5, sample_colonist_dicts, "council", rng)
        with_aid = earth.autonomy_desire
        e2 = EarthRelations()
        _tick_autonomy(e2, 5, sample_colonist_dicts, "council", rng)
        assert with_aid < e2.autonomy_desire

    def test_bounded(self, earth: EarthRelations, sample_colonist_dicts, rng):
        earth.autonomy_desire = 0.99
        _tick_autonomy(earth, 80, sample_colonist_dicts, "democratic", rng)
        assert earth.autonomy_desire <= 1.0

    def test_late_years_faster(self, earth: EarthRelations,
                                sample_colonist_dicts, rng):
        e1 = EarthRelations()
        _tick_autonomy(e1, 5, sample_colonist_dicts, "council", rng)
        e2 = EarthRelations()
        _tick_autonomy(e2, 80, sample_colonist_dicts, "council", rng)
        assert e2.autonomy_desire > e1.autonomy_desire


class TestExpireTreaties:
    def test_expires_old(self, earth: EarthRelations):
        earth.treaties.append(Treaty(kind="resource_pact", start_year=5, duration=10))
        events = _expire_treaties(earth, 15)
        assert len(events) == 1 and not earth.treaties[0].active

    def test_keeps_active(self, earth: EarthRelations):
        earth.treaties.append(Treaty(kind="resource_pact", start_year=5, duration=10))
        assert len(_expire_treaties(earth, 10)) == 0


class TestProposeTreaty:
    def test_no_proposal_after_independence(self, earth: EarthRelations, rng):
        earth.independence_declared = True
        assert _propose_treaty(earth, 10, 0.5, rng) == []

    def test_max_treaties_blocks(self, earth: EarthRelations, rng):
        for k in ["resource_pact", "tech_transfer", "emergency_aid"]:
            earth.treaties.append(Treaty(kind=k, start_year=1, duration=50))
        assert _propose_treaty(earth, 10, 0.5, rng) == []

    def test_emergency_on_low_resources(self, earth: EarthRelations):
        proposed = False
        for seed in range(100):
            e = EarthRelations(earth_trust=1.0)
            events = _propose_treaty(e, 10, 0.1, random.Random(seed))
            if events and events[0].get("kind") == "emergency_aid":
                proposed = True
                break
        assert proposed


class TestAcceptReject:
    def test_accept_adds(self, earth: EarthRelations):
        ev = _accept_treaty(earth, "resource_pact", 10)
        assert ev["type"] == "treaty_accepted" and len(earth.treaties) == 1

    def test_reject(self):
        ev = _reject_treaty("tech_transfer", 10)
        assert ev["type"] == "treaty_rejected"


class TestIndependenceVote:
    def test_no_vote_below_threshold(self, earth, sample_colonist_dicts, rng):
        earth.autonomy_desire = INDEPENDENCE_THRESHOLD - 0.1
        assert _check_independence_vote(earth, 50, sample_colonist_dicts, rng) == []

    def test_no_vote_before_year_20(self, earth, sample_colonist_dicts, rng):
        earth.autonomy_desire = 0.9
        assert _check_independence_vote(earth, 15, sample_colonist_dicts, rng) == []

    def test_no_double(self, earth, sample_colonist_dicts, rng):
        earth.independence_declared = True
        earth.autonomy_desire = 0.9
        assert _check_independence_vote(earth, 50, sample_colonist_dicts, rng) == []

    def test_vote_occurs(self, earth, sample_colonist_dicts, rng):
        earth.autonomy_desire = 0.9
        events = _check_independence_vote(earth, 50, sample_colonist_dicts, rng)
        assert len(events) >= 1 and events[0]["type"] == "independence_vote"

    def test_independence_cuts_treaties(self, earth, rng):
        earth.autonomy_desire = 0.95
        colonists = [{"id": f"c-{i}", "stats": {"resolve": 0.9, "paranoia": 0.9,
                       "empathy": 0.1, "faith": 0.1}} for i in range(10)]
        earth.treaties.append(Treaty(kind="resource_pact", start_year=1, duration=50))
        events = _check_independence_vote(earth, 50, colonists, rng)
        votes = [e for e in events if e["type"] == "independence_vote"]
        assert votes[0]["passed"] is True
        assert earth.independence_declared and not earth.treaties[0].active


class TestSupplyEffects:
    def test_positive(self, earth):
        assert all(v > 0 for v in compute_supply_effects(earth).values())

    def test_sums_correct(self, earth):
        total = sum(compute_supply_effects(earth).values())
        expected = SUPPLY_BASE * (1.0 * 0.5 + 0.8 * 0.3 + 1.0 * 0.2)
        assert abs(total - expected) < 0.001

    def test_independence_reduces(self, earth):
        before = sum(compute_supply_effects(earth).values())
        earth.independence_declared = True
        assert sum(compute_supply_effects(earth).values()) < before

    def test_pact_boosts(self, earth):
        before = sum(compute_supply_effects(earth).values())
        earth.treaties.append(Treaty(kind="resource_pact", start_year=1, duration=10,
                                      terms={"supply_bonus": 0.02}))
        assert sum(compute_supply_effects(earth).values()) > before

    def test_all_resources(self, earth):
        for res in ("food", "water", "medicine", "power", "air"):
            assert res in compute_supply_effects(earth)


class TestColonistStance:
    def test_high_resolve_pro_independence(self, earth):
        c = {"stats": {"resolve": 0.9, "paranoia": 0.5, "empathy": 0.3, "faith": 0.3}}
        assert compute_colonist_autonomy_stance(c, earth) > 0.5

    def test_high_empathy_pro_earth(self, earth):
        c = {"stats": {"resolve": 0.2, "paranoia": 0.2, "empathy": 0.9, "faith": 0.9}}
        assert compute_colonist_autonomy_stance(c, earth) < 0.5

    def test_social_pressure(self):
        c = {"stats": {"resolve": 0.5, "paranoia": 0.5, "empathy": 0.5, "faith": 0.5}}
        assert (compute_colonist_autonomy_stance(c, EarthRelations(autonomy_desire=0.9))
                > compute_colonist_autonomy_stance(c, EarthRelations(autonomy_desire=0.0)))

    def test_bounded(self, earth):
        for r in [0.0, 1.0]:
            for p in [0.0, 1.0]:
                c = {"stats": {"resolve": r, "paranoia": p, "empathy": 0.5, "faith": 0.5}}
                assert 0.0 <= compute_colonist_autonomy_stance(c, earth) <= 1.0


class TestTickEarthPre:
    def test_returns_dict(self, earth, rng):
        effects = tick_earth_pre(earth, [], 1, rng)
        assert isinstance(effects, dict) and "food" in effects

    def test_contact_updated(self, earth, rng):
        tick_earth_pre(earth, [], 5, rng)
        assert earth.contact_quality < 1.0


class TestTickEarthPost:
    def test_returns_list(self, earth, year_summary, sample_colonist_dicts, rng):
        events = tick_earth_post(earth, 10, year_summary, sample_colonist_dicts, "council", rng)
        assert isinstance(events, list)

    def test_autonomy_grows(self, earth, year_summary, sample_colonist_dicts, rng):
        for y in range(1, 20):
            tick_earth_post(earth, y, year_summary, sample_colonist_dicts, "council", rng)
        assert earth.autonomy_desire > 0.0


class TestComputeResourceAvg:
    def test_normal(self):
        s = {"resources_after": {"food": 0.6, "water": 0.5, "power": 0.7,
                                  "air": 0.8, "medicine": 0.4}}
        assert abs(_compute_resource_avg(s) - 0.6) < 0.01

    def test_empty(self):
        assert _compute_resource_avg({}) == 0.5


class TestBoundedProperties:
    def test_100_years_bounded(self, sample_colonist_dicts):
        earth = EarthRelations()
        rng = random.Random(42)
        summary = {"resources_after": dict.fromkeys(
            ("food", "water", "power", "air", "medicine"), 0.5)}
        for year in range(1, 101):
            events = []
            if rng.random() < 0.15:
                events.append({"name": "earth_contact", "severity": 0.1})
            supply = tick_earth_pre(earth, events, year, rng)
            assert all(0.0 <= v <= 1.0 for v in supply.values())
            tick_earth_post(earth, year, summary, sample_colonist_dicts, "council", rng)
            assert 0.0 <= earth.contact_quality <= 1.0
            assert 0.0 <= earth.earth_trust <= 1.0
            assert 0.0 <= earth.supply_pipeline <= 1.0
            assert 0.0 <= earth.autonomy_desire <= 1.0

    def test_deterministic(self, sample_colonist_dicts):
        summary = {"resources_after": dict.fromkeys(
            ("food", "water", "power", "air", "medicine"), 0.5)}
        def run(seed, years):
            e = EarthRelations()
            r = random.Random(seed)
            for y in range(1, years + 1):
                tick_earth_pre(e, [], y, r)
                tick_earth_post(e, y, summary, sample_colonist_dicts, "council", r)
            return e.to_dict()
        assert run(42, 50) == run(42, 50)

    def test_supply_never_negative(self):
        for seed in range(20):
            e = EarthRelations(
                contact_quality=random.Random(seed).random(),
                earth_trust=random.Random(seed + 1).random(),
                supply_pipeline=random.Random(seed + 2).random())
            assert all(v >= 0 for v in compute_supply_effects(e).values())
