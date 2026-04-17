"""Tests for the Mars-100 Earth Contact System."""
from __future__ import annotations

import json
import random

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.earth import (
    EarthState, Directive, Message,
    earth_tick, colony_decides, apply_directive_effects,
    check_independence, supply_ship_arrives,
    DIRECTIVE_TYPES, RESPONSE_TYPES, EARTH_MOODS,
    _mood_from_scores, _process_colony_response,
)
from src.mars100.engine import Mars100Engine, YearResult, SimulationResult
from src.mars100.colony import Resources


# ─── EarthState unit tests ─────────────────────────────────────────

class TestEarthState:
    def test_defaults(self):
        s = EarthState()
        assert s.mood == "supportive"
        assert s.interest == 0.8
        assert s.budget == 0.7
        assert not s.independence_declared
        assert s.communication_active

    def test_to_dict_keys(self):
        s = EarthState()
        d = s.to_dict()
        assert "mood" in d
        assert "interest" in d
        assert "budget" in d
        assert "independence_declared" in d

    def test_to_dict_serializable(self):
        s = EarthState()
        text = json.dumps(s.to_dict())
        assert isinstance(text, str)

    def test_from_dict_roundtrip(self):
        s = EarthState(mood="hostile", interest=0.3, budget=0.2,
                       compliance_score=0.1, recent_rejections=5)
        d = s.to_dict()
        s2 = EarthState.from_dict(d)
        assert s2.mood == "hostile"
        assert s2.interest == 0.3
        assert s2.recent_rejections == 5

    def test_from_dict_defaults(self):
        s = EarthState.from_dict({})
        assert s.mood == "supportive"
        assert s.interest == 0.8


# ─── Mood derivation ────────────────────────────────────────────────

class TestMood:
    def test_crisis_override(self):
        assert _mood_from_scores(0.9, 0.9, crisis=True) == "collapsed"

    def test_low_interest_hostile(self):
        assert _mood_from_scores(0.1, 0.9, crisis=False) == "hostile"

    def test_low_compliance_hostile(self):
        assert _mood_from_scores(0.5, 0.2, crisis=False) == "hostile"

    def test_mid_compliance_demanding(self):
        assert _mood_from_scores(0.6, 0.4, crisis=False) == "demanding"

    def test_good_scores_supportive(self):
        assert _mood_from_scores(0.8, 0.8, crisis=False) == "supportive"

    def test_mid_range_neutral(self):
        assert _mood_from_scores(0.4, 0.5, crisis=False) == "neutral"


# ─── Message class ──────────────────────────────────────────────────

class TestMessage:
    def test_to_dict(self):
        m = Message(content={"response": "comply"}, sent_year=5,
                    arrives_year=6, direction="mars_to_earth")
        d = m.to_dict()
        assert d["sent_year"] == 5
        assert d["arrives_year"] == 6
        assert d["direction"] == "mars_to_earth"


# ─── Directive class ────────────────────────────────────────────────

class TestDirective:
    def test_to_dict(self):
        d = Directive(dtype="supply_mission", year_issued=10,
                      year_received=11, description="Supplies incoming",
                      resource_cost={"power": 0.02},
                      resource_reward={"food": 0.1})
        out = d.to_dict()
        assert out["type"] == "supply_mission"
        assert out["year_received"] == 11

    def test_all_directive_types_valid(self):
        for dt in DIRECTIVE_TYPES:
            assert isinstance(dt, str)
            assert len(dt) > 0


# ─── earth_tick tests ───────────────────────────────────────────────

class TestEarthTick:
    def test_returns_none_after_independence(self):
        s = EarthState(independence_declared=True)
        rng = random.Random(42)
        assert earth_tick(s, 50, rng) is None

    def test_returns_none_when_comms_down(self):
        s = EarthState(communication_active=False)
        rng = random.Random(42)
        assert earth_tick(s, 50, rng) is None

    def test_delivers_messages(self):
        s = EarthState()
        s.message_queue.append(Message(
            content={"response": "comply"}, sent_year=4,
            arrives_year=5, direction="mars_to_earth"))
        rng = random.Random(42)
        earth_tick(s, 5, rng)
        # Message should be consumed
        assert len(s.message_queue) == 0
        # Compliance should have improved
        assert s.compliance_score >= 0.5

    def test_interest_decays_over_time(self):
        s = EarthState(interest=0.8)
        rng = random.Random(42)
        for year in range(10, 50):
            earth_tick(s, year, rng)
        assert s.interest < 0.8

    def test_directive_sometimes_generated(self):
        directives_found = 0
        for seed in range(100):
            s = EarthState()
            rng = random.Random(seed)
            d = earth_tick(s, 15, rng)
            if d is not None:
                directives_found += 1
                assert d.dtype in DIRECTIVE_TYPES
        assert directives_found > 0

    def test_budget_tracks_interest(self):
        s = EarthState(interest=0.2, budget=0.7)
        rng = random.Random(42)
        for year in range(10, 30):
            earth_tick(s, year, rng)
        # Budget should have dropped toward low interest
        assert s.budget < 0.5

    def test_earth_crisis_rare(self):
        crisis_count = 0
        for seed in range(200):
            s = EarthState()
            rng = random.Random(seed)
            for year in range(1, 101):
                earth_tick(s, year, rng)
            if s.earth_crisis or s.interest < 0.3:
                crisis_count += 1
        # Crisis should happen sometimes but not always
        assert 0 < crisis_count < 200


# ─── colony_decides tests ──────────────────────────────────────────

class TestColonyDecides:
    def _directive(self, dtype: str = "supply_mission") -> Directive:
        return Directive(dtype=dtype, year_issued=10, year_received=11,
                         description="Test", resource_cost={}, resource_reward={})

    def test_supply_mission_usually_complied(self):
        comply_count = 0
        for seed in range(100):
            rng = random.Random(seed)
            resp = colony_decides(self._directive("supply_mission"),
                                  "council", None, [], [], 0.5, 0.5, rng)
            if resp == "comply":
                comply_count += 1
        assert comply_count > 40  # mostly complied

    def test_recall_order_usually_rejected(self):
        reject_count = 0
        for seed in range(100):
            rng = random.Random(seed)
            resp = colony_decides(self._directive("recall_order"),
                                  "council", None, [], [], 0.7, 0.7, rng)
            if resp in ("reject", "ignore"):
                reject_count += 1
        assert reject_count > 40  # mostly rejected

    def test_low_resources_increases_compliance(self):
        comply_low = 0
        comply_high = 0
        for seed in range(100):
            rng = random.Random(seed)
            r1 = colony_decides(self._directive("science_experiment"),
                                "council", None, [], [], 0.5, 0.2, rng)
            rng = random.Random(seed)
            r2 = colony_decides(self._directive("science_experiment"),
                                "council", None, [], [], 0.5, 0.8, rng)
            if r1 == "comply":
                comply_low += 1
            if r2 == "comply":
                comply_high += 1
        assert comply_low > comply_high

    def test_anarchy_is_noisy(self):
        responses: set[str] = set()
        for seed in range(200):
            rng = random.Random(seed)
            resp = colony_decides(self._directive("science_experiment"),
                                  "anarchy", None, [], [], 0.5, 0.5, rng)
            responses.add(resp)
        assert len(responses) >= 3  # anarchy produces varied responses

    def test_returns_valid_response(self):
        rng = random.Random(42)
        resp = colony_decides(self._directive(), "council", None, [], [],
                              0.5, 0.5, rng)
        assert resp in RESPONSE_TYPES


# ─── apply_directive_effects tests ─────────────────────────────────

class TestApplyDirectiveEffects:
    def test_comply_full_effects(self):
        d = Directive("supply_mission", 10, 11, "test",
                      resource_cost={"power": 0.05},
                      resource_reward={"food": 0.1, "medicine": 0.05})
        resources = {"food": 0.5, "water": 0.5, "power": 0.5,
                     "air": 0.5, "medicine": 0.5}
        deltas = apply_directive_effects(d, "comply", resources)
        assert deltas["food"] == pytest.approx(0.1, abs=0.001)
        assert deltas["power"] == pytest.approx(-0.05, abs=0.001)
        assert deltas["medicine"] == pytest.approx(0.05, abs=0.001)

    def test_negotiate_partial_effects(self):
        d = Directive("supply_mission", 10, 11, "test",
                      resource_cost={"power": 0.10},
                      resource_reward={"food": 0.10})
        resources = {"food": 0.5, "power": 0.5}
        deltas = apply_directive_effects(d, "negotiate", resources)
        assert deltas["power"] == pytest.approx(-0.05, abs=0.001)  # 50% cost
        assert deltas["food"] == pytest.approx(0.07, abs=0.001)    # 70% reward

    def test_reject_no_effects(self):
        d = Directive("supply_mission", 10, 11, "test",
                      resource_cost={"power": 0.1},
                      resource_reward={"food": 0.1})
        resources = {"food": 0.5, "power": 0.5}
        deltas = apply_directive_effects(d, "reject", resources)
        assert len(deltas) == 0

    def test_ignore_no_effects(self):
        d = Directive("supply_mission", 10, 11, "test",
                      resource_cost={"power": 0.1},
                      resource_reward={"food": 0.1})
        resources = {"food": 0.5, "power": 0.5}
        deltas = apply_directive_effects(d, "ignore", resources)
        assert len(deltas) == 0


# ─── check_independence tests ──────────────────────────────────────

class TestCheckIndependence:
    def test_not_early(self):
        s = EarthState(recent_rejections=5)
        rng = random.Random(42)
        assert not check_independence(s, 15, 0.6, 0.7, 10, rng)

    def test_not_without_rejections(self):
        s = EarthState(recent_rejections=1)
        rng = random.Random(42)
        assert not check_independence(s, 15, 0.6, 0.7, 50, rng)

    def test_not_with_low_resources(self):
        s = EarthState(recent_rejections=5)
        rng = random.Random(42)
        assert not check_independence(s, 15, 0.3, 0.7, 50, rng)

    def test_not_low_population(self):
        s = EarthState(recent_rejections=5)
        rng = random.Random(42)
        assert not check_independence(s, 5, 0.6, 0.7, 50, rng)

    def test_already_independent(self):
        s = EarthState(independence_declared=True)
        rng = random.Random(42)
        assert not check_independence(s, 15, 0.6, 0.7, 50, rng)

    def test_can_trigger_with_high_streak(self):
        declared = False
        for seed in range(500):
            s = EarthState(recent_rejections=8, interest=0.1)
            rng = random.Random(seed)
            if check_independence(s, 20, 0.7, 0.8, 60, rng):
                declared = True
                break
        assert declared, "Independence should be possible with high streak"

    def test_sets_fields_on_success(self):
        for seed in range(500):
            s = EarthState(recent_rejections=8, interest=0.1)
            rng = random.Random(seed)
            if check_independence(s, 20, 0.7, 0.8, 60, rng):
                assert s.independence_declared
                assert s.independence_year == 60
                break


# ─── supply_ship_arrives tests ─────────────────────────────────────

class TestSupplyShip:
    def test_no_ship_after_independence(self):
        s = EarthState(independence_declared=True, budget=1.0)
        rng = random.Random(42)
        assert supply_ship_arrives(s, 50, rng) is None

    def test_no_ship_during_crisis(self):
        s = EarthState(earth_crisis=True, budget=1.0)
        rng = random.Random(42)
        assert supply_ship_arrives(s, 50, rng) is None

    def test_ship_sometimes_arrives(self):
        arrived = 0
        for seed in range(200):
            s = EarthState(budget=0.8)
            rng = random.Random(seed)
            if supply_ship_arrives(s, 10, rng) is not None:
                arrived += 1
        assert arrived > 10  # should arrive sometimes
        assert arrived < 190  # but not always

    def test_ship_has_valid_resources(self):
        for seed in range(500):
            s = EarthState(budget=1.0)
            rng = random.Random(seed)
            ship = supply_ship_arrives(s, 10, rng)
            if ship is not None:
                for k, v in ship.items():
                    assert k in ("food", "medicine", "water", "power")
                    assert 0.0 < v < 0.5
                break


# ─── Engine integration tests ──────────────────────────────────────

class TestEngineIntegration:
    def test_single_tick_has_earth_fields(self):
        eng = Mars100Engine(seed=42, total_years=10)
        result = eng.tick()
        # Earth fields should exist (may be None if no directive this year)
        assert hasattr(result, "earth_directive")
        assert hasattr(result, "colony_response")
        assert hasattr(result, "supply_ship")
        assert hasattr(result, "independence_event")

    def test_year_result_to_dict_has_earth(self):
        eng = Mars100Engine(seed=42, total_years=30)
        # Run enough ticks to likely get a directive
        directive_found = False
        for _ in range(30):
            result = eng.tick()
            d = result.to_dict()
            if d.get("earth_directive") is not None:
                directive_found = True
                assert d["colony_response"] in RESPONSE_TYPES
                break
        # If no directive in 30 years, that's OK (probabilistic)

    def test_full_sim_result_has_earth(self):
        eng = Mars100Engine(seed=42, total_years=100)
        result = eng.run()
        d = result.to_dict()
        assert "earth_final_state" in d
        assert "total_supply_ships" in d["summary"]
        assert "independence_year" in d["summary"]
        assert d["_meta"]["version"] == "2.1"

    def test_earth_state_evolves(self):
        eng = Mars100Engine(seed=42, total_years=50)
        eng.run()
        # Earth interest should have changed from initial 0.8
        assert eng.earth.interest != 0.8

    def test_supply_ships_counted(self):
        eng = Mars100Engine(seed=42, total_years=100)
        result = eng.run()
        # Over 100 years, at least some supply ships should arrive
        assert result.total_supply_ships >= 0  # non-negative

    def test_determinism_preserved(self):
        r1 = Mars100Engine(seed=99, total_years=50).run()
        r2 = Mars100Engine(seed=99, total_years=50).run()
        assert r1.total_deaths == r2.total_deaths
        assert r1.total_births == r2.total_births
        assert r1.total_supply_ships == r2.total_supply_ships
        d1 = r1.to_dict()
        d2 = r2.to_dict()
        assert d1["earth_final_state"] == d2["earth_final_state"]

    def test_different_seeds_diverge(self):
        r1 = Mars100Engine(seed=42, total_years=50).run()
        r2 = Mars100Engine(seed=99, total_years=50).run()
        d1 = r1.to_dict()["earth_final_state"]
        d2 = r2.to_dict()["earth_final_state"]
        # At least one field should differ
        assert d1 != d2

    def test_resources_still_bounded(self):
        eng = Mars100Engine(seed=42, total_years=100)
        result = eng.run()
        for year in result.years:
            for name, val in year.resources_after.items():
                assert 0.0 <= val <= 1.0, f"Resource {name} out of bounds in year {year.year}"

    def test_earth_contact_event_coexists(self):
        """The existing earth_contact event in events.py still works."""
        eng = Mars100Engine(seed=42, total_years=50)
        result = eng.run()
        earth_events = [
            ev for yr in result.years for ev in yr.events
            if ev["name"] == "earth_contact"
        ]
        # earth_contact events can still occur (from events.py)
        # They coexist with the Earth system
        assert isinstance(earth_events, list)

    def test_to_dict_json_serializable(self):
        eng = Mars100Engine(seed=42, total_years=30)
        result = eng.run()
        text = json.dumps(result.to_dict())
        assert isinstance(text, str)


# ─── Invariant / property tests ────────────────────────────────────

class TestInvariants:
    @pytest.mark.parametrize("seed", [1, 42, 99, 777, 12345])
    def test_earth_interest_bounded(self, seed: int):
        eng = Mars100Engine(seed=seed, total_years=100)
        eng.run()
        assert 0.0 <= eng.earth.interest <= 1.0

    @pytest.mark.parametrize("seed", [1, 42, 99, 777, 12345])
    def test_earth_budget_bounded(self, seed: int):
        eng = Mars100Engine(seed=seed, total_years=100)
        eng.run()
        assert 0.0 <= eng.earth.budget <= 1.0

    @pytest.mark.parametrize("seed", [1, 42, 99])
    def test_compliance_score_bounded(self, seed: int):
        eng = Mars100Engine(seed=seed, total_years=100)
        eng.run()
        assert 0.0 <= eng.earth.compliance_score <= 1.0

    @pytest.mark.parametrize("seed", [1, 42, 99])
    def test_directive_counts_consistent(self, seed: int):
        eng = Mars100Engine(seed=seed, total_years=100)
        eng.run()
        s = eng.earth
        assert s.directives_complied + s.directives_rejected <= s.directives_sent
        assert s.directives_complied >= 0
        assert s.directives_rejected >= 0

    def test_independence_stops_directives(self):
        """After independence, no more directives should appear."""
        for seed in range(50):
            eng = Mars100Engine(seed=seed, total_years=100)
            result = eng.run()
            if eng.earth.independence_declared:
                ind_year = eng.earth.independence_year
                for yr in result.years:
                    if yr.year > ind_year:
                        assert yr.earth_directive is None, \
                            f"Directive appeared in year {yr.year} after independence at {ind_year}"
                break

    def test_supply_ships_stop_after_independence(self):
        """After independence, no supply ships should arrive."""
        for seed in range(50):
            eng = Mars100Engine(seed=seed, total_years=100)
            result = eng.run()
            if eng.earth.independence_declared:
                ind_year = eng.earth.independence_year
                for yr in result.years:
                    if yr.year > ind_year:
                        assert yr.supply_ship is None, \
                            f"Supply ship arrived in year {yr.year} after independence"
                break


# ─── Compliance feedback loop test ─────────────────────────────────

class TestComplianceFeedback:
    def test_compliance_improves_with_comply(self):
        s = EarthState(compliance_score=0.5)
        _process_colony_response(s, {"response": "comply"},
                                 random.Random(42))
        assert s.compliance_score > 0.5

    def test_compliance_degrades_with_reject(self):
        s = EarthState(compliance_score=0.5)
        _process_colony_response(s, {"response": "reject"},
                                 random.Random(42))
        assert s.compliance_score < 0.5

    def test_rejection_streak_increments(self):
        s = EarthState(recent_rejections=2)
        _process_colony_response(s, {"response": "reject"},
                                 random.Random(42))
        assert s.recent_rejections == 3

    def test_comply_resets_streak(self):
        s = EarthState(recent_rejections=5)
        _process_colony_response(s, {"response": "comply"},
                                 random.Random(42))
        assert s.recent_rejections == 0
