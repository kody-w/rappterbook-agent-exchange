"""Tests for the medicine organ (engine v9.0)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.medicine import (
    HealthState, Disease, ColonistMedContext, MedicineTickResult,
    tick_medicine, disease_death_check, get_top_contacts,
    _apply_aging, _apply_stress_immunity, _trigger_diseases_from_events,
    _trigger_stress_fatigue, _spread_contagion, _resolve_conditions,
    _update_health_from_conditions,
    AGING_BASE_RATE, AGING_ACCELERATION_AGE,
    IMMUNE_STRESS_THRESHOLD, DISEASE_DEATH_HEALTH_THRESHOLD,
    CONTAGION_TOP_N, EPIDEMIC_THRESHOLD, DISEASE_TYPES,
)


def _ctx(
    colonist_id="c0", age=30, action="farm", stress=0.2,
    event_names=None, top_contacts=None, medicine_resource=0.5,
    has_med_bay=False, researchers=0,
):
    return ColonistMedContext(
        colonist_id=colonist_id, age=age, action=action, stress=stress,
        event_names=event_names or [], top_contacts=top_contacts or [],
        medicine_resource=medicine_resource, has_med_bay=has_med_bay,
        researchers=researchers,
    )


class _Rel:
    def __init__(self, trust=0.5):
        self.trust = trust


# --- aging ---

class TestAging:
    def test_reduces_health(self):
        hs = HealthState(health=1.0)
        delta = _apply_aging(hs, 30)
        assert delta < 0 and hs.health < 1.0

    def test_accelerates_after_threshold(self):
        young, old = HealthState(health=1.0), HealthState(health=1.0)
        _apply_aging(young, 30)
        _apply_aging(old, 70)
        assert old.health < young.health

    def test_never_negative(self):
        hs = HealthState(health=0.001)
        _apply_aging(hs, 80)
        assert hs.health >= 0.0


# --- immunity ---

class TestImmunity:
    def test_stress_suppresses(self):
        hs = HealthState(immune_strength=0.7)
        _apply_stress_immunity(hs, 0.9)
        assert hs.immune_strength < 0.7

    def test_low_stress_recovers(self):
        hs = HealthState(immune_strength=0.5)
        _apply_stress_immunity(hs, 0.2)
        assert hs.immune_strength > 0.5

    def test_clamped(self):
        hs = HealthState(immune_strength=0.99)
        _apply_stress_immunity(hs, 0.0)
        assert hs.immune_strength <= 1.0

    def test_never_negative(self):
        hs = HealthState(immune_strength=0.05)
        _apply_stress_immunity(hs, 1.0)
        assert hs.immune_strength >= 0.0


# --- event diseases ---

class TestEventDiseases:
    def test_dust_storm_respiratory(self):
        found = False
        for s in range(100):
            hs = HealthState(immune_strength=0.1)
            ds = _trigger_diseases_from_events(hs, ["dust_storm"], 0.1, random.Random(s))
            if any(d.disease_type == "respiratory" for d in ds):
                found = True
                break
        assert found

    def test_solar_flare_radiation(self):
        found = False
        for s in range(100):
            hs = HealthState(immune_strength=0.1)
            ds = _trigger_diseases_from_events(hs, ["solar_flare"], 0.1, random.Random(s))
            if any(d.disease_type == "radiation_sickness" for d in ds):
                found = True
                break
        assert found

    def test_high_immunity_resists(self):
        count = 0
        for s in range(200):
            hs = HealthState(immune_strength=0.95)
            ds = _trigger_diseases_from_events(hs, ["dust_storm"], 0.95, random.Random(s))
            count += len(ds)
        assert count < 100  # < 50% rate

    def test_no_duplicate_conditions(self):
        hs = HealthState(immune_strength=0.1, conditions=[Disease("respiratory", 0.3, 2)])
        ds = _trigger_diseases_from_events(hs, ["dust_storm"], 0.1, random.Random(1))
        assert not any(d.disease_type == "respiratory" for d in ds)


# --- stress fatigue ---

class TestStressFatigue:
    def test_high_stress_triggers(self):
        found = False
        for s in range(200):
            r = _trigger_stress_fatigue(HealthState(), 0.9, random.Random(s))
            if r and r.disease_type == "chronic_fatigue":
                found = True
                break
        assert found

    def test_low_stress_safe(self):
        for s in range(50):
            assert _trigger_stress_fatigue(HealthState(), 0.3, random.Random(s)) is None


# --- contagion ---

class TestContagion:
    def test_infection_spreads(self):
        found = False
        for s in range(100):
            hm = {"c0": HealthState(conditions=[Disease("infection", 0.8, 3)]),
                   "c1": HealthState(immune_strength=0.1)}
            ctxs = [_ctx("c0", top_contacts=["c1"]), _ctx("c1", top_contacts=["c0"])]
            inf = _spread_contagion(hm, ctxs, {"infection": 1}, random.Random(s))
            if any(cid == "c1" for cid, _ in inf):
                found = True
                break
        assert found

    def test_non_contagious_blocked(self):
        for s in range(50):
            hm = {"c0": HealthState(conditions=[Disease("injury", 0.8, 2)]),
                   "c1": HealthState()}
            ctxs = [_ctx("c0", top_contacts=["c1"]), _ctx("c1", top_contacts=["c0"])]
            inf = _spread_contagion(hm, ctxs, {"injury": 1}, random.Random(s))
            assert not any(cid == "c1" for cid, _ in inf)

    def test_epidemic_increases_spread(self):
        normal_count = epidemic_count = 0
        for s in range(500):
            hm1 = {"c0": HealthState(conditions=[Disease("infection", 0.5, 3)]),
                    "t": HealthState(immune_strength=0.3)}
            ctxs = [_ctx("c0", top_contacts=["t"]), _ctx(colonist_id="t", top_contacts=["c0"])]
            n = _spread_contagion(hm1, ctxs, {"infection": 1}, random.Random(s))
            normal_count += 1 if any(c == "t" for c, _ in n) else 0
            hm2 = {"c0": HealthState(conditions=[Disease("infection", 0.5, 3)]),
                    "t": HealthState(immune_strength=0.3)}
            e = _spread_contagion(hm2, ctxs, {"infection": 4}, random.Random(s))
            epidemic_count += 1 if any(c == "t" for c, _ in e) else 0
        assert epidemic_count >= normal_count


# --- condition resolution ---

class TestResolution:
    def test_resolves_over_time(self):
        hs = HealthState(conditions=[Disease("infection", 0.3, 1)])
        _resolve_conditions(hs, "rest", False, 0.0, random.Random(42))

    def test_rest_helps(self):
        hs = HealthState(conditions=[Disease("infection", 0.5, 5)])
        orig = hs.conditions[0].severity
        _resolve_conditions(hs, "rest", False, 0.0, random.Random(99))
        if hs.conditions:
            assert hs.conditions[0].severity < orig

    def test_med_bay_helps(self):
        hs1 = HealthState(conditions=[Disease("infection", 0.5, 5)])
        hs2 = HealthState(conditions=[Disease("infection", 0.5, 5)])
        _resolve_conditions(hs1, "farm", False, 0.0, random.Random(99))
        _resolve_conditions(hs2, "farm", True, 0.0, random.Random(99))
        if hs1.conditions and hs2.conditions:
            assert hs2.conditions[0].severity <= hs1.conditions[0].severity


# --- health from conditions ---

class TestHealthUpdate:
    def test_no_conditions_recovers(self):
        hs = HealthState(health=0.8)
        _update_health_from_conditions(hs)
        assert hs.health > 0.8

    def test_conditions_reduce(self):
        hs = HealthState(health=0.8, conditions=[Disease("infection", 0.5, 3)])
        _update_health_from_conditions(hs)
        assert hs.health < 0.8

    def test_never_negative(self):
        hs = HealthState(health=0.01, conditions=[
            Disease("infection", 0.9, 3), Disease("respiratory", 0.8, 2)])
        _update_health_from_conditions(hs)
        assert hs.health >= 0.0


# --- disease death ---

class TestDiseaseDeath:
    def test_healthy_safe(self):
        hs = HealthState(health=0.8, conditions=[Disease("infection", 0.3, 2)])
        for s in range(50):
            assert disease_death_check(hs, 0.5, random.Random(s)) is None

    def test_sick_can_die(self):
        found = False
        for s in range(500):
            hs = HealthState(health=0.05, conditions=[Disease("infection", 0.9, 3)])
            if disease_death_check(hs, 0.05, random.Random(s)):
                found = True
                break
        assert found

    def test_no_conditions_safe(self):
        hs = HealthState(health=0.05)
        for s in range(100):
            assert disease_death_check(hs, 0.05, random.Random(s)) is None


# --- top contacts ---

class TestTopContacts:
    def test_returns_top_n(self):
        edges = {"c0": {"c1": _Rel(0.9), "c2": _Rel(0.5), "c3": _Rel(0.8), "c4": _Rel(0.3)}}
        result = get_top_contacts("c0", ["c1", "c2", "c3", "c4"], edges, n=3)
        assert len(result) == 3
        assert result[0] == "c1"

    def test_empty(self):
        assert get_top_contacts("c0", ["c1"], {}, n=3) == []


# --- serialization ---

class TestSerialization:
    def test_health_state_roundtrip(self):
        hs = HealthState(health=0.75, immune_strength=0.6,
                         conditions=[Disease("infection", 0.4, 3, True)],
                         medical_history=["respiratory"])
        d = hs.to_dict()
        assert d["health"] == 0.75
        hs2 = HealthState.from_dict(d)
        assert abs(hs2.health - 0.75) < 0.001

    def test_disease_roundtrip(self):
        d = Disease("respiratory", 0.3, 2, True)
        d2 = Disease.from_dict(d.to_dict())
        assert d2.disease_type == "respiratory" and d2.chronic

    def test_result_to_dict(self):
        r = MedicineTickResult([], [], [], [], 0.8, 0.6, 0)
        assert r.to_dict()["avg_health"] == 0.8


# --- tick_medicine integration ---

class TestTickMedicine:
    def test_smoke_single(self):
        hm = {}
        r = tick_medicine(hm, [_ctx("c0")], 1, random.Random(42))
        assert isinstance(r, MedicineTickResult) and "c0" in hm

    def test_smoke_ten_colonists_ten_years(self):
        hm = {}
        for y in range(1, 11):
            ctxs = [_ctx(f"c{i}", age=y + 25, stress=0.3) for i in range(10)]
            tick_medicine(hm, ctxs, y, random.Random(42 + y))
        assert len(hm) == 10
        for hs in hm.values():
            assert 0.0 <= hs.health <= 1.0
            assert 0.0 <= hs.immune_strength <= 1.0

    def test_events_produce_diseases(self):
        found = False
        for s in range(50):
            hm = {"c0": HealthState(immune_strength=0.05)}
            r = tick_medicine(hm, [_ctx("c0", age=40, stress=0.8,
                                        event_names=["dust_storm", "epidemic"])],
                              1, random.Random(s))
            if r.new_diseases:
                found = True
                break
        assert found

    def test_bounds_invariant(self):
        hm = {}
        rng = random.Random(123)
        evts = ["dust_storm", "epidemic", "solar_flare", "equipment_failure"]
        for y in range(1, 51):
            ctxs = [_ctx(f"c{i}", age=y + 20, stress=rng.random(),
                         event_names=[rng.choice(evts)],
                         top_contacts=[f"c{(i+1)%5}"],
                         medicine_resource=rng.random()) for i in range(5)]
            tick_medicine(hm, ctxs, y, rng)
            for cid, hs in hm.items():
                assert 0.0 <= hs.health <= 1.0
                assert 0.0 <= hs.immune_strength <= 1.0

    def test_researchers_help(self):
        r0 = r1 = 0
        for s in range(200):
            hm = {"c0": HealthState(conditions=[Disease("infection", 0.3, 2)])}
            a = tick_medicine(hm, [_ctx("c0", researchers=0)], 1, random.Random(s))
            r0 += len(a.recovered)
            hm2 = {"c0": HealthState(conditions=[Disease("infection", 0.3, 2)])}
            b = tick_medicine(hm2, [_ctx("c0", researchers=5)], 1, random.Random(s))
            r1 += len(b.recovered)
        assert r1 >= r0


# --- engine integration ---

class TestEngineIntegration:
    def test_100_years_v9(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        assert result.to_dict()["_meta"]["version"] == "9.0"
        assert len(result.years) > 0
        assert len(engine.health_map) > 0

    def test_medicine_in_year_result(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=5)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "medicine" in d
            assert "avg_health" in d["medicine"]

    def test_disease_deaths_possible(self):
        """Disease death is possible when health is critically low."""
        hs = HealthState(health=0.05, conditions=[Disease("infection", 0.9, 5)])
        found = False
        for s in range(500):
            cause = disease_death_check(hs, 0.05, random.Random(s))
            if cause:
                assert cause == "sepsis"
                found = True
                break
        assert found, "Disease death must be possible with critical health"

    def test_psych_perturbation(self):
        from src.mars100.engine import Mars100Engine
        from src.mars100.psychology import PsychState
        engine = Mars100Engine(seed=42, total_years=1)
        for c in engine.colonists:
            engine.psych_map[c.id] = PsychState(stress=0.9, purpose=0.1, loneliness=0.8)
        result = engine.tick()
        rest_pray = sum(1 for a in result.actions.values() if a in ("rest", "pray"))
        assert rest_pray >= 1

    def test_health_perturbation(self):
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=1)
        for c in engine.colonists:
            engine.health_map[c.id] = HealthState(health=0.2)
        result = engine.tick()
        rest = sum(1 for a in result.actions.values() if a == "rest")
        assert rest >= 1
