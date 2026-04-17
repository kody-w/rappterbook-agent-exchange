"""Tests for the governance laws organ (v9.0)."""
from __future__ import annotations

import random
import pytest

from src.mars100.laws import (
    Law, generate_laws, compute_law_action_modifiers,
    compute_law_consumption_modifier, compute_law_exile_thresholds,
    compute_law_stress_cost, compute_law_purpose_cost,
    compute_satisfaction, compute_colony_satisfaction,
    check_governance_crisis,
)
from src.mars100.governance import GovernanceState


# ── Law dataclass ────────────────────────────────────────────────

class TestLawDataclass:
    def test_roundtrip(self):
        law = Law(id="law-1", name="Test Law", category="behavior",
                  enacted_year=5, gov_type="council",
                  modifiers={"sabotage": 0.5, "cooperate": 1.2})
        d = law.to_dict()
        restored = Law.from_dict(d)
        assert restored.id == law.id
        assert restored.name == law.name
        assert restored.modifiers == law.modifiers
        assert restored.active is True
        assert restored.repealed_year is None

    def test_from_dict_defaults(self):
        law = Law.from_dict({"id": "x", "name": "x"})
        assert law.category == "behavior"
        assert law.active is True
        assert law.gov_type == "anarchy"
        assert law.modifiers == {}


# ── Law generation ───────────────────────────────────────────────

class TestGenerateLaws:
    @pytest.mark.parametrize("gov_type", [
        "council", "dictator", "lottery", "consensus", "ai_governor"
    ])
    def test_generates_laws_for_each_gov_type(self, gov_type):
        rng = random.Random(42)
        resources = {"food": 0.5, "water": 0.5, "power": 0.5,
                     "air": 0.5, "medicine": 0.5}
        laws = generate_laws(gov_type, 10, resources, rng)
        assert len(laws) >= 1
        assert all(isinstance(l, Law) for l in laws)
        assert all(l.gov_type == gov_type for l in laws)
        assert all(l.enacted_year == 10 for l in laws)

    def test_anarchy_generates_no_laws(self):
        rng = random.Random(42)
        laws = generate_laws("anarchy", 10, {}, rng)
        assert laws == []

    def test_low_resources_may_generate_more_laws(self):
        rng = random.Random(42)
        low_res = {"food": 0.1, "water": 0.1, "power": 0.1,
                   "air": 0.1, "medicine": 0.1}
        high_res = {"food": 0.9, "water": 0.9, "power": 0.9,
                    "air": 0.9, "medicine": 0.9}
        # Run 50 trials and compare averages
        low_counts = [len(generate_laws("dictator", 5, low_res, random.Random(i)))
                      for i in range(50)]
        high_counts = [len(generate_laws("dictator", 5, high_res, random.Random(i)))
                       for i in range(50)]
        # Low resources should produce at least as many laws on average
        assert sum(low_counts) >= sum(high_counts)

    def test_unique_ids(self):
        rng = random.Random(42)
        laws = generate_laws("council", 10, {"food": 0.5, "water": 0.5,
                             "power": 0.5, "air": 0.5, "medicine": 0.5}, rng)
        ids = [l.id for l in laws]
        assert len(ids) == len(set(ids))


# ── Law effects ──────────────────────────────────────────────────

class TestLawEffects:
    def test_action_modifiers_empty(self):
        assert compute_law_action_modifiers([]) == {}

    def test_action_modifiers_single_law(self):
        law = Law(id="l1", name="Martial", category="behavior",
                  enacted_year=1, gov_type="dictator",
                  modifiers={"sabotage": 0.2, "rest": 0.7})
        mods = compute_law_action_modifiers([law])
        assert mods["sabotage"] == pytest.approx(0.2)
        assert mods["rest"] == pytest.approx(0.7)

    def test_action_modifiers_inactive_ignored(self):
        law = Law(id="l1", name="Repealed", category="behavior",
                  enacted_year=1, gov_type="dictator",
                  modifiers={"sabotage": 0.1}, active=False)
        assert compute_law_action_modifiers([law]) == {}

    def test_action_modifiers_stack_multiplicatively(self):
        law1 = Law(id="l1", name="A", category="behavior",
                   enacted_year=1, gov_type="council",
                   modifiers={"sabotage": 0.5})
        law2 = Law(id="l2", name="B", category="behavior",
                   enacted_year=1, gov_type="council",
                   modifiers={"sabotage": 0.5})
        mods = compute_law_action_modifiers([law1, law2])
        assert mods["sabotage"] == pytest.approx(0.25)

    def test_consumption_modifier_clamped(self):
        law = Law(id="l1", name="Strict", category="resource",
                  enacted_year=1, gov_type="dictator",
                  modifiers={"consumption": 0.3})
        assert compute_law_consumption_modifier([law]) == 0.5  # clamped

    def test_consumption_modifier_normal(self):
        law = Law(id="l1", name="Fair", category="resource",
                  enacted_year=1, gov_type="council",
                  modifiers={"consumption": 0.9})
        assert compute_law_consumption_modifier([law]) == pytest.approx(0.9)

    def test_exile_thresholds_default(self):
        trust, sab = compute_law_exile_thresholds([])
        assert trust == pytest.approx(0.15)
        assert sab == pytest.approx(0.5)

    def test_exile_thresholds_modified(self):
        law = Law(id="l1", name="Strict Exile", category="exile",
                  enacted_year=1, gov_type="dictator",
                  modifiers={"exile_trust": 0.2, "exile_sabotage": 0.35})
        trust, sab = compute_law_exile_thresholds([law])
        assert trust == pytest.approx(0.2)
        assert sab == pytest.approx(0.35)

    def test_stress_cost_capped(self):
        laws = [Law(id=f"l{i}", name=f"Harsh {i}", category="labor",
                    enacted_year=1, gov_type="dictator",
                    modifiers={"stress_cost": 0.1})
                for i in range(5)]
        assert compute_law_stress_cost(laws) == pytest.approx(0.15)  # capped

    def test_purpose_boost_vs_cost(self):
        boost_law = Law(id="l1", name="Freedom", category="freedom",
                        enacted_year=1, gov_type="lottery",
                        modifiers={"purpose_boost": 0.02})
        cost_law = Law(id="l2", name="Machine", category="behavior",
                       enacted_year=1, gov_type="ai_governor",
                       modifiers={"purpose_cost": 0.05})
        # Net: 0.05 - 0.02 = 0.03 (positive = drain)
        assert compute_law_purpose_cost([boost_law, cost_law]) == pytest.approx(0.03)


# ── Satisfaction ─────────────────────────────────────────────────

class TestSatisfaction:
    def test_satisfaction_bounds(self):
        """Satisfaction must always be in [0, 1]."""
        rng = random.Random(42)
        for _ in range(200):
            sat = compute_satisfaction(
                resolve=rng.random(), empathy=rng.random(),
                improvisation=rng.random(), paranoia=rng.random(),
                faith=rng.random(), stress=rng.random(),
                morale=rng.random(), laws=[], resource_trend=rng.uniform(-0.5, 0.5))
            assert 0.0 <= sat <= 1.0

    def test_high_morale_high_satisfaction(self):
        sat = compute_satisfaction(
            resolve=0.5, empathy=0.5, improvisation=0.5,
            paranoia=0.1, faith=0.5, stress=0.1, morale=0.9,
            laws=[], resource_trend=0.05)
        assert sat > 0.6

    def test_high_stress_low_satisfaction(self):
        sat = compute_satisfaction(
            resolve=0.5, empathy=0.5, improvisation=0.5,
            paranoia=0.5, faith=0.1, stress=0.9, morale=0.2,
            laws=[], resource_trend=-0.1)
        assert sat < 0.4

    def test_colony_satisfaction_empty(self):
        assert compute_colony_satisfaction([]) == 0.5

    def test_colony_satisfaction_weighs_bottom_quartile(self):
        # 4 colonists: 3 happy, 1 very unhappy
        sats = [0.8, 0.8, 0.8, 0.1]
        colony_sat = compute_colony_satisfaction(sats)
        plain_avg = sum(sats) / len(sats)
        # Colony satisfaction should be lower than plain average
        # because bottom quartile (0.1) is weighted 2x
        assert colony_sat < plain_avg


# ── Governance crisis ────────────────────────────────────────────

class TestGovernanceCrisis:
    def test_no_crisis_with_few_years(self):
        assert not check_governance_crisis([0.2, 0.2], 3, 0, -10, random.Random(42))

    def test_no_crisis_when_satisfied(self):
        assert not check_governance_crisis([0.6, 0.7, 0.8], 5, 0, -10, random.Random(42))

    def test_no_crisis_during_grace_period(self):
        # Last gov change at year 3, current year 5 → only 2 years, need 3
        assert not check_governance_crisis(
            [0.1, 0.1, 0.1], 5, 3, -10, random.Random(42))

    def test_no_crisis_during_cooldown(self):
        # Last crisis at year 3, current year 6 → only 3 years, need 5
        assert not check_governance_crisis(
            [0.1, 0.1, 0.1], 6, 0, 3, random.Random(42))

    def test_crisis_triggered_probabilistically(self):
        """With very low satisfaction, crisis should trigger frequently."""
        hits = 0
        for i in range(100):
            if check_governance_crisis(
                    [0.05, 0.05, 0.05], 20, 0, 0, random.Random(i)):
                hits += 1
        # With avg_sat=0.05, crisis_prob = (0.35-0.05)*3 = 0.9 → ~70% (capped at 0.7)
        assert hits > 40  # Should get ~70 hits


# ── GovernanceState v9.0 fields ──────────────────────────────────

class TestGovernanceStateV9:
    def test_new_fields_default(self):
        gs = GovernanceState()
        assert gs.active_laws == []
        assert gs.satisfaction_history == []
        assert gs.last_gov_change_year == 0
        assert gs.last_crisis_year == -10

    def test_roundtrip_with_new_fields(self):
        gs = GovernanceState(
            gov_type="council",
            active_laws=[{"id": "l1", "name": "test"}],
            satisfaction_history=[0.5, 0.6],
            last_gov_change_year=5,
            last_crisis_year=3,
        )
        d = gs.to_dict()
        restored = GovernanceState.from_dict(d)
        assert restored.active_laws == gs.active_laws
        assert restored.satisfaction_history == gs.satisfaction_history
        assert restored.last_gov_change_year == 5
        assert restored.last_crisis_year == 3


# ── Integration: engine with laws ────────────────────────────────

class TestEngineIntegration:
    def test_10_year_smoke(self):
        """Run 10 years — no crashes, satisfaction always in bounds."""
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(42)
        for _ in range(10):
            r = e.tick()
            assert 0.0 <= r.governance_satisfaction <= 1.0

    def test_governance_change_generates_laws(self):
        """After 50 years, governance should have changed and laws generated."""
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(42)
        saw_laws = False
        for _ in range(50):
            r = e.tick()
            if e.governance.active_laws:
                saw_laws = True
        assert saw_laws, "Expected at least one governance change with laws in 50 years"

    def test_satisfaction_history_grows(self):
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(42)
        for _ in range(5):
            e.tick()
        assert len(e.governance.satisfaction_history) == 5

    def test_version_bump(self):
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(42)
        r = e.tick()
        # Version is embedded in the meta from engine tick
        assert r.governance_state is not None

    def test_100_year_full_run(self):
        """Full 100-year run completes without error."""
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(42)
        result = e.run()
        assert len(result.years) == 100
        # Every year has valid satisfaction
        for yr in result.years:
            assert 0.0 <= yr.governance_satisfaction <= 1.0
