"""Tests for Mars-100 economy organ."""
from __future__ import annotations

import random

import pytest

from src.mars100.economy import (
    EconomyState, compute_income, apply_income, apply_taxation,
    resolve_sabotage, resolve_cooperation, apply_wealth_decay,
    redistribute_estates, compute_gini, compute_economic_pressure,
    tick_economy, ACTION_CREDITS, TAX_RATES,
    SABOTAGE_STEAL_FRAC, COOPERATE_SHARE_FRAC,
)


class TestComputeIncome:
    def test_farm_earns_base_credits(self):
        income = compute_income({"c1": "farm"}, {"c1": {"hydroponics": 0.0}})
        assert income["c1"] == ACTION_CREDITS["farm"]

    def test_skill_boosts_income(self):
        income = compute_income({"c1": "farm"}, {"c1": {"hydroponics": 1.0}})
        assert income["c1"] == pytest.approx(ACTION_CREDITS["farm"] * 1.5)

    def test_rest_earns_nothing(self):
        income = compute_income({"c1": "rest"}, {})
        assert income["c1"] == 0.0

    def test_hoard_earns_flat(self):
        income = compute_income({"c1": "hoard"}, {})
        assert income["c1"] == ACTION_CREDITS["hoard"]

    def test_sabotage_earns_nothing(self):
        income = compute_income({"c1": "sabotage"}, {})
        assert income["c1"] == 0.0

    def test_multiple_colonists(self):
        actions = {"c1": "farm", "c2": "code", "c3": "rest"}
        income = compute_income(actions, {})
        assert income["c1"] == ACTION_CREDITS["farm"]
        assert income["c2"] == ACTION_CREDITS["code"]
        assert income["c3"] == 0.0

    def test_unknown_action_earns_zero(self):
        income = compute_income({"c1": "dance"}, {})
        assert income["c1"] == 0.0


class TestApplyIncome:
    def test_adds_to_existing_balance(self):
        econ = EconomyState(credits={"c1": 10.0})
        produced = apply_income(econ, {"c1": 5.0}, {"c1"})
        assert econ.credits["c1"] == 15.0
        assert produced == 5.0

    def test_creates_entry_for_new_colonist(self):
        econ = EconomyState()
        apply_income(econ, {"c1": 3.0}, {"c1"})
        assert econ.credits["c1"] == 3.0

    def test_ignores_non_active(self):
        econ = EconomyState()
        apply_income(econ, {"dead": 5.0}, {"c1"})
        assert "dead" not in econ.credits


class TestApplyTaxation:
    def test_anarchy_no_tax(self):
        econ = EconomyState(credits={"c1": 100.0})
        taxed = apply_taxation(econ, "anarchy", None)
        assert taxed == 0.0
        assert econ.credits["c1"] == 100.0

    def test_council_rate(self):
        econ = EconomyState(credits={"c1": 100.0})
        taxed = apply_taxation(econ, "council", None)
        expected_tax = 100.0 * TAX_RATES["council"]
        assert taxed == pytest.approx(expected_tax)
        assert econ.credits["c1"] == pytest.approx(100.0 - expected_tax)
        assert econ.colony_fund == pytest.approx(expected_tax)

    def test_dictator_gets_half(self):
        econ = EconomyState(credits={"c1": 100.0, "dictator": 0.0})
        taxed = apply_taxation(econ, "dictator", "dictator")
        assert taxed > 0
        assert econ.credits["dictator"] > 0
        assert econ.colony_fund > 0
        assert econ.credits["dictator"] == pytest.approx(taxed * 0.5, abs=0.01)
        assert econ.colony_fund == pytest.approx(taxed * 0.5, abs=0.01)

    def test_zero_balance_not_taxed(self):
        econ = EconomyState(credits={"c1": 0.0})
        taxed = apply_taxation(econ, "council", None)
        assert taxed == 0.0

    def test_negative_balance_not_taxed(self):
        econ = EconomyState(credits={"c1": -5.0})
        taxed = apply_taxation(econ, "council", None)
        assert taxed == 0.0
        assert econ.credits["c1"] == -5.0

    def test_tax_accumulates_in_total(self):
        econ = EconomyState(credits={"c1": 100.0})
        apply_taxation(econ, "council", None)
        assert econ.total_taxed > 0


class TestResolveSabotage:
    def test_saboteur_steals(self):
        rng = random.Random(42)
        econ = EconomyState(credits={"c1": 0.0, "c2": 100.0})
        thefts = resolve_sabotage(econ, {"c1": "sabotage"}, ["c1", "c2"], rng)
        assert len(thefts) == 1
        assert econ.credits["c1"] > 0
        assert econ.credits["c2"] < 100.0

    def test_no_theft_from_empty(self):
        rng = random.Random(42)
        econ = EconomyState(credits={"c1": 0.0, "c2": 0.0})
        thefts = resolve_sabotage(econ, {"c1": "sabotage"}, ["c1", "c2"], rng)
        assert len(thefts) == 0

    def test_conservation_during_theft(self):
        rng = random.Random(42)
        econ = EconomyState(credits={"c1": 0.0, "c2": 100.0})
        before = sum(econ.credits.values())
        resolve_sabotage(econ, {"c1": "sabotage"}, ["c1", "c2"], rng)
        after = sum(econ.credits.values())
        assert before == pytest.approx(after)

    def test_non_saboteur_ignored(self):
        rng = random.Random(42)
        econ = EconomyState(credits={"c1": 0.0, "c2": 100.0})
        thefts = resolve_sabotage(econ, {"c1": "farm"}, ["c1", "c2"], rng)
        assert len(thefts) == 0

    def test_single_colonist_no_theft(self):
        rng = random.Random(42)
        econ = EconomyState(credits={"c1": 100.0})
        thefts = resolve_sabotage(econ, {"c1": "sabotage"}, ["c1"], rng)
        assert len(thefts) == 0


class TestResolveCooperation:
    def test_cooperator_shares(self):
        rng = random.Random(42)
        econ = EconomyState(credits={"c1": 100.0, "c2": 0.0})
        trust = {"c1": {"c2": 0.8}, "c2": {"c1": 0.8}}
        transfers = resolve_cooperation(econ, {"c1": "cooperate"},
                                        trust, ["c1", "c2"], rng)
        assert len(transfers) == 1
        assert econ.credits["c1"] < 100.0
        assert econ.credits["c2"] > 0.0

    def test_conservation_during_sharing(self):
        econ = EconomyState(credits={"c1": 100.0, "c2": 50.0})
        trust = {"c1": {"c2": 0.9}, "c2": {"c1": 0.9}}
        before = sum(econ.credits.values())
        resolve_cooperation(econ, {"c1": "cooperate"},
                            trust, ["c1", "c2"], random.Random(42))
        after = sum(econ.credits.values())
        assert before == pytest.approx(after)

    def test_no_share_below_trust_threshold(self):
        rng = random.Random(42)
        econ = EconomyState(credits={"c1": 100.0, "c2": 0.0})
        trust = {"c1": {"c2": 0.2}, "c2": {"c1": 0.2}}
        transfers = resolve_cooperation(econ, {"c1": "cooperate"},
                                        trust, ["c1", "c2"], rng)
        assert len(transfers) == 0

    def test_no_share_with_zero_balance(self):
        rng = random.Random(42)
        econ = EconomyState(credits={"c1": 0.0, "c2": 100.0})
        trust = {"c1": {"c2": 0.9}}
        transfers = resolve_cooperation(econ, {"c1": "cooperate"},
                                        trust, ["c1", "c2"], rng)
        assert len(transfers) == 0

    def test_non_cooperator_ignored(self):
        rng = random.Random(42)
        econ = EconomyState(credits={"c1": 100.0, "c2": 0.0})
        trust = {"c1": {"c2": 0.9}}
        transfers = resolve_cooperation(econ, {"c1": "farm"},
                                        trust, ["c1", "c2"], rng)
        assert len(transfers) == 0


class TestComputeGini:
    def test_perfect_equality(self):
        assert compute_gini([10, 10, 10, 10]) == 0.0

    def test_high_inequality(self):
        gini = compute_gini([0, 0, 0, 100])
        assert gini > 0.7

    def test_single_person(self):
        assert compute_gini([100]) == 0.0

    def test_empty(self):
        assert compute_gini([]) == 0.0

    def test_all_zero(self):
        assert compute_gini([0, 0, 0]) == 0.0

    def test_result_in_range(self):
        rng = random.Random(42)
        values = [rng.random() * 100 for _ in range(20)]
        gini = compute_gini(values)
        assert 0.0 <= gini <= 1.0

    def test_two_values(self):
        gini = compute_gini([0, 100])
        assert gini == pytest.approx(0.5)

    def test_negative_values_clamped(self):
        gini = compute_gini([-10, 0, 50])
        assert 0.0 <= gini <= 1.0


class TestWealthDecay:
    def test_decay_reduces_wealth(self):
        econ = EconomyState(credits={"c1": 100.0})
        apply_wealth_decay(econ)
        assert econ.credits["c1"] < 100.0
        assert econ.credits["c1"] > 0.0

    def test_decay_is_multiplicative(self):
        econ = EconomyState(credits={"c1": 100.0, "c2": 50.0})
        apply_wealth_decay(econ)
        assert econ.credits["c1"] / econ.credits["c2"] == pytest.approx(2.0)


class TestRedistributeEstates:
    def test_dead_colonist_wealth_to_fund(self):
        econ = EconomyState(credits={"c1": 50.0, "c2": 30.0})
        redistributed = redistribute_estates(econ, {"c1"})
        assert "c2" not in econ.credits
        assert econ.colony_fund >= 30.0
        assert redistributed >= 30.0

    def test_no_redistribution_when_all_alive(self):
        econ = EconomyState(credits={"c1": 50.0})
        redistributed = redistribute_estates(econ, {"c1"})
        assert redistributed == 0.0

    def test_negative_wealth_not_redistributed(self):
        econ = EconomyState(credits={"dead": -10.0})
        old_fund = econ.colony_fund
        redistributed = redistribute_estates(econ, set())
        assert redistributed == 0.0
        assert econ.colony_fund == old_fund


class TestComputeEconomicPressure:
    def test_low_inequality_no_pressure(self):
        econ = EconomyState(gini_history=[0.2])
        pressure = compute_economic_pressure(econ)
        assert "cooperate" not in pressure
        assert "sabotage" not in pressure

    def test_high_inequality_encourages_cooperation(self):
        econ = EconomyState(gini_history=[0.7])
        pressure = compute_economic_pressure(econ)
        assert pressure.get("cooperate", 0) > 0
        assert pressure.get("mediate", 0) > 0

    def test_extreme_inequality_enables_sabotage(self):
        econ = EconomyState(gini_history=[0.8])
        pressure = compute_economic_pressure(econ)
        assert pressure.get("sabotage", 0) > 0

    def test_high_colony_fund_enables_rest(self):
        econ = EconomyState(gini_history=[0.2], colony_fund=100.0)
        pressure = compute_economic_pressure(econ)
        assert pressure.get("rest", 0) > 0

    def test_empty_history_no_crash(self):
        econ = EconomyState()
        pressure = compute_economic_pressure(econ)
        assert isinstance(pressure, dict)


class TestEconomyStateSerialization:
    def test_to_dict(self):
        econ = EconomyState(credits={"c1": 10.5}, colony_fund=5.0,
                            gini_history=[0.3, 0.4])
        d = econ.to_dict()
        assert d["gini"] == 0.4
        assert d["colony_fund"] == 5.0
        assert len(d["gini_history"]) == 2

    def test_summary_empty(self):
        econ = EconomyState()
        s = econ.summary()
        assert s["gini"] == 0.0
        assert s["active_count"] == 0

    def test_summary_identifies_extremes(self):
        econ = EconomyState(credits={"rich": 100.0, "poor": 1.0},
                            gini_history=[0.5])
        s = econ.summary()
        assert s["wealthiest"] == "rich"
        assert s["poorest"] == "poor"


class TestTickEconomy:
    def test_full_tick_no_crash(self):
        rng = random.Random(42)
        econ = EconomyState()
        actions = {"c1": "farm", "c2": "code", "c3": "cooperate"}
        skills = {"c1": {"hydroponics": 0.5}, "c2": {"coding": 0.8}, "c3": {}}
        trust = {"c1": {"c2": 0.6, "c3": 0.7},
                 "c2": {"c1": 0.5, "c3": 0.4},
                 "c3": {"c1": 0.8, "c2": 0.5}}
        result = tick_economy(econ, actions, skills, trust,
                              ["c1", "c2", "c3"], "council", None, 1, rng)
        assert "gini" in result
        assert result["gini"] >= 0.0
        assert result["produced"] > 0

    def test_deterministic_with_seed(self):
        def run_tick(seed):
            rng = random.Random(seed)
            econ = EconomyState()
            actions = {"c1": "farm", "c2": "sabotage"}
            skills = {}
            trust = {"c1": {"c2": 0.5}, "c2": {"c1": 0.5}}
            return tick_economy(econ, actions, skills, trust,
                                ["c1", "c2"], "anarchy", None, 1, rng)
        assert run_tick(42) == run_tick(42)

    def test_dead_colonist_removed(self):
        econ = EconomyState(credits={"c1": 50.0, "dead": 100.0})
        actions = {"c1": "rest"}
        tick_economy(econ, actions, {}, {}, ["c1"],
                     "anarchy", None, 1, random.Random(42))
        assert "dead" not in econ.credits
        assert econ.colony_fund >= 100.0

    def test_newborn_gets_credits(self):
        econ = EconomyState()
        actions = {"newborn": "farm"}
        skills = {"newborn": {"hydroponics": 0.0}}
        tick_economy(econ, actions, skills, {},
                     ["newborn"], "anarchy", None, 1, random.Random(42))
        assert econ.credits["newborn"] > 0

    def test_gini_history_grows(self):
        econ = EconomyState()
        actions = {"c1": "farm", "c2": "rest"}
        for year in range(5):
            tick_economy(econ, actions, {}, {}, ["c1", "c2"],
                         "anarchy", None, year, random.Random(year))
        assert len(econ.gini_history) == 5

    def test_trade_history_bounded(self):
        econ = EconomyState()
        actions = {"c1": "sabotage", "c2": "farm"}
        for year in range(100):
            econ.credits["c2"] = 100.0
            tick_economy(econ, actions, {}, {}, ["c1", "c2"],
                         "anarchy", None, year, random.Random(year))
        assert len(econ.trade_history) <= 50


class TestPropertyInvariants:
    def test_wealth_non_negative_after_many_ticks(self):
        rng = random.Random(42)
        econ = EconomyState()
        ids = ["c" + str(i) for i in range(5)]
        for year in range(50):
            actions = {cid: rng.choice(["farm", "code", "sabotage",
                                        "hoard", "rest", "cooperate"])
                       for cid in ids}
            skills = {cid: {"hydroponics": 0.5, "coding": 0.5,
                            "terraforming": 0.5, "mediation": 0.5}
                      for cid in ids}
            trust = {a: {b: 0.5 for b in ids if b != a} for a in ids}
            tick_economy(econ, actions, skills, trust, ids,
                         "council", None, year, rng)
        for cid, val in econ.credits.items():
            assert val >= -0.1, cid + " has negative wealth: " + str(val)

    def test_gini_always_in_range(self):
        rng = random.Random(123)
        for _ in range(100):
            n = rng.randint(2, 20)
            values = [rng.expovariate(0.1) for _ in range(n)]
            gini = compute_gini(values)
            assert 0.0 <= gini <= 1.0

    def test_total_credits_conservation(self):
        rng = random.Random(42)
        econ = EconomyState()
        ids = ["c1", "c2", "c3"]
        total_income = 0.0
        for year in range(10):
            actions = {cid: "farm" for cid in ids}
            skills = {}
            trust = {a: {b: 0.5 for b in ids if b != a} for a in ids}
            result = tick_economy(econ, actions, skills, trust, ids,
                                  "anarchy", None, year, rng)
            total_income += result["produced"]
        total_held = sum(econ.credits.values()) + econ.colony_fund
        assert total_held <= total_income + 0.01
        assert total_held > 0
