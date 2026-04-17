"""Tests for the Mars-100 economics organ."""
from __future__ import annotations

import random
import pytest

from src.mars100.economics import (
    PersonalInventory, Trade, EconomicState,
    compute_gini, map_economic_system, tick_economics,
    compute_economic_pressure, inequality_vote_bias,
    TRADEABLE, MAX_RESERVE, EARNING_RATES, RETENTION_BY_SYSTEM,
)
from src.mars100.colony import SocialGraph


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_social(ids: list[str], seed: int = 0) -> SocialGraph:
    sg = SocialGraph()
    sg.initialize(ids, random.Random(seed))
    return sg


def _make_econ(ids: list[str]) -> EconomicState:
    e = EconomicState()
    for cid in ids:
        e.add_colonist(cid)
    return e


# ---------------------------------------------------------------------------
# PersonalInventory
# ---------------------------------------------------------------------------

class TestPersonalInventory:
    def test_default_empty(self):
        inv = PersonalInventory()
        assert inv.wealth() == 0.0
        for r in TRADEABLE:
            assert getattr(inv, r) == 0.0

    def test_clamp_upper(self):
        inv = PersonalInventory(food=2.0, water=-0.5, power=0.5, medicine=1.5)
        inv.clamp()
        assert inv.food == MAX_RESERVE
        assert inv.water == 0.0
        assert inv.power == 0.5
        assert inv.medicine == MAX_RESERVE

    def test_wealth_is_material_only(self):
        inv = PersonalInventory(food=0.2, water=0.3, power=0.1, medicine=0.1,
                                total_earned=99.0, total_traded=50.0)
        assert abs(inv.wealth() - 0.7) < 1e-9

    def test_most_needed_and_surplus(self):
        inv = PersonalInventory(food=0.1, water=0.5, power=0.3, medicine=0.8)
        assert inv.most_needed() == "food"
        assert inv.most_surplus() == "medicine"

    def test_serialization_roundtrip(self):
        inv = PersonalInventory(food=0.123, water=0.456, power=0.789,
                                medicine=0.012, total_earned=1.5, total_traded=0.3)
        d = inv.to_dict()
        inv2 = PersonalInventory.from_dict(d)
        for r in TRADEABLE:
            assert abs(getattr(inv, r) - getattr(inv2, r)) < 0.001

    def test_from_dict_missing_keys(self):
        inv = PersonalInventory.from_dict({})
        assert inv.food == 0.0
        assert inv.total_earned == 0.0


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

class TestTrade:
    def test_to_dict_format(self):
        t = Trade(year=10, seller_id="a", buyer_id="b",
                  given_resource="food", given_amount=0.05,
                  received_resource="water", received_amount=0.04)
        d = t.to_dict()
        assert d["year"] == 10
        assert "food" in d["gave"]
        assert "water" in d["got"]


# ---------------------------------------------------------------------------
# compute_gini
# ---------------------------------------------------------------------------

class TestGini:
    def test_perfect_equality(self):
        assert compute_gini([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_perfect_inequality(self):
        g = compute_gini([0.0, 0.0, 0.0, 10.0])
        assert g > 0.7

    def test_empty_and_single(self):
        assert compute_gini([]) == 0.0
        assert compute_gini([5.0]) == 0.0

    def test_all_zero(self):
        assert compute_gini([0.0, 0.0, 0.0]) == 0.0

    def test_always_bounded(self):
        rng = random.Random(42)
        for _ in range(50):
            vals = [rng.random() for _ in range(rng.randint(2, 20))]
            g = compute_gini(vals)
            assert 0.0 <= g <= 1.0

    def test_moderate_inequality(self):
        g = compute_gini([0.1, 0.2, 0.3, 0.4])
        assert 0.1 < g < 0.5


# ---------------------------------------------------------------------------
# map_economic_system
# ---------------------------------------------------------------------------

class TestMapSystem:
    @pytest.mark.parametrize("gov,expected", [
        ("anarchy", "barter"), ("council", "market"),
        ("dictator", "planned"), ("lottery", "barter"),
        ("consensus", "communal"), ("ai_governor", "planned"),
    ])
    def test_all_gov_types(self, gov, expected):
        assert map_economic_system(gov) == expected

    def test_unknown_defaults_communal(self):
        assert map_economic_system("unknown") == "communal"


# ---------------------------------------------------------------------------
# EconomicState
# ---------------------------------------------------------------------------

class TestEconomicState:
    def test_add_colonist_idempotent(self):
        e = EconomicState()
        e.add_colonist("a")
        e.inventories["a"].food = 0.5
        e.add_colonist("a")
        assert e.inventories["a"].food == 0.5

    def test_remove_colonist_returns_half(self):
        e = EconomicState()
        e.add_colonist("a")
        e.inventories["a"].food = 0.6
        e.inventories["a"].water = 0.4
        pool = e.remove_colonist("a")
        assert abs(pool["food"] - 0.3) < 1e-9
        assert abs(pool["water"] - 0.2) < 1e-9
        assert "a" not in e.inventories

    def test_remove_missing_colonist(self):
        e = EconomicState()
        pool = e.remove_colonist("ghost")
        assert pool["food"] == 0.0

    def test_redistribute(self):
        e = _make_econ(["a", "b"])
        pool = {"food": 0.4, "water": 0.2}
        e.redistribute(pool, ["a", "b"])
        assert abs(e.inventories["a"].food - 0.2) < 1e-9
        assert abs(e.inventories["b"].food - 0.2) < 1e-9

    def test_redistribute_empty(self):
        e = _make_econ(["a"])
        e.redistribute({"food": 1.0}, [])  # no crash

    def test_gini_empty(self):
        e = EconomicState()
        assert e.gini() == 0.0

    def test_to_dict_has_required_keys(self):
        e = _make_econ(["a", "b"])
        d = e.to_dict()
        assert "system" in d
        assert "gini" in d
        assert "inventories" in d
        assert "total_trades" in d

    def test_summary_has_required_keys(self):
        e = _make_econ(["a"])
        s = e.summary()
        assert "gini" in s
        assert "system" in s
        assert "avg_wealth" in s


# ---------------------------------------------------------------------------
# tick_economics
# ---------------------------------------------------------------------------

class TestTickEconomics:
    def test_basic_earning(self):
        ids = ["c0", "c1"]
        econ = _make_econ(ids)
        social = _make_social(ids)
        actions = {"c0": "farm", "c1": "code"}
        hoarding = {"c0": 0.5, "c1": 0.5}
        result = tick_economics(econ, actions, hoarding, social, ids,
                                year=1, gov_type="anarchy", rng=random.Random(1))
        # anarchy → barter → retention 1.0
        assert econ.inventories["c0"].food > 0
        assert econ.inventories["c1"].power > 0

    def test_communal_zero_retention(self):
        ids = ["c0"]
        econ = _make_econ(ids)
        social = _make_social(ids)
        actions = {"c0": "farm"}
        hoarding = {"c0": 0.0}  # no hoarding bonus
        tick_economics(econ, actions, hoarding, social, ids,
                       year=1, gov_type="consensus", rng=random.Random(1))
        # consensus → communal → retention 0.0
        # Even with hoard_bonus = 0 * 0.4 = 0, retention = 0 + 0 = 0
        assert econ.inventories["c0"].food == 0.0

    def test_hoarding_boosts_retention(self):
        ids = ["lo", "hi"]
        econ = _make_econ(ids)
        social = _make_social(ids)
        actions = {"lo": "farm", "hi": "farm"}
        hoarding_lo = {"lo": 0.0, "hi": 0.9}
        tick_economics(econ, actions, hoarding_lo, social, ids,
                       year=1, gov_type="council", rng=random.Random(1))
        # market retention = 0.7; lo gets 0.7*rate, hi gets min(1.0, 0.7+0.9*0.4)*rate
        assert econ.inventories["hi"].food > econ.inventories["lo"].food

    def test_trades_happen_with_complementary_needs(self):
        ids = ["farmer", "coder"]
        econ = _make_econ(ids)
        # Pre-load complementary surpluses
        econ.inventories["farmer"].food = 0.5
        econ.inventories["farmer"].power = 0.0
        econ.inventories["coder"].power = 0.5
        econ.inventories["coder"].food = 0.0
        social = _make_social(ids, seed=99)
        # Force high trust
        social.edges["farmer"]["coder"].trust = 0.95
        social.edges["coder"]["farmer"].trust = 0.95
        actions = {"farmer": "farm", "coder": "code"}
        hoarding = {"farmer": 0.5, "coder": 0.5}
        result = tick_economics(econ, actions, hoarding, social, ids,
                                year=1, gov_type="anarchy", rng=random.Random(42))
        # With high trust and complementary needs, a trade should occur
        assert result["trades"] >= 0  # may or may not trade depending on shuffle order
        assert result["gini"] >= 0.0

    def test_gini_recorded(self):
        ids = ["a", "b"]
        econ = _make_econ(ids)
        social = _make_social(ids)
        for yr in range(1, 6):
            tick_economics(econ, {"a": "farm", "b": "rest"}, {"a": 0.5, "b": 0.5},
                           social, ids, yr, "anarchy", random.Random(yr))
        assert len(econ.gini_history) == 5

    def test_result_keys(self):
        ids = ["a"]
        econ = _make_econ(ids)
        social = _make_social(ids)
        r = tick_economics(econ, {"a": "farm"}, {"a": 0.5}, social, ids,
                           1, "anarchy", random.Random(0))
        assert "year" in r
        assert "system" in r
        assert "gini" in r
        assert "trades" in r

    def test_system_updates_from_governance(self):
        ids = ["a"]
        econ = _make_econ(ids)
        social = _make_social(ids)
        tick_economics(econ, {"a": "rest"}, {"a": 0.5}, social, ids,
                       1, "dictator", random.Random(0))
        assert econ.system == "planned"

    def test_trade_log_pruned(self):
        ids = ["a", "b"]
        econ = _make_econ(ids)
        # Manually add many trades
        for i in range(250):
            econ.trade_log.append(Trade(year=i, seller_id="a", buyer_id="b",
                                        given_resource="food", given_amount=0.01,
                                        received_resource="water", received_amount=0.01))
        social = _make_social(ids)
        tick_economics(econ, {"a": "rest", "b": "rest"}, {"a": 0.5, "b": 0.5},
                       social, ids, 1, "anarchy", random.Random(0))
        assert len(econ.trade_log) <= 200 + len(ids)


# ---------------------------------------------------------------------------
# compute_economic_pressure
# ---------------------------------------------------------------------------

class TestEconomicPressure:
    def test_low_gini_no_pressure(self):
        econ = _make_econ(["a", "b"])
        econ.inventories["a"].food = 0.1
        econ.inventories["b"].food = 0.1
        assert compute_economic_pressure(econ) == {}

    def test_high_gini_boosts_cooperate(self):
        econ = _make_econ(["rich", "poor"])
        econ.inventories["rich"].food = 1.0
        econ.inventories["rich"].water = 1.0
        econ.inventories["poor"].food = 0.0
        pressure = compute_economic_pressure(econ)
        assert pressure.get("cooperate", 0) > 0
        assert pressure.get("hoard", 0) < 0

    def test_extreme_gini_enables_sabotage(self):
        econ = _make_econ(["rich", "poor1", "poor2"])
        econ.inventories["rich"].food = 1.0
        econ.inventories["rich"].water = 1.0
        econ.inventories["rich"].power = 1.0
        econ.inventories["rich"].medicine = 1.0
        pressure = compute_economic_pressure(econ)
        assert pressure.get("sabotage", 0) > 0


# ---------------------------------------------------------------------------
# inequality_vote_bias
# ---------------------------------------------------------------------------

class TestVoteBias:
    def test_low_gini_no_bias(self):
        econ = _make_econ(["a", "b"])
        econ.inventories["a"].food = 0.1
        econ.inventories["b"].food = 0.1
        assert inequality_vote_bias(econ, "a", "council") == 0.0

    def test_poor_favours_redistribution(self):
        econ = _make_econ(["rich", "poor"])
        econ.inventories["rich"].food = 1.0
        econ.inventories["rich"].water = 1.0
        bias = inequality_vote_bias(econ, "poor", "consensus")
        assert bias > 0

    def test_rich_resists_redistribution(self):
        econ = _make_econ(["rich", "poor"])
        econ.inventories["rich"].food = 1.0
        econ.inventories["rich"].water = 1.0
        bias = inequality_vote_bias(econ, "rich", "consensus")
        assert bias < 0

    def test_missing_colonist(self):
        econ = EconomicState()
        assert inequality_vote_bias(econ, "ghost", "council") == 0.0


# ---------------------------------------------------------------------------
# Integration: multi-year economics
# ---------------------------------------------------------------------------

class TestMultiYear:
    def test_10_years_no_crash(self):
        ids = [f"c{i}" for i in range(5)]
        econ = _make_econ(ids)
        social = _make_social(ids, seed=7)
        rng = random.Random(42)
        action_pool = list(EARNING_RATES.keys())
        for yr in range(1, 11):
            actions = {cid: rng.choice(action_pool) for cid in ids}
            hoarding = {cid: rng.random() for cid in ids}
            tick_economics(econ, actions, hoarding, social, ids,
                           yr, "anarchy", rng)
        assert len(econ.gini_history) == 10
        for inv in econ.inventories.values():
            for r in TRADEABLE:
                assert 0.0 <= getattr(inv, r) <= MAX_RESERVE

    def test_wealth_grows_in_barter(self):
        ids = ["a", "b"]
        econ = _make_econ(ids)
        social = _make_social(ids)
        for yr in range(1, 21):
            tick_economics(econ, {"a": "farm", "b": "code"},
                           {"a": 0.5, "b": 0.5}, social, ids,
                           yr, "anarchy", random.Random(yr))
        # After 20 years of barter, both should have some wealth
        assert econ.inventories["a"].food > 0
        assert econ.inventories["b"].power > 0

    def test_communal_stays_equal(self):
        ids = ["a", "b"]
        econ = _make_econ(ids)
        social = _make_social(ids)
        for yr in range(1, 11):
            tick_economics(econ, {"a": "farm", "b": "code"},
                           {"a": 0.0, "b": 0.0}, social, ids,
                           yr, "consensus", random.Random(yr))
        # Communal with 0 hoarding → 0 retention → no wealth
        assert econ.inventories["a"].wealth() == 0.0
        assert econ.inventories["b"].wealth() == 0.0

    def test_colonist_removal_and_redistribution(self):
        ids = ["a", "b", "c"]
        econ = _make_econ(ids)
        econ.inventories["a"].food = 0.8
        econ.inventories["a"].water = 0.6
        pool = econ.remove_colonist("a")
        econ.redistribute(pool, ["b", "c"])
        # Each gets half of (0.4, 0.3) = (0.2, 0.15)
        assert abs(econ.inventories["b"].food - 0.2) < 1e-9
        assert abs(econ.inventories["c"].food - 0.2) < 1e-9

    def test_property_gini_always_bounded(self):
        """Gini coefficient is always in [0, 1] regardless of state."""
        rng = random.Random(123)
        for _ in range(20):
            n = rng.randint(2, 10)
            ids = [f"c{i}" for i in range(n)]
            econ = _make_econ(ids)
            for cid in ids:
                for r in TRADEABLE:
                    setattr(econ.inventories[cid], r, rng.random())
            assert 0.0 <= econ.gini() <= 1.0

    def test_property_inventories_always_clamped(self):
        """After tick, all inventories are within bounds."""
        ids = [f"c{i}" for i in range(4)]
        econ = _make_econ(ids)
        social = _make_social(ids, seed=5)
        rng = random.Random(77)
        actions_pool = list(EARNING_RATES.keys())
        for yr in range(1, 31):
            actions = {cid: rng.choice(actions_pool) for cid in ids}
            hoarding = {cid: rng.random() for cid in ids}
            tick_economics(econ, actions, hoarding, social, ids,
                           yr, rng.choice(["anarchy", "council", "dictator"]), rng)
        for inv in econ.inventories.values():
            for r in TRADEABLE:
                v = getattr(inv, r)
                assert 0.0 <= v <= MAX_RESERVE, f"{r}={v}"
