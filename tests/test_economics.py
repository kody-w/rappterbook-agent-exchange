"""Tests for the Mars-100 economics organ."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.economics import (
    PersonalAccount, TradeRecord, EconomyState,
    RESOURCE_KINDS, MAX_PERSONAL_TOTAL, TAX_RATES,
    allocate_labor_income, allocate_hoard_action,
    collect_taxes, redistribute, execute_trades,
    compute_gini, check_black_market, spend_personal_reserves,
    handle_death, handle_exile, handle_birth, handle_immigrant,
    tick_economy,
)
from src.mars100.colonist import Colonist, create_founding_ten
from src.mars100.colony import Resources, SocialGraph


# ---------------------------------------------------------------------------
# PersonalAccount
# ---------------------------------------------------------------------------

class TestPersonalAccount:
    def test_initial_zero(self):
        a = PersonalAccount()
        assert a.total() == 0.0

    def test_total(self):
        a = PersonalAccount(food=0.05, water=0.03)
        assert abs(a.total() - 0.08) < 1e-9

    def test_clamp_non_negative(self):
        a = PersonalAccount(food=-0.1, water=0.05)
        a.clamp()
        assert a.food == 0.0
        assert a.water == 0.05

    def test_clamp_cap(self):
        a = PersonalAccount(food=0.15, water=0.15, power=0.15, medicine=0.15)
        a.clamp()
        assert abs(a.total() - MAX_PERSONAL_TOTAL) < 1e-9

    def test_to_dict(self):
        a = PersonalAccount(food=0.01)
        d = a.to_dict()
        assert set(d.keys()) == set(RESOURCE_KINDS)
        assert d["food"] == 0.01


# ---------------------------------------------------------------------------
# TradeRecord
# ---------------------------------------------------------------------------

class TestTradeRecord:
    def test_to_dict(self):
        t = TradeRecord(year=5, giver_id="a", receiver_id="b",
                        resource="food", amount=0.01)
        d = t.to_dict()
        assert d["year"] == 5
        assert d["giver"] == "a"
        assert "black_market" not in d

    def test_black_market_flag(self):
        t = TradeRecord(year=1, giver_id="a", receiver_id="b",
                        resource="water", amount=0.01, black_market=True)
        assert t.to_dict()["black_market"] is True


# ---------------------------------------------------------------------------
# EconomyState
# ---------------------------------------------------------------------------

class TestEconomyState:
    def test_get_account_creates(self):
        e = EconomyState()
        a = e.get_account("c-0")
        assert a.total() == 0.0
        assert "c-0" in e.accounts

    def test_get_account_idempotent(self):
        e = EconomyState()
        a1 = e.get_account("c-0")
        a1.food = 0.05
        a2 = e.get_account("c-0")
        assert a2.food == 0.05

    def test_remove_account(self):
        e = EconomyState()
        e.get_account("c-0").food = 0.05
        removed = e.remove_account("c-0")
        assert removed is not None
        assert removed.food == 0.05
        assert "c-0" not in e.accounts

    def test_remove_missing(self):
        e = EconomyState()
        assert e.remove_account("nonexistent") is None

    def test_to_dict(self):
        e = EconomyState()
        e.get_account("c-0").food = 0.01
        e.gini_history.append(0.3)
        d = e.to_dict()
        assert "accounts" in d
        assert d["gini"] == 0.3

    def test_summary(self):
        e = EconomyState()
        e.get_account("c-0").food = 0.05
        e.gini_history.append(0.25)
        s = e.summary()
        assert s["gini"] == 0.25
        assert s["total_wealth"] > 0


# ---------------------------------------------------------------------------
# Labour Income
# ---------------------------------------------------------------------------

class TestLaborIncome:
    def test_farm_produces_food_income(self):
        e = EconomyState()
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        c = colonists[0]
        res = Resources()
        res.food = 0.8
        before = res.food
        allocate_labor_income(e, c, "farm", res, rng)
        assert res.food < before  # colony lost some
        assert e.get_account(c.id).food > 0  # colonist gained

    def test_rest_produces_nothing(self):
        e = EconomyState()
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        c = colonists[0]
        res = Resources()
        ret = allocate_labor_income(e, c, "rest", res, rng)
        assert ret == 0.0

    def test_zero_sum(self):
        """Labour income is deducted from colony — not double-minted."""
        e = EconomyState()
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        c = colonists[0]
        res = Resources()
        total_before = res.food + e.get_account(c.id).food
        allocate_labor_income(e, c, "farm", res, rng)
        total_after = res.food + e.get_account(c.id).food
        assert abs(total_before - total_after) < 1e-9

    def test_hoard_action(self):
        e = EconomyState()
        rng = random.Random(42)
        colonists = create_founding_ten(42)
        c = colonists[0]
        res = Resources()
        taken = allocate_hoard_action(e, c, res, rng)
        assert taken > 0
        assert e.get_account(c.id).total() > 0


# ---------------------------------------------------------------------------
# Taxation
# ---------------------------------------------------------------------------

class TestTaxation:
    def test_anarchy_no_tax(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        e.get_account(colonists[0].id).food = 0.1
        collected = collect_taxes(e, "anarchy", colonists, rng)
        assert collected == 0.0

    def test_council_collects(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        for c in colonists:
            e.get_account(c.id).food = 0.1
        collected = collect_taxes(e, "council", colonists, rng)
        assert collected > 0

    def test_redistribution(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        for c in colonists:
            e.get_account(c.id).food = 0.1
        collect_taxes(e, "council", colonists, rng)
        distributed = redistribute(e, "council", colonists, None)
        assert distributed > 0

    def test_dictator_leader_gets_more(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        for c in colonists:
            e.get_account(c.id).food = 0.1
        collect_taxes(e, "dictator", colonists, rng)
        redistribute(e, "dictator", colonists, colonists[0].id)
        leader_total = e.get_account(colonists[0].id).total()
        other_total = e.get_account(colonists[1].id).total()
        assert leader_total > other_total


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

class TestTrade:
    def test_trades_occur(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        social = SocialGraph()
        social.initialize([c.id for c in colonists], rng)
        # Give some colonists surplus
        for c in colonists[:3]:
            e.get_account(c.id).food = 0.1
        # Run many years to get some trades (trust needs to build)
        all_trades = []
        for y in range(20):
            social.update_from_event([c.id for c in colonists], 0.3, rng)
            for c in colonists:
                partner = social.most_trusted_by(c.id, [x.id for x in colonists])
                if partner:
                    social.update_from_cooperation(c.id, partner, rng)
            trades = execute_trades(e, social, colonists, y, rng)
            all_trades.extend(trades)
        assert len(all_trades) > 0

    def test_trade_zero_sum(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        social = SocialGraph()
        social.initialize([c.id for c in colonists], rng)
        # Build trust
        for _ in range(20):
            social.update_from_event([c.id for c in colonists], 0.3, rng)
            for c in colonists:
                partner = social.most_trusted_by(c.id, [x.id for x in colonists])
                if partner:
                    social.update_from_cooperation(c.id, partner, rng)
        for c in colonists:
            e.get_account(c.id).food = 0.05
        before = sum(a.total() for a in e.accounts.values())
        execute_trades(e, social, colonists, 1, rng)
        after = sum(a.total() for a in e.accounts.values())
        assert abs(before - after) < 1e-9


# ---------------------------------------------------------------------------
# Gini
# ---------------------------------------------------------------------------

class TestGini:
    def test_perfect_equality(self):
        accounts = {f"c-{i}": PersonalAccount(food=0.05) for i in range(10)}
        assert abs(compute_gini(accounts)) < 1e-12

    def test_perfect_inequality(self):
        accounts = {f"c-{i}": PersonalAccount() for i in range(10)}
        accounts["c-0"].food = 0.1
        g = compute_gini(accounts)
        assert g > 0.8

    def test_empty(self):
        assert compute_gini({}) == 0.0

    def test_single(self):
        assert compute_gini({"a": PersonalAccount(food=0.1)}) == 0.0

    def test_range(self):
        """Gini is always between 0 and 1."""
        rng = random.Random(42)
        for _ in range(20):
            n = rng.randint(2, 20)
            accounts = {}
            for i in range(n):
                accounts[f"c-{i}"] = PersonalAccount(
                    food=rng.uniform(0, 0.05),
                    water=rng.uniform(0, 0.05))
            g = compute_gini(accounts)
            assert 0.0 <= g <= 1.0


# ---------------------------------------------------------------------------
# Black Market
# ---------------------------------------------------------------------------

class TestBlackMarket:
    def test_no_black_market_under_low_tax(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        assert not check_black_market(e, "anarchy", colonists)

    def test_black_market_with_high_paranoia(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        for c in colonists:
            c.stats.paranoia = 0.9
        result = check_black_market(e, "dictator", colonists)
        assert result is True
        assert e.black_market_active is True


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_handle_death_returns_to_pool(self):
        e = EconomyState()
        res = Resources()
        e.get_account("c-0").food = 0.1
        res.food = 0.5
        returned = handle_death(e, "c-0", res)
        assert returned > 0
        assert res.food > 0.5
        assert "c-0" not in e.accounts

    def test_handle_exile(self):
        e = EconomyState()
        res = Resources()
        e.get_account("c-0").water = 0.08
        handle_exile(e, "c-0", res)
        assert "c-0" not in e.accounts

    def test_handle_birth(self):
        e = EconomyState()
        handle_birth(e, "child-1")
        assert "child-1" in e.accounts
        assert e.get_account("child-1").total() == 0.0

    def test_handle_immigrant(self):
        e = EconomyState()
        rng = random.Random(42)
        handle_immigrant(e, "imm-1", rng)
        assert "imm-1" in e.accounts
        assert e.get_account("imm-1").total() > 0

    def test_spend_personal_reserves(self):
        e = EconomyState()
        e.get_account("c-0").food = 0.1
        spent = spend_personal_reserves(e, "c-0", "food")
        assert spent > 0
        assert e.get_account("c-0").food < 0.1


# ---------------------------------------------------------------------------
# Full tick
# ---------------------------------------------------------------------------

class TestTickEconomy:
    def test_tick_returns_summary(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        social = SocialGraph()
        ids = [c.id for c in colonists]
        social.initialize(ids, rng)
        for c in colonists:
            e.get_account(c.id)
        actions = {c.id: "farm" for c in colonists}
        res = Resources()
        summary = tick_economy(e, colonists, actions, "council", None,
                               social, res, 1, rng)
        assert "gini" in summary
        assert "total_wealth" in summary
        assert "tax_collected" in summary

    def test_gini_recorded(self):
        e = EconomyState()
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        social = SocialGraph()
        ids = [c.id for c in colonists]
        social.initialize(ids, rng)
        for c in colonists:
            e.get_account(c.id)
        res = Resources()
        for year in range(5):
            actions = {c.id: rng.choice(["farm", "code", "hoard", "rest"])
                       for c in colonists}
            tick_economy(e, colonists, actions, "council", None,
                         social, res, year, rng)
        assert len(e.gini_history) == 5


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_economy_in_year_result(self):
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(seed=42, total_years=3)
        r = e.run()
        for yr in r.years:
            assert "gini" in yr.economy
            assert "total_wealth" in yr.economy

    def test_economy_in_sim_result(self):
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(seed=42, total_years=3)
        r = e.run()
        d = r.to_dict()
        assert "final_economy" in d
        assert "gini" in d["final_economy"]
        assert "accounts" in d["final_economy"]

    def test_version_bumped(self):
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(seed=42, total_years=1)
        r = e.run()
        assert r.to_dict()["_meta"]["version"] == "7.0"

    def test_gini_evolves_over_time(self):
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(seed=42, total_years=20)
        r = e.run()
        ginis = [yr.economy.get("gini", 0) for yr in r.years]
        assert len(set(ginis)) > 1  # Gini changes over time

    def test_accounts_match_alive_colonists(self):
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(seed=42, total_years=50)
        e.run()
        active_ids = {c.id for c in e.colonists if c.is_active()}
        account_ids = set(e.economy.accounts.keys())
        # All active colonists should have accounts
        assert active_ids <= account_ids

    def test_10_year_smoke(self):
        """Run 10 years without crash, verify economy invariants."""
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(seed=99, total_years=10)
        r = e.run()
        assert len(r.years) == 10
        for yr in r.years:
            gini = yr.economy.get("gini", 0)
            assert 0.0 <= gini <= 1.0
            assert yr.economy.get("total_wealth", 0) >= 0

    def test_100_year_full(self):
        """Full 100-year run with economics organ."""
        from src.mars100.engine import Mars100Engine
        e = Mars100Engine(seed=42, total_years=100)
        r = e.run()
        assert len(r.years) > 0
        d = r.to_dict()
        assert d["_meta"]["version"] == "7.0"
        final_gini = d["final_economy"]["gini"]
        assert 0.0 <= final_gini <= 1.0
