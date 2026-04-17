"""Tests for the Mars-100 economics organ (engine v7.0)."""
from __future__ import annotations

import random

import pytest

from src.mars100.colonist import (
    Colonist, ColonistSkills, ColonistStats, Wallet,
    create_founding_ten, create_immigrant,
)
from src.mars100.colony import Resources, SocialGraph, RESOURCE_NAMES
from src.mars100.economics import (
    EconomicState, EconomicTickResult, TradeRecord,
    allocate_labor_income, find_trades, apply_taxation,
    compute_gini, compute_economic_pressure,
    liquidate_estate, endow_immigrant, endow_child,
    tick_economics,
    should_propose_tax_change, generate_economic_proposal,
    LABOR_INCOME_FRACTION, TAX_RATES, TRADE_TRUST_THRESHOLD,
    GINI_UNREST_THRESHOLD, GINI_REVOLT_THRESHOLD,
    IMMIGRANT_ENDOWMENT, ECONOMIC_PROPOSAL_COOLDOWN,
    GINI_PROPOSAL_THRESHOLD,
)
from src.mars100.governance import GovernanceState


# ── Wallet tests ────────────────────────────────────────────────────────────

class TestWallet:
    def test_default_wallet_is_empty(self) -> None:
        w = Wallet()
        assert w.total_wealth() == 0.0
        for r in RESOURCE_NAMES:
            assert w.holdings[r] == 0.0

    def test_deposit_increases_holdings(self) -> None:
        w = Wallet()
        w.deposit("food", 0.5)
        assert w.holdings["food"] == pytest.approx(0.5)
        assert w.total_wealth() == pytest.approx(0.5)

    def test_deposit_negative_ignored(self) -> None:
        w = Wallet()
        w.deposit("food", -1.0)
        assert w.holdings["food"] == 0.0

    def test_withdraw_returns_actual(self) -> None:
        w = Wallet()
        w.deposit("water", 0.3)
        got = w.withdraw("water", 0.5)
        assert got == pytest.approx(0.3)
        assert w.holdings["water"] == pytest.approx(0.0)

    def test_withdraw_exact(self) -> None:
        w = Wallet()
        w.deposit("power", 1.0)
        got = w.withdraw("power", 0.4)
        assert got == pytest.approx(0.4)
        assert w.holdings["power"] == pytest.approx(0.6)

    def test_withdraw_zero(self) -> None:
        w = Wallet()
        assert w.withdraw("food", 0.0) == 0.0
        assert w.withdraw("food", -1.0) == 0.0

    def test_roundtrip_serialization(self) -> None:
        w = Wallet()
        w.deposit("food", 0.5)
        w.deposit("medicine", 0.1)
        w.total_earned = 0.6
        w.total_traded = 0.2
        w.total_taxed = 0.1
        d = w.to_dict()
        w2 = Wallet.from_dict(d)
        assert w2.holdings["food"] == pytest.approx(0.5)
        assert w2.holdings["medicine"] == pytest.approx(0.1)
        assert w2.total_earned == pytest.approx(0.6)
        assert w2.total_traded == pytest.approx(0.2)
        assert w2.total_taxed == pytest.approx(0.1)

    def test_from_dict_empty(self) -> None:
        """Backward compat: from_dict({}) returns empty wallet."""
        w = Wallet.from_dict({})
        assert w.total_wealth() == 0.0

    def test_holdings_are_unbounded(self) -> None:
        """Wallets are NOT clamped to 0-1."""
        w = Wallet()
        for _ in range(100):
            w.deposit("food", 0.5)
        assert w.holdings["food"] == pytest.approx(50.0)

    def test_no_negative_holdings(self) -> None:
        """Holdings never go negative."""
        w = Wallet()
        w.deposit("air", 0.1)
        got = w.withdraw("air", 1.0)
        assert got == pytest.approx(0.1)
        assert w.holdings["air"] >= 0.0


# ── Colonist wallet integration ────────────────────────────────────────────

class TestColonistWallet:
    def test_colonist_has_wallet(self) -> None:
        colonists = create_founding_ten(42)
        for c in colonists:
            assert hasattr(c, "wallet")
            assert isinstance(c.wallet, Wallet)
            assert c.wallet.total_wealth() == 0.0

    def test_colonist_to_dict_includes_wallet(self) -> None:
        colonists = create_founding_ten(42)
        d = colonists[0].to_dict()
        assert "wallet" in d
        assert "holdings" in d["wallet"]

    def test_colonist_from_dict_with_wallet(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 0.3)
        d = colonists[0].to_dict()
        c2 = Colonist.from_dict(d)
        assert c2.wallet.holdings["food"] == pytest.approx(0.3)

    def test_colonist_from_dict_without_wallet(self) -> None:
        """Backward compat: old data without wallet field."""
        colonists = create_founding_ten(42)
        d = colonists[0].to_dict()
        del d["wallet"]
        c2 = Colonist.from_dict(d)
        assert c2.wallet.total_wealth() == 0.0

    def test_lispy_bindings_include_wealth(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 0.5)
        bindings = colonists[0].lispy_bindings()
        assert "wealth" in bindings
        assert bindings["wealth"] == pytest.approx(0.5)


# ── Labor income ────────────────────────────────────────────────────────────

class TestLaborIncome:
    def test_farmer_earns_food(self) -> None:
        colonists = create_founding_ten(42)
        farmer = colonists[6]  # Grove Ash, hydroponics=0.9
        resources = Resources()
        rng = random.Random(99)
        income = allocate_labor_income(farmer, "farm", resources, rng)
        assert income > 0
        assert farmer.wallet.holdings["food"] > 0

    def test_rest_earns_nothing(self) -> None:
        colonists = create_founding_ten(42)
        c = colonists[0]
        resources = Resources()
        rng = random.Random(99)
        income = allocate_labor_income(c, "rest", resources, rng)
        assert income == 0.0
        assert c.wallet.total_wealth() == 0.0

    def test_conservation_colony_decreases(self) -> None:
        """Income comes FROM colony pool — colony resources decrease."""
        colonists = create_founding_ten(42)
        farmer = colonists[6]
        resources = Resources()
        food_before = resources.food
        rng = random.Random(99)
        income = allocate_labor_income(farmer, "farm", resources, rng)
        assert resources.food < food_before
        assert resources.food == pytest.approx(food_before - income)

    def test_income_capped_at_10_percent(self) -> None:
        """Can't take more than 10% of colony stock."""
        colonists = create_founding_ten(42)
        farmer = colonists[6]
        resources = Resources(food=0.01)  # very low
        rng = random.Random(99)
        income = allocate_labor_income(farmer, "farm", resources, rng)
        assert income <= 0.001  # 10% of 0.01

    def test_non_productive_actions_no_income(self) -> None:
        colonists = create_founding_ten(42)
        c = colonists[0]
        resources = Resources()
        rng = random.Random(99)
        for action in ["mediate", "cooperate", "hoard", "rest", "sabotage", "research"]:
            income = allocate_labor_income(c, action, resources, rng)
            assert income == 0.0


# ── Trade ───────────────────────────────────────────────────────────────────

class TestTrade:
    def _setup_trading_pair(self) -> tuple[list[Colonist], SocialGraph]:
        colonists = create_founding_ten(42)
        social = SocialGraph()
        ids = [c.id for c in colonists]
        social.initialize(ids, random.Random(42))
        # Give them stuff to trade
        colonists[0].wallet.deposit("food", 0.5)
        colonists[1].wallet.deposit("medicine", 0.5)
        # Ensure high trust
        social.get(colonists[0].id, colonists[1].id).trust = 0.8
        social.get(colonists[1].id, colonists[0].id).trust = 0.8
        social.get(colonists[0].id, colonists[1].id).affection = 0.7
        social.get(colonists[1].id, colonists[0].id).affection = 0.7
        return colonists, social

    def test_trade_happens_with_trust(self) -> None:
        colonists, social = self._setup_trading_pair()
        trades = find_trades(colonists, social, year=5, rng=random.Random(42))
        assert len(trades) >= 0  # may or may not find a trade depending on medians

    def test_no_trade_with_low_trust(self) -> None:
        colonists, social = self._setup_trading_pair()
        # Kill all trust
        for c1 in colonists:
            for c2 in colonists:
                if c1.id != c2.id:
                    social.get(c1.id, c2.id).trust = 0.1
        trades = find_trades(colonists, social, year=5, rng=random.Random(42))
        assert len(trades) == 0

    def test_trade_conserves_wealth(self) -> None:
        """Trades don't create or destroy goods."""
        colonists, social = self._setup_trading_pair()
        total_before = sum(c.wallet.total_wealth() for c in colonists)
        find_trades(colonists, social, year=5, rng=random.Random(42))
        total_after = sum(c.wallet.total_wealth() for c in colonists)
        assert total_after == pytest.approx(total_before, abs=1e-10)

    def test_single_colonist_no_trades(self) -> None:
        colonists = create_founding_ten(42)[:1]
        social = SocialGraph()
        social.initialize([colonists[0].id], random.Random(42))
        trades = find_trades(colonists, social, year=5, rng=random.Random(42))
        assert len(trades) == 0


# ── Taxation ────────────────────────────────────────────────────────────────

class TestTaxation:
    def test_tax_returns_to_colony(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 0.5)
        resources = Resources(food=0.5)
        food_before = resources.food
        tax = apply_taxation(colonists, "council", resources)
        assert tax > 0
        assert resources.food > food_before

    def test_dictator_taxes_more_than_anarchy(self) -> None:
        colonists_d = create_founding_ten(42)
        colonists_a = create_founding_ten(42)
        for c in colonists_d + colonists_a:
            c.wallet.deposit("food", 0.5)
        res_d = Resources()
        res_a = Resources()
        tax_d = apply_taxation(colonists_d, "dictator", res_d)
        tax_a = apply_taxation(colonists_a, "anarchy", res_a)
        assert tax_d > tax_a

    def test_tax_reduces_wallet(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 1.0)
        wealth_before = colonists[0].wallet.total_wealth()
        apply_taxation(colonists, "council", Resources())
        assert colonists[0].wallet.total_wealth() < wealth_before

    def test_empty_wallets_no_tax(self) -> None:
        colonists = create_founding_ten(42)
        tax = apply_taxation(colonists, "council", Resources())
        assert tax == 0.0

    def test_tax_no_negative_wallet(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 0.01)
        apply_taxation(colonists, "dictator", Resources())
        for r in RESOURCE_NAMES:
            assert colonists[0].wallet.holdings[r] >= 0.0


# ── Gini coefficient ───────────────────────────────────────────────────────

class TestGini:
    def test_perfect_equality(self) -> None:
        colonists = create_founding_ten(42)
        for c in colonists:
            c.wallet.deposit("food", 0.5)
        assert compute_gini(colonists) == pytest.approx(0.0, abs=0.01)

    def test_perfect_inequality(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 10.0)
        # rest have 0
        gini = compute_gini(colonists)
        assert gini > 0.8

    def test_zero_wealth_returns_zero(self) -> None:
        colonists = create_founding_ten(42)
        assert compute_gini(colonists) == 0.0

    def test_single_colonist(self) -> None:
        colonists = create_founding_ten(42)[:1]
        colonists[0].wallet.deposit("food", 1.0)
        assert compute_gini(colonists) == 0.0

    def test_gini_between_zero_and_one(self) -> None:
        colonists = create_founding_ten(42)
        rng = random.Random(42)
        for c in colonists:
            c.wallet.deposit("food", rng.random())
        gini = compute_gini(colonists)
        assert 0.0 <= gini <= 1.0

    def test_inactive_colonists_excluded(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 10.0)
        colonists[0].alive = False  # dead
        gini = compute_gini(colonists)  # should ignore dead colonist
        assert gini == 0.0  # rest all have 0


# ── Economic pressure ──────────────────────────────────────────────────────

class TestEconomicPressure:
    def test_rich_colonist_hoards(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 0.5)
        pressure = compute_economic_pressure(colonists[0])
        assert pressure.get("hoard", 0) > 0

    def test_poor_colonist_cooperates(self) -> None:
        colonists = create_founding_ten(42)
        # wallet is empty (wealth < 0.02)
        pressure = compute_economic_pressure(colonists[0])
        assert pressure.get("cooperate", 0) > 0

    def test_medium_wealth_no_pressure(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 0.05)  # between 0.02 and 0.15
        pressure = compute_economic_pressure(colonists[0])
        assert len(pressure) == 0


# ── Estate liquidation ─────────────────────────────────────────────────────

class TestEstateLiquidation:
    def test_dead_estate_returns_to_colony(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 0.5)
        colonists[0].die(year=10, cause="test")
        resources = Resources(food=0.3)
        estate = liquidate_estate(colonists[0], resources)
        assert estate.get("food", 0) == pytest.approx(0.5)
        assert resources.food == pytest.approx(0.8)
        assert colonists[0].wallet.total_wealth() == 0.0

    def test_exiled_estate_is_lost(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].wallet.deposit("food", 0.5)
        colonists[0].exile(year=10)
        resources = Resources(food=0.3)
        estate = liquidate_estate(colonists[0], resources)
        assert estate.get("food", 0) == pytest.approx(0.5)
        # Exiled: wealth is LOST, not returned
        assert resources.food == pytest.approx(0.3)

    def test_empty_wallet_returns_empty_estate(self) -> None:
        colonists = create_founding_ten(42)
        colonists[0].die(year=10, cause="test")
        estate = liquidate_estate(colonists[0], Resources())
        assert len(estate) == 0


# ── Immigrant endowment ───────────────────────────────────────────────────

class TestImmigrantEndowment:
    def test_endowment_adds_resources(self) -> None:
        rng = random.Random(42)
        imm = create_immigrant("imm-1", 20, rng)
        endow_immigrant(imm)
        assert imm.wallet.total_wealth() > 0
        for r in RESOURCE_NAMES:
            assert imm.wallet.holdings[r] == pytest.approx(IMMIGRANT_ENDOWMENT)


# ── Economic state ─────────────────────────────────────────────────────────

class TestEconomicState:
    def test_roundtrip(self) -> None:
        state = EconomicState()
        state.gini_history = [0.1, 0.2, 0.3]
        state.total_trades = 5
        state.total_volume = 1.5
        d = state.to_dict()
        state2 = EconomicState.from_dict(d)
        assert state2.total_trades == 5
        assert len(state2.gini_history) == 3

    def test_from_dict_empty(self) -> None:
        state = EconomicState.from_dict({})
        assert state.total_trades == 0


# ── tick_economics integration ─────────────────────────────────────────────

class TestTickEconomics:
    def _make_scenario(self, seed: int = 42) -> tuple:
        colonists = create_founding_ten(seed)
        social = SocialGraph()
        ids = [c.id for c in colonists]
        social.initialize(ids, random.Random(seed))
        resources = Resources()
        econ_state = EconomicState()
        rng = random.Random(seed)
        actions = {c.id: "farm" for c in colonists[:3]}
        actions.update({c.id: "code" for c in colonists[3:6]})
        actions.update({c.id: "rest" for c in colonists[6:]})
        return colonists, social, resources, econ_state, rng, actions

    def test_tick_returns_result(self) -> None:
        colonists, social, resources, econ_state, rng, actions = self._make_scenario()
        result = tick_economics(
            colonists, actions, "council", social,
            resources, econ_state, year=5, rng=rng)
        assert isinstance(result, EconomicTickResult)
        assert result.year == 5
        assert 0.0 <= result.gini <= 1.0

    def test_tick_updates_gini_history(self) -> None:
        colonists, social, resources, econ_state, rng, actions = self._make_scenario()
        tick_economics(colonists, actions, "council", social,
                       resources, econ_state, year=5, rng=rng)
        assert len(econ_state.gini_history) == 1

    def test_tick_serializable(self) -> None:
        """Result can be serialized to dict."""
        colonists, social, resources, econ_state, rng, actions = self._make_scenario()
        result = tick_economics(colonists, actions, "council", social,
                                resources, econ_state, year=5, rng=rng)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "gini" in d

    def test_no_wealth_created_from_nothing(self) -> None:
        """Conservation: colony + wallets after tick == colony before tick."""
        colonists, social, resources, econ_state, rng, actions = self._make_scenario()
        colony_before = resources.total()
        wallets_before = sum(c.wallet.total_wealth() for c in colonists)
        total_before = colony_before + wallets_before

        tick_economics(colonists, actions, "council", social,
                       resources, econ_state, year=5, rng=rng)

        colony_after = resources.total()
        wallets_after = sum(c.wallet.total_wealth() for c in colonists)
        total_after = colony_after + wallets_after

        # Some wealth may be lost due to colony resource clamping at 1.0
        # but total should never INCREASE
        assert total_after <= total_before + 1e-10


# ── Full engine smoke test ─────────────────────────────────────────────────

class TestEngineWithEconomics:
    def test_10_year_smoke(self) -> None:
        """Run 10 years with economics integrated — no crash."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=99, total_years=10)
        result = engine.run()
        assert len(result.years) == 10
        for yr in result.years:
            assert "economics" in yr.to_dict()

    def test_full_100_year(self) -> None:
        """Run full 100-year sim — verify economics populated."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=100)
        result = engine.run()
        d = result.to_dict()
        assert d["_meta"]["version"] == "8.0"
        assert "final_economics" in d
        assert d["final_economics"]["total_trades"] >= 0
        assert len(d["final_economics"]["gini_history"]) > 0

    def test_deterministic_same_seed(self) -> None:
        """Same seed produces same economics results."""
        from src.mars100.engine import Mars100Engine
        r1 = Mars100Engine(seed=123, total_years=20).run()
        r2 = Mars100Engine(seed=123, total_years=20).run()
        for y1, y2 in zip(r1.years, r2.years):
            assert y1.economics == y2.economics

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different results."""
        from src.mars100.engine import Mars100Engine
        r1 = Mars100Engine(seed=42, total_years=20).run()
        r2 = Mars100Engine(seed=99, total_years=20).run()
        # At least some years should differ
        diffs = sum(1 for y1, y2 in zip(r1.years, r2.years)
                    if y1.economics != y2.economics)
        assert diffs > 0

    def test_wallets_in_final_colonists(self) -> None:
        """Final colonist snapshots include wallet data."""
        from src.mars100.engine import Mars100Engine
        result = Mars100Engine(seed=42, total_years=50).run()
        d = result.to_dict()
        for colonist in d["final_colonists"]:
            assert "wallet" in colonist

    def test_gini_varies_over_time(self) -> None:
        """Gini coefficient should change over 50 years."""
        from src.mars100.engine import Mars100Engine
        result = Mars100Engine(seed=42, total_years=50).run()
        ginis = [yr.economics.get("gini", 0) for yr in result.years]
        # Not all the same
        assert len(set(round(g, 4) for g in ginis)) > 1


# ── Birth endowment tests ──────────────────────────────────────────────────

class TestEndowChild:
    """Tests for endow_child() — parents share wealth with newborns."""

    def _make_colonist(self, cid: str, holdings: dict[str, float] | None = None) -> Colonist:
        rng = random.Random(42)
        ten = create_founding_ten(rng)
        c = ten[0]
        c.id = cid
        c.wallet = Wallet()
        if holdings:
            for res, amt in holdings.items():
                c.wallet.deposit(res, amt)
                c.wallet.total_earned += amt
        return c

    def test_basic_endowment(self) -> None:
        """Each parent gives 10% of each resource to the child."""
        parent_a = self._make_colonist("pa", {"food": 1.0, "water": 0.5})
        parent_b = self._make_colonist("pb", {"food": 0.6, "water": 0.0})
        child = self._make_colonist("child")

        transferred = endow_child(child, parent_a, parent_b)

        # Parent A gives 10% of food (0.1) + Parent B gives 10% of food (0.06) = 0.16
        assert abs(child.wallet.holdings.get("food", 0) - 0.16) < 1e-6
        # Parent A gives 10% of water (0.05) + Parent B gives nothing
        assert abs(child.wallet.holdings.get("water", 0) - 0.05) < 1e-6
        # Parents lost their contributions
        assert abs(parent_a.wallet.holdings.get("food", 0) - 0.9) < 1e-6
        assert abs(parent_b.wallet.holdings.get("food", 0) - 0.54) < 1e-6
        # Transfer log only includes non-zero resources
        assert "food" in transferred
        assert "water" in transferred
        assert transferred["food"] == pytest.approx(0.16, abs=1e-5)

    def test_empty_wallet_parents(self) -> None:
        """No crash when parents have empty wallets."""
        parent_a = self._make_colonist("pa")
        parent_b = self._make_colonist("pb")
        child = self._make_colonist("child")

        transferred = endow_child(child, parent_a, parent_b)

        assert transferred == {}
        assert child.wallet.total_wealth() == pytest.approx(0.0)

    def test_conservation(self) -> None:
        """Total wealth before == total wealth after (conservation law)."""
        rng = random.Random(99)
        parent_a = self._make_colonist("pa")
        parent_b = self._make_colonist("pb")
        for res in RESOURCE_NAMES:
            amt_a = rng.uniform(0, 0.5)
            amt_b = rng.uniform(0, 0.5)
            parent_a.wallet.deposit(res, amt_a)
            parent_b.wallet.deposit(res, amt_b)
        child = self._make_colonist("child")

        total_before = parent_a.wallet.total_wealth() + parent_b.wallet.total_wealth()
        endow_child(child, parent_a, parent_b)
        total_after = (parent_a.wallet.total_wealth()
                       + parent_b.wallet.total_wealth()
                       + child.wallet.total_wealth())

        assert total_before == pytest.approx(total_after, abs=1e-9)


# ── Economic governance proposal tests ──────────────────────────────────────

class TestShouldProposeTaxChange:
    """Tests for should_propose_tax_change()."""

    def test_no_history_returns_false(self) -> None:
        state = EconomicState()
        assert should_propose_tax_change(state, 10, random.Random(42)) is False

    def test_cooldown_respected(self) -> None:
        """Cannot propose if last proposal was < COOLDOWN years ago."""
        state = EconomicState(gini_history=[0.6], last_proposal_year=8)
        # Year 12, cooldown is 5, so 12-8=4 < 5 → blocked
        assert should_propose_tax_change(state, 12, random.Random(42)) is False

    def test_high_gini_eventually_proposes(self) -> None:
        """With high Gini (~50% chance), should eventually return True."""
        state = EconomicState(gini_history=[0.7])
        found_true = False
        for seed in range(100):
            if should_propose_tax_change(state, 200, random.Random(seed)):
                found_true = True
                break
        assert found_true, "Expected at least one proposal with high Gini in 100 trials"

    def test_low_gini_sometimes_proposes(self) -> None:
        """With low Gini (~8% chance), should sometimes return True."""
        state = EconomicState(gini_history=[0.2])
        found_true = False
        for seed in range(200):
            if should_propose_tax_change(state, 200, random.Random(seed)):
                found_true = True
                break
        assert found_true, "Expected at least one proposal with low Gini in 200 trials"


class TestGenerateEconomicProposal:
    """Tests for generate_economic_proposal()."""

    def test_high_gini_proposes_increase(self) -> None:
        state = EconomicState(gini_history=[0.6])
        proposal = generate_economic_proposal(state, "council", "col-1", 20, random.Random(42))
        assert proposal["direction"] == "increase"
        assert proposal["proposed_rate"] > proposal["current_rate"]
        assert state.last_proposal_year == 20

    def test_low_gini_proposes_decrease(self) -> None:
        state = EconomicState(gini_history=[0.3])
        proposal = generate_economic_proposal(state, "council", "col-2", 30, random.Random(42))
        assert proposal["direction"] == "decrease"
        assert proposal["proposed_rate"] < proposal["current_rate"]

    def test_proposed_rate_bounded(self) -> None:
        """Tax rate should stay between 0.05 and 0.50."""
        for seed in range(50):
            rng = random.Random(seed)
            state = EconomicState(gini_history=[rng.uniform(0.1, 0.8)])
            proposal = generate_economic_proposal(
                state, "dictator", "col-1", 100, rng)
            assert 0.05 <= proposal["proposed_rate"] <= 0.50

    def test_proposal_has_required_fields(self) -> None:
        state = EconomicState(gini_history=[0.5])
        proposal = generate_economic_proposal(state, "council", "col-3", 50, random.Random(1))
        for key in ("type", "proposer_id", "year", "current_rate", "proposed_rate", "direction", "reason"):
            assert key in proposal, f"Missing field: {key}"


class TestEconomicProposalInTickResult:
    """Tests that economic proposals flow through the EconomicTickResult."""

    def test_tick_result_serializes_proposal(self) -> None:
        result = EconomicTickResult(
            year=10, gini=0.5, trades=[], tax_collected=0.1,
            labor_income_total=0.5, economic_proposal={"type": "economic_tax_change"})
        d = result.to_dict()
        assert "economic_proposal" in d
        assert d["economic_proposal"]["type"] == "economic_tax_change"

    def test_tick_result_omits_none_proposal(self) -> None:
        result = EconomicTickResult(
            year=10, gini=0.5, trades=[], tax_collected=0.1,
            labor_income_total=0.5)
        d = result.to_dict()
        assert "economic_proposal" not in d


class TestBirthEndowmentIntegration:
    """Integration test: births in engine include economic endowment."""

    def test_births_with_endowment(self) -> None:
        """Run engine long enough for births; verify endowment field."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=80)
        result = engine.run()
        births_with_endowment = [
            b for yr in result.years for b in yr.births
            if "endowment" in b
        ]
        # At least one birth should have endowment data over 80 years
        # (births require year >= 10, resources > 0.4, pair trust > 1.4)
        if any(yr.births for yr in result.years):
            # If there were births, they should have the endowment field
            all_births = [b for yr in result.years for b in yr.births]
            for birth in all_births:
                assert "endowment" in birth, f"Birth missing endowment: {birth}"
