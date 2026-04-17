"""Tests for Mars-100 economy organ."""
from __future__ import annotations

import pytest
from src.mars100.economy import (
    compute_gini, scarcity_multiplier, ColonistLedger,
    EconomyState, tick_economy, BASE_WAGE, TAX_RATE,
    RESOURCE_ACTIONS,
)


class TestGini:
    def test_empty(self):
        assert compute_gini([]) == 0.0

    def test_equal(self):
        assert compute_gini([100, 100, 100]) == 0.0

    def test_one_person(self):
        assert compute_gini([50]) == 0.0

    def test_extreme_inequality(self):
        g = compute_gini([0, 0, 0, 1000])
        assert 0.7 < g <= 1.0

    def test_moderate_inequality(self):
        g = compute_gini([10, 20, 30, 40])
        assert 0.0 < g < 0.5

    def test_all_zeros(self):
        assert compute_gini([0, 0, 0]) == 0.0

    def test_range(self):
        import random
        rng = random.Random(42)
        for _ in range(50):
            balances = [rng.uniform(0, 1000) for _ in range(10)]
            g = compute_gini(balances)
            assert 0.0 <= g <= 1.0, f"Gini {g} out of range"


class TestScarcity:
    def test_abundant(self):
        assert scarcity_multiplier(0.5) == 1.0

    def test_at_threshold(self):
        assert scarcity_multiplier(0.3) == 1.0

    def test_scarce(self):
        m = scarcity_multiplier(0.15)
        assert 1.0 < m < 3.0

    def test_empty(self):
        assert scarcity_multiplier(0.0) == 3.0


class TestLedger:
    def test_initial_balance(self):
        ledger = ColonistLedger()
        assert ledger.balance == 0.0

    def test_record_action(self):
        ledger = ColonistLedger()
        ledger.record_action("farm", 10.0)
        assert ledger.balance == 9.0  # 10 - 10% tax
        assert ledger.lifetime_earnings == 9.0
        assert ledger.action_counts["farm"] == 1

    def test_specialisation(self):
        ledger = ColonistLedger()
        for _ in range(8):
            ledger.record_action("farm", 10.0)
        for _ in range(2):
            ledger.record_action("code", 10.0)
        spec = ledger.specialisation()
        assert abs(spec["farm"] - 0.8) < 0.01
        assert abs(spec["code"] - 0.2) < 0.01

    def test_specialist_bonus_below_threshold(self):
        ledger = ColonistLedger()
        for a in ["farm", "code", "terraform"]:
            ledger.record_action(a, 10.0)
        assert ledger.specialist_bonus() == 0.0

    def test_specialist_bonus_above_threshold(self):
        ledger = ColonistLedger()
        for _ in range(5):
            ledger.record_action("farm", 10.0)
        assert ledger.specialist_bonus() > 0.0


class TestEconomyState:
    def test_get_ledger_creates(self):
        state = EconomyState()
        ledger = state.get_ledger(0)
        assert isinstance(ledger, ColonistLedger)
        assert 0 in state.ledgers

    def test_gini_empty(self):
        state = EconomyState()
        assert state.gini() == 0.0

    def test_to_dict(self):
        state = EconomyState()
        state.get_ledger(0).balance = 100.0
        d = state.to_dict()
        assert "treasury" in d
        assert "gini" in d
        assert "ledgers" in d


class SimpleResources:
    """Minimal resources stub for testing."""
    def __init__(self, **kwargs):
        defaults = {"air": 0.8, "food": 0.8, "water": 0.8,
                    "materials": 0.8, "energy": 0.8}
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestTickEconomy:
    def test_basic_tick(self):
        econ = EconomyState()
        resources = SimpleResources()
        actions = {0: "farm"}
        summary = tick_economy(econ, [], resources, actions, 1)
        assert summary["year"] == 1
        assert summary["treasury"] > 0
        assert summary["total_minted"] > 0

    def test_scarcity_wages(self):
        econ1 = EconomyState()
        econ2 = EconomyState()
        abundant = SimpleResources(food=0.8)
        scarce = SimpleResources(food=0.1)
        actions = {0: "farm"}
        tick_economy(econ1, [], abundant, actions, 1)
        tick_economy(econ2, [], scarce, actions, 1)
        assert econ2.get_ledger(0).balance > econ1.get_ledger(0).balance

    def test_tax_collected(self):
        econ = EconomyState()
        resources = SimpleResources()
        actions = {0: "rest"}
        tick_economy(econ, [], resources, actions, 1)
        assert econ.treasury > 0

    def test_no_actions_no_minting(self):
        econ = EconomyState()
        resources = SimpleResources()
        summary = tick_economy(econ, [], resources, {}, 1)
        assert summary["total_minted"] == 0

    def test_multiple_colonists(self):
        econ = EconomyState()
        resources = SimpleResources()
        actions = {0: "farm", 1: "terraform", 2: "code"}
        tick_economy(econ, [], resources, actions, 1)
        assert len(econ.ledgers) == 3

    def test_gini_evolves(self):
        econ = EconomyState()
        resources = SimpleResources(food=0.1)
        for year in range(10):
            actions = {0: "farm", 1: "code"}
            tick_economy(econ, [], resources, actions, year)
        assert econ.gini() > 0, "Unequal wages should produce nonzero Gini"


class TestEconomyInvariants:
    def test_balances_non_negative(self):
        import random
        rng = random.Random(42)
        econ = EconomyState()
        resources = SimpleResources()
        action_names = list(RESOURCE_ACTIONS.keys()) + ["code", "pray", "sabotage"]
        for year in range(100):
            acts = {i: rng.choice(action_names) for i in range(5)}
            tick_economy(econ, [], resources, acts, year)
        for lid, ledger in econ.ledgers.items():
            assert ledger.balance >= 0, f"Colonist {lid} has negative balance"

    def test_gini_in_range(self):
        import random
        rng = random.Random(99)
        econ = EconomyState()
        resources = SimpleResources()
        action_names = list(RESOURCE_ACTIONS.keys()) + ["code"]
        for year in range(50):
            acts = {i: rng.choice(action_names) for i in range(10)}
            tick_economy(econ, [], resources, acts, year)
        g = econ.gini()
        assert 0.0 <= g <= 1.0, f"Gini {g} out of range"

    def test_minted_equals_wages_plus_tax(self):
        econ = EconomyState()
        resources = SimpleResources()
        actions = {i: "farm" for i in range(5)}
        tick_economy(econ, [], resources, actions, 1)
        total_net = sum(l.balance for l in econ.ledgers.values())
        assert abs(econ.total_minted - (total_net + econ.treasury)) < 0.01

    def test_specialisation_sums_to_one(self):
        ledger = ColonistLedger()
        for _ in range(5):
            ledger.record_action("farm", 10)
        for _ in range(3):
            ledger.record_action("code", 10)
        spec = ledger.specialisation()
        assert abs(sum(spec.values()) - 1.0) < 0.001
