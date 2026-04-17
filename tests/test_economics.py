"""Tests for the Mars-100 economics organ."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.economics import (
    EconomicState,
    EconomicTickResult,
    compute_labor,
    compute_gini,
    default_model_for_governance,
    distribute_surplus,
    compute_economic_pressure,
    initialize_wealth,
    handle_birth,
    handle_death,
    tick_economics,
    compute_wealth_effects,
    inequality_trust_erosion,
    BASE_LABOR,
    FOUNDER_WEALTH,
    CHILD_WEALTH,
    IMMIGRANT_WEALTH,
    HOARDING_BONUS,
    ACTION_SKILL_MAP,
)


# ── compute_labor ──────────────────────────────────────────────────────

class TestComputeLabor:
    def test_base_labor_with_no_skills(self) -> None:
        result = compute_labor("rest", {}, 0.0)
        assert abs(result - BASE_LABOR) < 1e-9

    def test_labor_increases_with_relevant_skill(self) -> None:
        low = compute_labor("farm", {"hydroponics": 0.1}, 0.5)
        high = compute_labor("farm", {"hydroponics": 0.9}, 0.5)
        assert high > low

    def test_labor_increases_with_resolve(self) -> None:
        low = compute_labor("farm", {"hydroponics": 0.5}, 0.0)
        high = compute_labor("farm", {"hydroponics": 0.5}, 1.0)
        assert high > low

    def test_all_actions_have_skill_mapping(self) -> None:
        actions = ["terraform", "farm", "mediate", "code", "pray",
                   "sabotage", "cooperate", "hoard", "explore", "rest", "research"]
        for action in actions:
            result = compute_labor(action, {"terraforming": 0.5, "hydroponics": 0.5,
                                            "mediation": 0.5, "coding": 0.5,
                                            "prayer": 0.5, "sabotage": 0.5}, 0.5)
            assert result >= BASE_LABOR, f"action {action} should produce >= base labor"

    def test_labor_is_positive(self) -> None:
        for _ in range(100):
            result = compute_labor("farm", {"hydroponics": 0.0}, 0.0)
            assert result >= 0.0

    def test_unknown_action_gets_base_labor(self) -> None:
        result = compute_labor("unknown_action", {}, 0.5)
        assert result >= BASE_LABOR


# ── compute_gini ───────────────────────────────────────────────────────

class TestComputeGini:
    def test_perfect_equality(self) -> None:
        gini = compute_gini([0.5, 0.5, 0.5, 0.5])
        assert gini == 0.0

    def test_maximal_inequality(self) -> None:
        gini = compute_gini([0.0, 0.0, 0.0, 1.0])
        assert gini > 0.9, f"expected near 1.0, got {gini}"

    def test_single_colonist_is_zero(self) -> None:
        assert compute_gini([0.5]) == 0.0

    def test_empty_is_zero(self) -> None:
        assert compute_gini([]) == 0.0

    def test_two_colonists_one_has_all(self) -> None:
        gini = compute_gini([0.0, 1.0])
        assert gini > 0.9

    def test_all_zero_wealth(self) -> None:
        assert compute_gini([0.0, 0.0, 0.0]) == 0.0

    def test_gini_bounded_0_1(self) -> None:
        rng = random.Random(42)
        for _ in range(200):
            n = rng.randint(2, 20)
            values = [rng.random() for _ in range(n)]
            gini = compute_gini(values)
            assert 0.0 <= gini <= 1.0, f"gini {gini} out of bounds for {values}"

    def test_moderate_inequality(self) -> None:
        gini = compute_gini([0.1, 0.2, 0.3, 0.4, 0.5])
        assert 0.0 < gini < 0.5

    def test_gini_increases_with_inequality(self) -> None:
        equal = compute_gini([0.5, 0.5, 0.5, 0.5])
        mild = compute_gini([0.3, 0.4, 0.5, 0.8])
        extreme = compute_gini([0.0, 0.0, 0.0, 1.0])
        assert equal <= mild <= extreme


# ── distribute_surplus ─────────────────────────────────────────────────

class TestDistributeSurplus:
    def test_communal_equal_shares(self) -> None:
        labor = {"a": 1.0, "b": 0.5}
        dist = distribute_surplus("communal", 1.0, labor, ["a", "b"])
        assert abs(dist["a"] - 0.5) < 1e-9
        assert abs(dist["b"] - 0.5) < 1e-9

    def test_market_proportional(self) -> None:
        labor = {"a": 1.0, "b": 0.5}
        dist = distribute_surplus("market", 1.5, labor, ["a", "b"])
        assert dist["a"] > dist["b"]
        assert abs(dist["a"] + dist["b"] - 1.5) < 1e-9

    def test_zero_surplus(self) -> None:
        dist = distribute_surplus("communal", 0.0, {"a": 1.0}, ["a"])
        assert dist["a"] == 0.0

    def test_negative_surplus(self) -> None:
        dist = distribute_surplus("communal", -0.5, {"a": 1.0}, ["a"])
        assert dist["a"] == 0.0

    def test_empty_colonists(self) -> None:
        dist = distribute_surplus("communal", 1.0, {}, [])
        assert dist == {}

    def test_market_conserves_surplus(self) -> None:
        labor = {"a": 0.8, "b": 0.6, "c": 1.0}
        surplus = 0.9
        dist = distribute_surplus("market", surplus, labor, ["a", "b", "c"])
        assert abs(sum(dist.values()) - surplus) < 1e-9

    def test_communal_conserves_surplus(self) -> None:
        surplus = 1.2
        dist = distribute_surplus("communal", surplus, {"a": 0.5, "b": 0.5, "c": 0.5},
                                  ["a", "b", "c"])
        assert abs(sum(dist.values()) - surplus) < 1e-9


# ── default_model_for_governance ───────────────────────────────────────

class TestGovernanceMapping:
    def test_anarchy_is_communal(self) -> None:
        assert default_model_for_governance("anarchy") == "communal"

    def test_dictator_is_market(self) -> None:
        assert default_model_for_governance("dictator") == "market"

    def test_council_is_communal(self) -> None:
        assert default_model_for_governance("council") == "communal"

    def test_unknown_defaults_communal(self) -> None:
        assert default_model_for_governance("unknown_gov") == "communal"

    def test_ai_governor_is_market(self) -> None:
        assert default_model_for_governance("ai_governor") == "market"


# ── compute_economic_pressure ──────────────────────────────────────────

class TestEconomicPressure:
    def test_no_inequality_no_pressure(self) -> None:
        assert compute_economic_pressure(0.0, 0.8) == 0.0

    def test_high_inequality_low_resources(self) -> None:
        p = compute_economic_pressure(0.8, 0.2)
        assert p > 0.5

    def test_high_inequality_high_resources(self) -> None:
        p = compute_economic_pressure(0.8, 0.9)
        assert p < compute_economic_pressure(0.8, 0.2)

    def test_pressure_bounded_0_1(self) -> None:
        rng = random.Random(42)
        for _ in range(100):
            gini = rng.random()
            resource_avg = rng.random()
            p = compute_economic_pressure(gini, resource_avg)
            assert 0.0 <= p <= 1.0


# ── handle_birth / handle_death ────────────────────────────────────────

class TestWealthLifecycle:
    def test_birth_child_wealth(self) -> None:
        econ = EconomicState()
        handle_birth(econ, "child-1", is_immigrant=False)
        assert econ.wealth["child-1"] == CHILD_WEALTH

    def test_birth_immigrant_wealth(self) -> None:
        econ = EconomicState()
        handle_birth(econ, "imm-1", is_immigrant=True)
        assert econ.wealth["imm-1"] == IMMIGRANT_WEALTH

    def test_death_redistributes_estate(self) -> None:
        econ = EconomicState(wealth={"a": 0.6, "b": 0.2, "c": 0.3})
        handle_death(econ, "a", active_ids=["a", "b", "c"])
        assert "a" not in econ.wealth
        assert econ.wealth["b"] > 0.2
        assert econ.wealth["c"] > 0.3

    def test_death_conserves_wealth(self) -> None:
        econ = EconomicState(wealth={"a": 0.6, "b": 0.2, "c": 0.3})
        total_before = sum(econ.wealth.values())
        handle_death(econ, "a", active_ids=["a", "b", "c"])
        total_after = sum(econ.wealth.values())
        assert abs(total_before - total_after) < 1e-9

    def test_death_no_survivors(self) -> None:
        econ = EconomicState(wealth={"a": 0.5})
        handle_death(econ, "a", active_ids=["a"])
        assert "a" not in econ.wealth

    def test_death_missing_colonist(self) -> None:
        econ = EconomicState(wealth={"b": 0.3})
        handle_death(econ, "a", active_ids=["a", "b"])
        assert econ.wealth["b"] == 0.3


# ── initialize_wealth ─────────────────────────────────────────────────

class TestInitializeWealth:
    def test_founder_default(self) -> None:
        w = initialize_wealth(["a", "b", "c"])
        assert all(v == FOUNDER_WEALTH for v in w.values())
        assert len(w) == 3

    def test_custom_initial(self) -> None:
        w = initialize_wealth(["x"], initial_value=0.7)
        assert w["x"] == 0.7


# ── EconomicState serialization ────────────────────────────────────────

class TestEconomicStateSerialization:
    def test_round_trip(self) -> None:
        econ = EconomicState(
            wealth={"a": 0.5, "b": 0.3},
            model="market", gini=0.25, pressure=0.1,
            total_surplus_distributed=2.5)
        d = econ.to_dict()
        restored = EconomicState.from_dict(d)
        assert restored.wealth == econ.wealth
        assert restored.model == econ.model
        assert abs(restored.gini - econ.gini) < 1e-3
        assert abs(restored.pressure - econ.pressure) < 1e-3

    def test_empty_dict(self) -> None:
        econ = EconomicState.from_dict({})
        assert econ.wealth == {}
        assert econ.model == "communal"


# ── tick_economics integration ─────────────────────────────────────────

class TestTickEconomics:
    def _make_colonist_data(self, n: int = 4) -> list[dict]:
        return [
            {"id": f"c-{i}", "stats": {"resolve": 0.5, "hoarding": 0.3},
             "skills": {"terraforming": 0.3, "hydroponics": 0.5,
                        "mediation": 0.2, "coding": 0.4,
                        "prayer": 0.1, "sabotage": 0.0}}
            for i in range(n)
        ]

    def test_tick_returns_valid_result(self) -> None:
        econ = EconomicState()
        colonists = self._make_colonist_data()
        actions = {c["id"]: "farm" for c in colonists}
        rng = random.Random(42)
        result = tick_economics(econ, actions, colonists, 0.6, "council", rng)
        assert isinstance(result, EconomicTickResult)
        assert result.model_used == "communal"
        assert 0.0 <= result.gini_after <= 1.0
        assert 0.0 <= result.pressure <= 1.0

    def test_wealth_stays_bounded(self) -> None:
        econ = EconomicState()
        colonists = self._make_colonist_data()
        rng = random.Random(42)
        for year in range(50):
            actions = {c["id"]: "farm" for c in colonists}
            tick_economics(econ, actions, colonists, 0.7, "council", rng)
        for cid in econ.wealth:
            assert 0.0 <= econ.wealth[cid] <= 1.0, f"wealth {econ.wealth[cid]} out of bounds"

    def test_market_creates_more_inequality(self) -> None:
        rng_c = random.Random(42)
        rng_m = random.Random(42)
        colonists = self._make_colonist_data()
        colonists[0]["skills"]["hydroponics"] = 0.9
        colonists[1]["skills"]["hydroponics"] = 0.1

        econ_c = EconomicState()
        econ_m = EconomicState()
        for _ in range(20):
            actions = {c["id"]: "farm" for c in colonists}
            tick_economics(econ_c, actions, colonists, 0.6, "council", rng_c)
            tick_economics(econ_m, actions, colonists, 0.6, "dictator", rng_m)
        assert econ_m.gini >= econ_c.gini, (
            f"market gini {econ_m.gini} should be >= communal gini {econ_c.gini}")

    def test_initializes_missing_colonists(self) -> None:
        econ = EconomicState()
        colonists = self._make_colonist_data(2)
        actions = {c["id"]: "farm" for c in colonists}
        rng = random.Random(42)
        tick_economics(econ, actions, colonists, 0.6, "council", rng)
        for c in colonists:
            assert c["id"] in econ.wealth

    def test_serialization_round_trip(self) -> None:
        econ = EconomicState()
        colonists = self._make_colonist_data()
        actions = {c["id"]: "farm" for c in colonists}
        rng = random.Random(42)
        result = tick_economics(econ, actions, colonists, 0.6, "council", rng)
        d = result.to_dict()
        assert isinstance(d["labor"], dict)
        assert isinstance(d["gini_after"], float)
        assert d["model_used"] == "communal"

    def test_hoarding_increases_wealth(self) -> None:
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        low_hoard = [{"id": "a", "stats": {"resolve": 0.5, "hoarding": 0.0},
                      "skills": {"hydroponics": 0.5}}]
        high_hoard = [{"id": "a", "stats": {"resolve": 0.5, "hoarding": 0.9},
                       "skills": {"hydroponics": 0.5}}]
        econ1 = EconomicState(wealth={"a": 0.3})
        econ2 = EconomicState(wealth={"a": 0.3})
        for _ in range(10):
            tick_economics(econ1, {"a": "farm"}, low_hoard, 0.6, "council", rng1)
            tick_economics(econ2, {"a": "farm"}, high_hoard, 0.6, "council", rng2)
        assert econ2.wealth["a"] >= econ1.wealth["a"], (
            f"hoarder {econ2.wealth['a']} should have >= non-hoarder {econ1.wealth['a']}")

    def test_surplus_accumulates(self) -> None:
        econ = EconomicState()
        colonists = self._make_colonist_data(3)
        rng = random.Random(42)
        for _ in range(5):
            actions = {c["id"]: "farm" for c in colonists}
            tick_economics(econ, actions, colonists, 0.6, "council", rng)
        assert econ.total_surplus_distributed > 0.0


# ── compute_wealth_effects ─────────────────────────────────────────────

class TestWealthEffects:
    def test_rich_colonist_hoards_more(self) -> None:
        effects = compute_wealth_effects(0.9)
        assert effects["hoard"] > 0

    def test_poor_colonist_farms_more(self) -> None:
        effects = compute_wealth_effects(0.1)
        assert effects["farm"] > 0

    def test_middle_wealth_moderate(self) -> None:
        effects = compute_wealth_effects(0.5)
        assert abs(effects["farm"]) < 0.1


# ── inequality_trust_erosion ───────────────────────────────────────────

class TestTrustErosion:
    def test_low_inequality_no_erosion(self) -> None:
        assert inequality_trust_erosion(0.2) == 0.0

    def test_high_inequality_erodes_trust(self) -> None:
        erosion = inequality_trust_erosion(0.8)
        assert erosion < 0

    def test_threshold_at_0_3(self) -> None:
        assert inequality_trust_erosion(0.29) == 0.0
        assert inequality_trust_erosion(0.31) < 0
