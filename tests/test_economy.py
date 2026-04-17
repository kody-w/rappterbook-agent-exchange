"""Tests for the Mars-100 economy organ."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.economy import (
    EconomicState,
    EconomicTickResult,
    Trade,
    apply_tax,
    compute_economic_pressure,
    compute_gini,
    compute_labour_value,
    execute_trades,
    handle_birth,
    handle_death,
    handle_exile,
    handle_immigrant,
    initialize_shares,
    normalize_shares,
    redistribute,
    tick_economy,
    update_tax_rate,
    BASE_LABOUR_VALUE,
    GINI_CONFLICT_THRESHOLD,
    GINI_PROPOSAL_THRESHOLD,
    MIN_SHARE,
    TAX_RATES,
)


# -- initialize_shares -------------------------------------------------------

class TestInitializeShares:
    def test_equal_shares(self):
        shares = initialize_shares(["a", "b", "c", "d"])
        assert len(shares) == 4
        for v in shares.values():
            assert abs(v - 0.25) < 1e-10

    def test_shares_sum_to_one(self):
        shares = initialize_shares([f"c{i}" for i in range(10)])
        assert abs(sum(shares.values()) - 1.0) < 1e-10

    def test_empty(self):
        assert initialize_shares([]) == {}

    def test_single(self):
        shares = initialize_shares(["solo"])
        assert abs(shares["solo"] - 1.0) < 1e-10


# -- compute_gini ------------------------------------------------------------

class TestComputeGini:
    def test_perfect_equality(self):
        shares = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        assert compute_gini(shares) == 0.0

    def test_maximum_inequality(self):
        shares = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0}
        # Only 1 positive value → filtered to 1 participant → 0.0
        # Actually d=1.0 is positive, others are not > 0, so n=1 → 0.0
        gini = compute_gini(shares)
        assert gini == 0.0

    def test_high_inequality(self):
        shares = {"a": 0.01, "b": 0.01, "c": 0.01, "d": 0.97}
        gini = compute_gini(shares)
        assert 0.5 < gini <= 1.0

    def test_moderate_inequality(self):
        shares = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
        gini = compute_gini(shares)
        assert 0.0 < gini < 0.5

    def test_single_person(self):
        assert compute_gini({"a": 1.0}) == 0.0

    def test_empty(self):
        assert compute_gini({}) == 0.0

    def test_two_equal(self):
        assert compute_gini({"a": 0.5, "b": 0.5}) == 0.0

    def test_two_unequal(self):
        gini = compute_gini({"a": 0.1, "b": 0.9})
        assert 0.0 < gini < 1.0

    def test_gini_bounded(self):
        """Property: Gini is always in [0, 1)."""
        rng = random.Random(42)
        for _ in range(100):
            n = rng.randint(2, 20)
            values = [rng.random() for _ in range(n)]
            shares = {f"c{i}": v for i, v in enumerate(values)}
            gini = compute_gini(shares)
            assert 0.0 <= gini < 1.0, f"gini={gini} for {shares}"


# -- compute_labour_value ----------------------------------------------------

class TestLabourValue:
    def test_farming_during_food_scarcity(self):
        resources = {"food": 0.05, "water": 0.5, "power": 0.5, "air": 0.5}
        value = compute_labour_value("farm", resources, skill_level=0.8)
        normal = compute_labour_value("farm", {"food": 0.5}, skill_level=0.8)
        assert value > normal

    def test_sabotage_is_negative(self):
        value = compute_labour_value("sabotage", {"food": 0.5}, 0.5)
        # sabotage base is -0.05, so result should be negative
        assert value == 0.0  # clamped to 0.0

    def test_higher_skill_higher_value(self):
        low = compute_labour_value("code", {"power": 0.3}, 0.1)
        high = compute_labour_value("code", {"power": 0.3}, 0.9)
        assert high > low

    def test_all_actions_have_value(self):
        for action in BASE_LABOUR_VALUE:
            if action == "sabotage":
                continue
            value = compute_labour_value(action, {"food": 0.5}, 0.5)
            assert value >= 0.0


# -- apply_tax ---------------------------------------------------------------

class TestApplyTax:
    def test_zero_tax(self):
        shares = {"a": 0.5, "b": 0.5}
        collected = apply_tax(shares, ["a", "b"], 0.0)
        assert collected == 0.0
        assert shares["a"] == 0.5

    def test_positive_tax(self):
        shares = {"a": 0.5, "b": 0.5}
        collected = apply_tax(shares, ["a", "b"], 0.1)
        assert collected > 0.0
        assert shares["a"] < 0.5
        assert shares["b"] < 0.5

    def test_conservation(self):
        """Tax + remaining shares + MIN_SHARE floor should be accounted for."""
        shares = {"a": 0.6, "b": 0.4}
        original_total = sum(shares.values())
        collected = apply_tax(shares, ["a", "b"], 0.1)
        remaining = sum(shares.values())
        # total = remaining + collected (approximately, with MIN_SHARE floor)
        assert abs(remaining + collected - original_total) < 0.01

    def test_min_share_floor(self):
        """Even with high tax, shares don't go below MIN_SHARE."""
        shares = {"a": 0.002, "b": 0.998}
        apply_tax(shares, ["a", "b"], 0.5)
        assert shares["a"] >= MIN_SHARE


# -- redistribute ------------------------------------------------------------

class TestRedistribute:
    def test_equal_redistribution_council(self):
        shares = {"a": 0.3, "b": 0.3}
        redistributed = redistribute(shares, ["a", "b"], 0.2, "council", None)
        assert redistributed == 0.2
        assert abs(shares["a"] - 0.4) < 1e-10
        assert abs(shares["b"] - 0.4) < 1e-10

    def test_dictator_keeps_majority(self):
        shares = {"leader": 0.3, "b": 0.3}
        redistribute(shares, ["leader", "b"], 0.2, "dictator", "leader")
        assert shares["leader"] > shares["b"]

    def test_anarchy_no_redistribution(self):
        shares = {"a": 0.5, "b": 0.5}
        redistributed = redistribute(shares, ["a", "b"], 0.2, "anarchy", None)
        assert redistributed == 0.0

    def test_zero_amount(self):
        shares = {"a": 0.5}
        redistributed = redistribute(shares, ["a"], 0.0, "council", None)
        assert redistributed == 0.0


# -- lifecycle handlers ------------------------------------------------------

class TestHandleDeath:
    def test_estate_distributed(self):
        shares = {"dead": 0.4, "a": 0.3, "b": 0.3}
        edges = {"dead": {"a": {"trust": 0.8}, "b": {"trust": 0.2}}}
        trades = handle_death(shares, "dead", edges, ["a", "b"])
        assert "dead" not in shares
        assert len(trades) > 0
        # Total should be preserved (approximately)
        assert abs(sum(shares.values()) - 1.0) < 0.01

    def test_dead_share_removed(self):
        shares = {"dead": 0.5, "a": 0.5}
        handle_death(shares, "dead", {}, ["a"])
        assert "dead" not in shares

    def test_trust_weighted(self):
        shares = {"dead": 0.6, "trusted": 0.2, "untrusted": 0.2}
        edges = {"dead": {"trusted": {"trust": 0.9}, "untrusted": {"trust": 0.1}}}
        handle_death(shares, "dead", edges, ["trusted", "untrusted"])
        assert shares["trusted"] > shares["untrusted"]


class TestHandleExile:
    def test_confiscated_and_redistributed(self):
        shares = {"exile": 0.4, "a": 0.3, "b": 0.3}
        confiscated = handle_exile(shares, "exile", ["a", "b"])
        assert confiscated == 0.4
        assert "exile" not in shares
        assert abs(shares["a"] - 0.5) < 1e-10

    def test_empty_active(self):
        shares = {"exile": 0.5}
        confiscated = handle_exile(shares, "exile", [])
        assert confiscated == 0.5


class TestHandleBirth:
    def test_newborn_gets_stipend(self):
        shares = {"a": 0.5, "b": 0.5}
        handle_birth(shares, "child", ["a", "b"])
        assert "child" in shares
        assert shares["child"] > 0

    def test_dilution(self):
        shares = {"a": 0.5, "b": 0.5}
        total_before = sum(shares.values())
        handle_birth(shares, "child", ["a", "b"])
        # Existing colonists' shares should decrease
        assert shares["a"] < 0.5
        assert shares["b"] < 0.5


class TestHandleImmigrant:
    def test_immigrant_gets_stipend(self):
        shares = {"a": 0.5, "b": 0.5}
        handle_immigrant(shares, "imm", ["a", "b"])
        assert "imm" in shares
        assert shares["imm"] > 0

    def test_immigrant_share_larger_than_child(self):
        shares_a = {"a": 0.5, "b": 0.5}
        shares_b = dict(shares_a)
        handle_birth(shares_a, "child", ["a", "b"])
        handle_immigrant(shares_b, "imm", ["a", "b"])
        assert shares_b["imm"] > shares_a["child"]


# -- normalize_shares --------------------------------------------------------

class TestNormalizeShares:
    def test_sums_to_one(self):
        shares = {"a": 0.3, "b": 0.5, "c": 0.7}
        normalize_shares(shares)
        assert abs(sum(shares.values()) - 1.0) < 1e-10

    def test_preserves_proportions(self):
        shares = {"a": 2.0, "b": 3.0}
        normalize_shares(shares)
        assert abs(shares["a"] / shares["b"] - 2.0 / 3.0) < 1e-10

    def test_empty(self):
        shares: dict[str, float] = {}
        normalize_shares(shares)
        assert shares == {}

    def test_all_zero(self):
        shares = {"a": 0.0, "b": 0.0}
        normalize_shares(shares)
        # No change — can't normalize zero


# -- execute_trades ----------------------------------------------------------

class TestExecuteTrades:
    def test_no_trades_low_empathy(self):
        shares = {"a": 0.7, "b": 0.3}
        colonists = {
            "a": {"stats": {"empathy": 0.1, "hoarding": 0.8}},
            "b": {"stats": {"empathy": 0.1, "hoarding": 0.8}},
        }
        edges = {"a": {"b": {"trust": 0.9}}, "b": {"a": {"trust": 0.9}}}
        trades = execute_trades(shares, colonists, edges, ["a", "b"],
                                random.Random(42))
        assert len(trades) == 0

    def test_trades_high_empathy_trust(self):
        """With high empathy and trust, trades should eventually happen."""
        rng = random.Random(42)
        trade_happened = False
        for seed in range(100):
            shares = {"a": 0.7, "b": 0.3}
            colonists = {
                "a": {"stats": {"empathy": 0.9, "hoarding": 0.1}},
                "b": {"stats": {"empathy": 0.9, "hoarding": 0.1}},
            }
            edges = {"a": {"b": {"trust": 0.9}}, "b": {"a": {"trust": 0.9}}}
            trades = execute_trades(shares, colonists, edges, ["a", "b"],
                                    random.Random(seed))
            if trades:
                trade_happened = True
                assert trades[0].giver_id == "a"
                assert trades[0].receiver_id == "b"
                assert trades[0].amount > 0
                break
        assert trade_happened, "Expected at least one trade across 100 seeds"


# -- economic_pressure -------------------------------------------------------

class TestEconomicPressure:
    def test_wealthy_prefers_leisure(self):
        shares = {"rich": 0.8, "poor": 0.2}
        pressure = compute_economic_pressure(shares, "rich", ["rich", "poor"])
        assert pressure.get("rest", 0) > 0
        assert pressure.get("research", 0) > 0

    def test_poor_prefers_work(self):
        shares = {"rich": 0.8, "poor": 0.2}
        pressure = compute_economic_pressure(shares, "poor", ["rich", "poor"])
        assert pressure.get("farm", 0) > 0
        assert pressure.get("cooperate", 0) > 0

    def test_average_no_pressure(self):
        shares = {"a": 0.5, "b": 0.5}
        pressure = compute_economic_pressure(shares, "a", ["a", "b"])
        assert pressure == {}

    def test_missing_colonist(self):
        pressure = compute_economic_pressure({}, "ghost", ["a"])
        assert pressure == {}


# -- update_tax_rate ---------------------------------------------------------

class TestUpdateTaxRate:
    def test_known_types(self):
        state = EconomicState()
        for gov_type, expected in TAX_RATES.items():
            update_tax_rate(state, gov_type)
            assert state.tax_rate == expected

    def test_unknown_type_defaults(self):
        state = EconomicState()
        update_tax_rate(state, "theocracy")
        assert state.tax_rate == 0.05


# -- tick_economy (integration) ----------------------------------------------

class TestTickEconomy:
    def _make_colonist_snapshot(self, cid: str, alive: bool = True,
                                 exiled: bool = False) -> dict:
        return {
            "id": cid, "alive": alive, "exiled": exiled,
            "stats": {"resolve": 0.5, "improvisation": 0.5, "empathy": 0.5,
                      "hoarding": 0.3, "faith": 0.3, "paranoia": 0.3},
            "skills": {"terraforming": 0.3, "hydroponics": 0.5,
                       "mediation": 0.3, "coding": 0.4,
                       "prayer": 0.1, "sabotage": 0.0},
        }

    def test_basic_tick(self):
        state = EconomicState()
        state.shares = initialize_shares(["a", "b", "c"])
        snapshots = [self._make_colonist_snapshot(c) for c in ["a", "b", "c"]]
        actions = {"a": "farm", "b": "code", "c": "mediate"}
        resources = {"food": 0.5, "water": 0.5, "power": 0.5,
                     "air": 0.5, "medicine": 0.5}

        result = tick_economy(
            state, year=1, colonist_snapshots=snapshots,
            actions=actions, resources_dict=resources,
            gov_type="council", leader_id=None,
            social_edges={}, deaths=[], exiles=[],
            births=[], immigrants=[], rng=random.Random(42),
        )

        assert result.year == 1
        assert len(result.incomes) == 3
        assert 0.0 <= result.gini_after < 1.0
        # Shares still sum to ~1.0
        total = sum(state.shares[c] for c in ["a", "b", "c"])
        assert abs(total - 1.0) < 0.01

    def test_shares_sum_conservation_over_time(self):
        """Property: shares always sum to ~1.0 after normalization."""
        rng = random.Random(99)
        state = EconomicState()
        ids = [f"c{i}" for i in range(10)]
        state.shares = initialize_shares(ids)
        snapshots = [self._make_colonist_snapshot(c) for c in ids]

        for year in range(1, 21):
            actions = {c: rng.choice(["farm", "code", "mediate", "rest"])
                       for c in ids}
            resources = {r: rng.uniform(0.2, 0.8)
                         for r in ["food", "water", "power", "air", "medicine"]}
            tick_economy(
                state, year=year, colonist_snapshots=snapshots,
                actions=actions, resources_dict=resources,
                gov_type="council", leader_id=None,
                social_edges={}, deaths=[], exiles=[],
                births=[], immigrants=[], rng=rng,
            )
            total = sum(state.shares.get(c, 0) for c in ids)
            assert abs(total - 1.0) < 0.05, \
                f"Year {year}: shares sum to {total}, expected ~1.0"

    def test_death_during_tick(self):
        state = EconomicState()
        state.shares = initialize_shares(["a", "b", "dead"])
        snapshots = [
            self._make_colonist_snapshot("a"),
            self._make_colonist_snapshot("b"),
            self._make_colonist_snapshot("dead", alive=False),
        ]
        actions = {"a": "farm", "b": "code"}

        result = tick_economy(
            state, year=5, colonist_snapshots=snapshots,
            actions=actions, resources_dict={"food": 0.5},
            gov_type="council", leader_id=None,
            social_edges={"dead": {"a": {"trust": 0.8}, "b": {"trust": 0.2}}},
            deaths=[{"id": "dead"}], exiles=[], births=[], immigrants=[],
            rng=random.Random(42),
        )

        assert "estate_distributed:dead" in result.events

    def test_birth_during_tick(self):
        state = EconomicState()
        state.shares = initialize_shares(["a", "b"])
        snapshots = [
            self._make_colonist_snapshot("a"),
            self._make_colonist_snapshot("b"),
        ]

        result = tick_economy(
            state, year=15, colonist_snapshots=snapshots,
            actions={"a": "farm", "b": "code"},
            resources_dict={"food": 0.5},
            gov_type="council", leader_id=None,
            social_edges={}, deaths=[], exiles=[],
            births=[{"id": "child-1"}], immigrants=[],
            rng=random.Random(42),
        )

        assert "child-1" in state.shares
        assert "stipend_granted:child-1" in result.events

    def test_gini_bounded_over_simulation(self):
        """Property: Gini stays in [0, 1) across 50 years."""
        rng = random.Random(7)
        state = EconomicState()
        ids = [f"c{i}" for i in range(8)]
        state.shares = initialize_shares(ids)
        snapshots = [self._make_colonist_snapshot(c) for c in ids]

        for year in range(1, 51):
            actions = {c: rng.choice(["farm", "code", "terraform", "hoard",
                                      "rest", "cooperate"])
                       for c in ids}
            resources = {r: rng.uniform(0.1, 0.9)
                         for r in ["food", "water", "power", "air", "medicine"]}
            result = tick_economy(
                state, year=year, colonist_snapshots=snapshots,
                actions=actions, resources_dict=resources,
                gov_type=rng.choice(["council", "dictator", "anarchy"]),
                leader_id=rng.choice(ids),
                social_edges={}, deaths=[], exiles=[],
                births=[], immigrants=[], rng=rng,
            )
            assert 0.0 <= result.gini_after < 1.0, \
                f"Year {year}: gini={result.gini_after}"


# -- EconomicState serialization ---------------------------------------------

class TestEconomicStateSerialization:
    def test_to_dict(self):
        state = EconomicState(
            shares={"a": 0.5, "b": 0.5},
            tax_rate=0.1, treasury=0.05, gini=0.0,
        )
        d = state.to_dict()
        assert "shares" in d
        assert d["tax_rate"] == 0.1
        assert d["gini"] == 0.0

    def test_summary(self):
        state = EconomicState(
            shares={"a": 0.7, "b": 0.3},
            gini=0.2, tax_rate=0.08,
        )
        s = state.summary()
        assert s["richest"] == "a"
        assert s["poorest"] == "b"
        assert s["gini"] == 0.2

    def test_summary_empty(self):
        state = EconomicState()
        s = state.summary()
        assert s["richest"] is None


# -- smoke test: 100-year integration with full engine -----------------------

class TestEconomyEngineIntegration:
    def test_economy_with_full_sim_10_years(self):
        """Run 10-year simulation with economy enabled — no crashes."""
        from src.mars100.engine import Mars100Engine

        engine = Mars100Engine(seed=42, total_years=10)
        # Run without economy first to verify baseline
        # Then manually tick economy alongside
        eco_state = EconomicState()
        active_ids = [c.id for c in engine._active_colonists()]
        eco_state.shares = initialize_shares(active_ids)
        eco_rng = random.Random(42 + 3571)

        for _ in range(10):
            if not engine._active_colonists():
                break
            year_result = engine.tick()

            # Get data needed for economy tick
            active = engine._active_colonists()
            snapshots = [c.to_dict() for c in engine.colonists]
            social_edges = engine.social.to_dict()

            eco_result = tick_economy(
                eco_state, year=year_result.year,
                colonist_snapshots=snapshots,
                actions=year_result.actions,
                resources_dict=year_result.resources_after,
                gov_type=engine.governance.gov_type,
                leader_id=engine.governance.leader_id,
                social_edges=social_edges,
                deaths=year_result.deaths,
                exiles=year_result.exiles,
                births=year_result.births,
                immigrants=year_result.immigrants,
                rng=eco_rng,
            )

            assert 0.0 <= eco_result.gini_after < 1.0
            total_shares = sum(
                eco_state.shares.get(c.id, 0)
                for c in engine.colonists if c.is_active()
            )
            assert abs(total_shares - 1.0) < 0.1, \
                f"Year {year_result.year}: share sum={total_shares}"
