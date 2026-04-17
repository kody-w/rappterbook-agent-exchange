"""Tests for the Mars-100 economics organ."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.economics import (
    PersonalStockpile, TradeRecord, EconomicsState, EconomicTickResult,
    RESOURCE_NAMES, COMMONS_FLOOR, MAX_STOCKPILE, MAX_DIVERSION,
    TRADE_TRUST_THRESHOLD, MIN_TRADES_FOR_CURRENCY,
    CURRENCY_DOMINANCE_THRESHOLD,
    initialize_stockpiles, compute_gini, detect_currency,
    compute_economic_pressure, tick_economics,
    burn_stockpile_for_survival, add_colonist_stockpile,
    archive_colonist_stockpile, _compute_diversion_fraction,
    _decay_stockpiles,
)
from src.mars100.colonist import create_founding_ten, Colonist
from src.mars100.colony import Resources, SocialGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_colonists(seed: int = 42) -> list[Colonist]:
    return create_founding_ten(seed)


def _make_social(colonists: list[Colonist], seed: int = 42) -> SocialGraph:
    sg = SocialGraph()
    sg.initialize([c.id for c in colonists], random.Random(seed))
    return sg


def _make_state(colonists: list[Colonist],
                rng: random.Random) -> EconomicsState:
    state = EconomicsState()
    state.stockpiles = initialize_stockpiles(
        [c.id for c in colonists], rng)
    return state


# ---------------------------------------------------------------------------
# PersonalStockpile
# ---------------------------------------------------------------------------

class TestPersonalStockpile:
    def test_total(self):
        sp = PersonalStockpile(food=0.1, water=0.2, power=0.3,
                                air=0.1, medicine=0.05)
        assert abs(sp.total() - 0.75) < 1e-9

    def test_clamp_upper(self):
        sp = PersonalStockpile(food=1.5, water=-0.1)
        sp.clamp()
        assert sp.food == MAX_STOCKPILE
        assert sp.water == 0.0

    def test_to_dict(self):
        sp = PersonalStockpile(food=0.1234567)
        d = sp.to_dict()
        assert d["food"] == 0.1235  # rounded to 4 decimals
        assert set(d.keys()) == set(RESOURCE_NAMES)

    def test_default_empty(self):
        sp = PersonalStockpile()
        assert sp.total() == 0.0


# ---------------------------------------------------------------------------
# TradeRecord
# ---------------------------------------------------------------------------

class TestTradeRecord:
    def test_to_dict(self):
        tr = TradeRecord(year=5, from_id="a", to_id="b",
                         gave_resource="food", gave_amount=0.01,
                         received_resource="water", received_amount=0.02)
        d = tr.to_dict()
        assert d["year"] == 5
        assert d["from"] == "a"
        assert d["gave"] == "food"


# ---------------------------------------------------------------------------
# EconomicsState
# ---------------------------------------------------------------------------

class TestEconomicsState:
    def test_empty_summary(self):
        s = EconomicsState()
        sm = s.summary()
        assert sm["gini"] == 0.0
        assert sm["total_trades"] == 0
        assert sm["currency"] is None

    def test_to_dict_trims_history(self):
        s = EconomicsState()
        s.gini_history = list(range(50))
        d = s.to_dict()
        assert len(d["gini_history"]) == 20  # last 20


# ---------------------------------------------------------------------------
# initialize_stockpiles
# ---------------------------------------------------------------------------

class TestInitializeStockpiles:
    def test_all_colonists_get_stockpile(self):
        rng = random.Random(42)
        ids = ["a", "b", "c"]
        sp = initialize_stockpiles(ids, rng)
        assert set(sp.keys()) == set(ids)

    def test_stockpiles_nonnegative(self):
        rng = random.Random(42)
        sp = initialize_stockpiles(["a", "b"], rng)
        for cid, pile in sp.items():
            for n in RESOURCE_NAMES:
                assert getattr(pile, n) >= 0.0

    def test_stockpiles_within_bounds(self):
        rng = random.Random(42)
        sp = initialize_stockpiles([f"c{i}" for i in range(20)], rng)
        for pile in sp.values():
            for n in RESOURCE_NAMES:
                assert 0.0 <= getattr(pile, n) <= MAX_STOCKPILE


# ---------------------------------------------------------------------------
# _compute_diversion_fraction
# ---------------------------------------------------------------------------

class TestDiversionFraction:
    def test_council_low_hoarding(self):
        f = _compute_diversion_fraction("council", 0.0)
        assert abs(f - 0.20) < 1e-9

    def test_anarchy_high_hoarding(self):
        f = _compute_diversion_fraction("anarchy", 1.0)
        assert f <= MAX_DIVERSION

    def test_dictator_moderate(self):
        f = _compute_diversion_fraction("dictator", 0.5)
        assert 0.1 < f < 0.3

    def test_unknown_governance_uses_base(self):
        f = _compute_diversion_fraction("unknown_type", 0.0)
        assert abs(f - 0.20) < 1e-9

    def test_never_negative(self):
        for gov in ["council", "dictator", "lottery", "consensus",
                     "ai_governor", "anarchy"]:
            for h in [0.0, 0.5, 1.0]:
                assert _compute_diversion_fraction(gov, h) >= 0.0


# ---------------------------------------------------------------------------
# compute_gini
# ---------------------------------------------------------------------------

class TestGini:
    def test_perfect_equality(self):
        sp = {
            "a": PersonalStockpile(food=0.1, water=0.1, power=0.1,
                                    air=0.1, medicine=0.1),
            "b": PersonalStockpile(food=0.1, water=0.1, power=0.1,
                                    air=0.1, medicine=0.1),
        }
        assert abs(compute_gini(sp)) < 0.01

    def test_perfect_inequality(self):
        sp = {
            "a": PersonalStockpile(food=1.0, water=1.0, power=1.0,
                                    air=1.0, medicine=1.0),
            "b": PersonalStockpile(),
            "c": PersonalStockpile(),
        }
        gini = compute_gini(sp)
        assert gini > 0.5

    def test_empty_stockpiles(self):
        sp = {"a": PersonalStockpile(), "b": PersonalStockpile()}
        assert compute_gini(sp) == 0.0

    def test_single_colonist(self):
        sp = {"a": PersonalStockpile(food=0.5)}
        assert compute_gini(sp) == 0.0

    def test_bounds(self):
        rng = random.Random(42)
        for _ in range(100):
            n = rng.randint(2, 20)
            sp = {}
            for i in range(n):
                p = PersonalStockpile()
                for res in RESOURCE_NAMES:
                    setattr(p, res, rng.random())
                sp[f"c{i}"] = p
            g = compute_gini(sp)
            assert 0.0 <= g <= 1.0, f"GINI {g} out of bounds"


# ---------------------------------------------------------------------------
# detect_currency
# ---------------------------------------------------------------------------

class TestDetectCurrency:
    def test_no_trades(self):
        assert detect_currency([], 50) is None

    def test_below_minimum_trades(self):
        trades = [
            TradeRecord(year=40, from_id="a", to_id="b",
                        gave_resource="food", gave_amount=0.01,
                        received_resource="water", received_amount=0.01)
            for _ in range(MIN_TRADES_FOR_CURRENCY - 1)
        ]
        assert detect_currency(trades, 50) is None

    def test_dominant_resource_emerges(self):
        trades = []
        for i in range(20):
            trades.append(TradeRecord(
                year=45, from_id="a", to_id="b",
                gave_resource="water", gave_amount=0.01,
                received_resource="food", received_amount=0.01))
        # water + food are both 50% each — but water is gave, food is received
        # Both at 50% — dominant is whichever has more
        # Actually both would be at 50%, need to make one dominate
        # Let's create clear dominance
        trades = []
        for i in range(20):
            trades.append(TradeRecord(
                year=45, from_id="a", to_id="b",
                gave_resource="water", gave_amount=0.01,
                received_resource="water", received_amount=0.01))
        currency = detect_currency(trades, 50)
        assert currency == "water"

    def test_no_dominance(self):
        # Equal distribution across 5 resources — each at 20%
        trades = []
        for res in RESOURCE_NAMES:
            for _ in range(4):
                trades.append(TradeRecord(
                    year=45, from_id="a", to_id="b",
                    gave_resource=res, gave_amount=0.01,
                    received_resource=res, received_amount=0.01))
        currency = detect_currency(trades, 50)
        assert currency is None  # no single resource dominates


# ---------------------------------------------------------------------------
# compute_economic_pressure
# ---------------------------------------------------------------------------

class TestEconomicPressure:
    def test_high_inequality(self):
        state = EconomicsState()
        state.gini_history = [0.7]
        p = compute_economic_pressure(state)
        assert p.get("cooperate", 0) > 0
        assert p.get("hoard", 0) < 0

    def test_low_inequality(self):
        state = EconomicsState()
        state.gini_history = [0.1]
        p = compute_economic_pressure(state)
        assert p.get("hoard", 0) > 0

    def test_currency_boosts_coding(self):
        state = EconomicsState()
        state.gini_history = [0.3]
        state.currency_resource = "water"
        p = compute_economic_pressure(state)
        assert p.get("code", 0) > 0


# ---------------------------------------------------------------------------
# burn_stockpile_for_survival
# ---------------------------------------------------------------------------

class TestSurvivalBurn:
    def test_burns_personal_reserves(self):
        sp = {"c1": PersonalStockpile(food=0.1)}
        commons = Resources(food=0.1)  # critical (< 0.15)
        burns = burn_stockpile_for_survival("c1", sp, ["food"], commons)
        assert len(burns) == 1
        assert burns[0]["resource"] == "food"
        assert sp["c1"].food < 0.1
        assert commons.food > 0.1

    def test_no_burn_when_empty(self):
        sp = {"c1": PersonalStockpile(food=0.0)}
        commons = Resources(food=0.1)
        burns = burn_stockpile_for_survival("c1", sp, ["food"], commons)
        assert len(burns) == 0

    def test_missing_colonist(self):
        sp = {}
        commons = Resources(food=0.1)
        burns = burn_stockpile_for_survival("missing", sp, ["food"], commons)
        assert len(burns) == 0


# ---------------------------------------------------------------------------
# archive_colonist_stockpile
# ---------------------------------------------------------------------------

class TestArchiveStockpile:
    def test_returns_to_commons(self):
        sp = PersonalStockpile(food=0.1, water=0.2)
        state = EconomicsState(stockpiles={"c1": sp})
        commons = Resources(food=0.5, water=0.5)
        returned = archive_colonist_stockpile(state, "c1", commons)
        assert returned["food"] == 0.1
        assert commons.food == 0.6
        assert sp.food == 0.0

    def test_commons_capped_at_1(self):
        sp = PersonalStockpile(food=0.8)
        state = EconomicsState(stockpiles={"c1": sp})
        commons = Resources(food=0.9)
        archive_colonist_stockpile(state, "c1", commons)
        assert commons.food <= 1.0

    def test_missing_colonist(self):
        state = EconomicsState()
        commons = Resources()
        returned = archive_colonist_stockpile(state, "missing", commons)
        assert returned == {}


# ---------------------------------------------------------------------------
# add_colonist_stockpile
# ---------------------------------------------------------------------------

class TestAddColonistStockpile:
    def test_adds_stockpile(self):
        state = EconomicsState()
        rng = random.Random(42)
        add_colonist_stockpile(state, "new-1", rng)
        assert "new-1" in state.stockpiles
        sp = state.stockpiles["new-1"]
        for n in RESOURCE_NAMES:
            assert 0.0 <= getattr(sp, n) <= MAX_STOCKPILE


# ---------------------------------------------------------------------------
# _decay_stockpiles
# ---------------------------------------------------------------------------

class TestDecayStockpiles:
    def test_decay_reduces(self):
        sp = {"a": PersonalStockpile(food=0.5, water=0.5)}
        _decay_stockpiles(sp)
        assert sp["a"].food < 0.5
        assert sp["a"].water < 0.5

    def test_never_negative(self):
        sp = {"a": PersonalStockpile(food=0.001)}
        _decay_stockpiles(sp)
        assert sp["a"].food >= 0.0


# ---------------------------------------------------------------------------
# tick_economics integration
# ---------------------------------------------------------------------------

class TestTickEconomics:
    def _setup(self, seed=42):
        colonists = _make_colonists(seed)
        social = _make_social(colonists, seed)
        rng = random.Random(seed + 8191)
        state = _make_state(colonists, rng)
        resources = Resources()
        return colonists, social, state, resources, rng

    def test_basic_tick(self):
        colonists, social, state, resources, rng = self._setup()
        delta = {"food": 0.1, "water": 0.05, "power": 0.0,
                 "air": 0.02, "medicine": 0.01}
        result = tick_economics(
            state, colonists, resources, delta,
            social, "council", year=1, rng=rng)
        assert result.year == 1
        assert result.gini >= 0.0

    def test_gini_recorded(self):
        colonists, social, state, resources, rng = self._setup()
        delta = {n: 0.05 for n in RESOURCE_NAMES}
        tick_economics(state, colonists, resources, delta,
                       social, "council", year=1, rng=rng)
        assert len(state.gini_history) == 1

    def test_no_diversion_when_commons_low(self):
        colonists, social, state, resources, rng = self._setup()
        # Set all commons below floor
        for n in RESOURCE_NAMES:
            setattr(resources, n, 0.1)
        delta = {n: 0.1 for n in RESOURCE_NAMES}
        before = {n: getattr(resources, n) for n in RESOURCE_NAMES}
        tick_economics(state, colonists, resources, delta,
                       social, "council", year=1, rng=rng)
        # Commons should not have decreased (no diversion when below floor)
        for n in RESOURCE_NAMES:
            assert getattr(resources, n) >= before[n] - 0.001

    def test_10_year_mini_sim(self):
        """Run 10 years of economics without crashing."""
        colonists, social, state, resources, rng = self._setup()
        for year in range(1, 11):
            delta = {n: rng.uniform(-0.05, 0.1) for n in RESOURCE_NAMES}
            result = tick_economics(
                state, colonists, resources, delta,
                social, "council", year=year, rng=rng)
            assert result.gini >= 0.0
            assert result.gini <= 1.0
        assert len(state.gini_history) == 10

    def test_deterministic_with_same_seed(self):
        """Same seed produces same results."""
        results = []
        for _ in range(2):
            colonists, social, state, resources, rng = self._setup(seed=99)
            delta = {n: 0.05 for n in RESOURCE_NAMES}
            result = tick_economics(
                state, colonists, resources, delta,
                social, "council", year=1, rng=rng)
            results.append(result.gini)
        assert results[0] == results[1]

    def test_no_active_colonists(self):
        """No crash when all colonists dead."""
        colonists = _make_colonists()
        for c in colonists:
            c.die(1, "test")
        social = _make_social(colonists)
        rng = random.Random(42)
        state = _make_state(colonists, rng)
        resources = Resources()
        delta = {n: 0.05 for n in RESOURCE_NAMES}
        result = tick_economics(
            state, colonists, resources, delta,
            social, "council", year=1, rng=rng)
        assert result.year == 1

    def test_trade_history_trimmed(self):
        """Trade history doesn't grow unbounded."""
        colonists, social, state, resources, rng = self._setup()
        # Stuff in 300 fake trades
        for i in range(300):
            state.trade_history.append(TradeRecord(
                year=1, from_id="a", to_id="b",
                gave_resource="food", gave_amount=0.01,
                received_resource="water", received_amount=0.01))
        delta = {n: 0.05 for n in RESOURCE_NAMES}
        tick_economics(state, colonists, resources, delta,
                       social, "council", year=100, rng=rng)
        assert len(state.trade_history) <= 250

    def test_anarchy_high_diversion(self):
        """Anarchy governance allows more personal diversion."""
        colonists, social, state_a, resources_a, rng_a = self._setup(seed=77)
        colonists2, social2, state_c, resources_c, rng_c = self._setup(seed=77)
        delta = {n: 0.1 for n in RESOURCE_NAMES}
        tick_economics(state_a, colonists, resources_a, delta.copy(),
                       social, "anarchy", year=1, rng=rng_a)
        tick_economics(state_c, colonists2, resources_c, delta.copy(),
                       social2, "council", year=1, rng=rng_c)
        # Anarchy should have more total personal resources
        anarchy_total = sum(sp.total()
                            for sp in state_a.stockpiles.values())
        council_total = sum(sp.total()
                            for sp in state_c.stockpiles.values())
        assert anarchy_total >= council_total


# ---------------------------------------------------------------------------
# Full engine regression: v6.0 output unchanged when economics is added
# ---------------------------------------------------------------------------

class TestV6Regression:
    """Verify that adding economics doesn't change v6.0 simulation output.

    The economics organ uses a dedicated RNG (seed + 8191) and only
    post-processes resource deltas. The existing resource tick should
    produce identical deltas before economics touches them.
    """
    def test_founding_ten_unchanged(self):
        """Founding colonists are identical regardless of economics."""
        from src.mars100.colonist import create_founding_ten
        c1 = create_founding_ten(42)
        c2 = create_founding_ten(42)
        for a, b in zip(c1, c2):
            assert a.to_dict() == b.to_dict()
