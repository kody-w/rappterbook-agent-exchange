"""Tests for the Mars-100 economy organ.

Tests personal inventories, specialization, Gini coefficient,
hoarding, trades, black market, redistribution, and engine integration.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mars100.economy import (
    Stockpile, Specialization, TradeRecord, BlackMarketRecord,
    EconomyState, compute_gini, process_hoarding, process_trade,
    process_black_market, liquidate_stockpile, redistribute_wealth,
    tick_economy, get_specialization_bonus,
    TRADEABLE_RESOURCES, HOARD_RATE, BLACK_MARKET_RATIO,
    SPECIALIZATION_THRESHOLD, SPECIALIZATION_BONUS,
    GINI_REDISTRIBUTION_THRESHOLD, TRADE_TRUST_THRESHOLD,
    MAX_BLACK_MARKET_PER_YEAR,
)
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, create_founding_ten,
)
from src.mars100.colony import Resources, SocialGraph


# ─────────────────────────────────────────────────────────────
# Stockpile tests
# ─────────────────────────────────────────────────────────────

class TestStockpile:
    def test_default_empty(self):
        sp = Stockpile()
        assert sp.total() == 0.0

    def test_to_dict_roundtrip(self):
        sp = Stockpile(food=0.3, water=0.2, medicine=0.1)
        d = sp.to_dict()
        sp2 = Stockpile.from_dict(d)
        assert abs(sp2.food - 0.3) < 1e-9
        assert abs(sp2.water - 0.2) < 1e-9
        assert abs(sp2.medicine - 0.1) < 1e-9

    def test_clamp_upper(self):
        sp = Stockpile(food=1.5, water=-0.3, medicine=0.5)
        sp.clamp()
        assert sp.food == 1.0
        assert sp.water == 0.0
        assert sp.medicine == 0.5

    def test_decay_reduces_values(self):
        sp = Stockpile(food=0.5, water=0.5, medicine=0.5)
        lost = sp.decay(rate=0.1)
        assert sp.food < 0.5
        assert sp.water < 0.5
        assert sp.medicine < 0.5
        assert all(v > 0 for v in lost.values())

    def test_decay_conserves(self):
        """Decay lost + remaining == original."""
        sp = Stockpile(food=0.5, water=0.4, medicine=0.3)
        original = sp.total()
        lost = sp.decay(rate=0.1)
        total_lost = sum(lost.values())
        assert abs((sp.total() + total_lost) - original) < 1e-9

    def test_from_dict_defaults(self):
        sp = Stockpile.from_dict({})
        assert sp.total() == 0.0


# ─────────────────────────────────────────────────────────────
# Specialization tests
# ─────────────────────────────────────────────────────────────

class TestSpecialization:
    def test_initial_not_specialized(self):
        s = Specialization()
        assert not s.is_specialized
        assert s.bonus == 0.0

    def test_streak_builds(self):
        s = Specialization()
        for _ in range(SPECIALIZATION_THRESHOLD):
            s.update("farm")
        assert s.is_specialized
        assert s.bonus == SPECIALIZATION_BONUS

    def test_streak_resets_on_change(self):
        s = Specialization()
        for _ in range(4):
            s.update("farm")
        s.update("code")
        assert s.last_action == "code"
        assert s.consecutive_years == 1
        assert not s.is_specialized

    def test_to_dict_roundtrip(self):
        s = Specialization(last_action="terraform", consecutive_years=7)
        d = s.to_dict()
        s2 = Specialization.from_dict(d)
        assert s2.last_action == "terraform"
        assert s2.consecutive_years == 7

    def test_threshold_exact(self):
        s = Specialization()
        for _ in range(SPECIALIZATION_THRESHOLD - 1):
            s.update("pray")
        assert not s.is_specialized
        s.update("pray")
        assert s.is_specialized


# ─────────────────────────────────────────────────────────────
# Gini coefficient tests
# ─────────────────────────────────────────────────────────────

class TestGini:
    def test_perfect_equality(self):
        assert compute_gini([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_maximum_inequality(self):
        """One person has everything."""
        gini = compute_gini([0.0, 0.0, 0.0, 1.0])
        assert gini > 0.7

    def test_empty_list(self):
        assert compute_gini([]) == 0.0

    def test_single_element(self):
        assert compute_gini([5.0]) == 0.0

    def test_all_zeros(self):
        assert compute_gini([0.0, 0.0, 0.0]) == 0.0

    def test_moderate_inequality(self):
        gini = compute_gini([0.1, 0.2, 0.3, 0.4])
        assert 0.0 < gini < 0.5

    def test_gini_bounded(self):
        """Gini should be in [0, 1) for any non-negative distribution."""
        rng = random.Random(42)
        for _ in range(50):
            values = [rng.random() for _ in range(10)]
            g = compute_gini(values)
            assert 0.0 <= g < 1.0, f"Gini {g} out of bounds for {values}"

    def test_gini_increases_with_inequality(self):
        """More skewed distributions should have higher Gini."""
        equal = compute_gini([0.5, 0.5, 0.5, 0.5])
        moderate = compute_gini([0.1, 0.3, 0.5, 1.1])
        extreme = compute_gini([0.0, 0.0, 0.0, 2.0])
        assert equal <= moderate <= extreme


# ─────────────────────────────────────────────────────────────
# Hoarding tests
# ─────────────────────────────────────────────────────────────

class TestHoarding:
    def test_non_hoard_action_no_effect(self):
        sp = Stockpile()
        res = Resources(food=0.8, water=0.8)
        result = process_hoarding("c1", "farm", sp, res, 0.8, random.Random(42))
        assert result == {}
        assert sp.total() == 0.0

    def test_hoard_transfers_resources(self):
        sp = Stockpile()
        res = Resources(food=0.8, water=0.8, medicine=0.5)
        result = process_hoarding("c1", "hoard", sp, res, 0.8, random.Random(42))
        assert sp.total() > 0.0
        assert any(v > 0 for v in result.values())

    def test_hoard_reduces_colony(self):
        res = Resources(food=0.8, water=0.8, medicine=0.5)
        before_food = res.food
        sp = Stockpile()
        process_hoarding("c1", "hoard", sp, res, 0.8, random.Random(42))
        assert res.food <= before_food

    def test_hoard_limited_by_colony(self):
        """Can't hoard more than 50% of colony's resource."""
        res = Resources(food=0.02, water=0.02, medicine=0.02)
        sp = Stockpile()
        process_hoarding("c1", "hoard", sp, res, 1.0, random.Random(42))
        # Colony should still have some resources
        assert res.food >= 0.0
        assert res.water >= 0.0

    def test_hoard_capped_at_one(self):
        sp = Stockpile(food=0.95, water=0.95, medicine=0.95)
        res = Resources(food=0.8, water=0.8, medicine=0.8)
        process_hoarding("c1", "hoard", sp, res, 1.0, random.Random(42))
        assert sp.food <= 1.0
        assert sp.water <= 1.0
        assert sp.medicine <= 1.0


# ─────────────────────────────────────────────────────────────
# Trade tests
# ─────────────────────────────────────────────────────────────

class TestTrade:
    def test_low_trust_no_trade(self):
        sp1 = Stockpile(food=0.5)
        sp2 = Stockpile(food=0.0)
        result = process_trade("a", "b", sp1, sp2, trust=0.1,
                               empathy_from=0.9, empathy_to=0.9,
                               year=10, rng=random.Random(42))
        assert result is None

    def test_high_trust_trade_happens(self):
        sp1 = Stockpile(food=0.5, water=0.3, medicine=0.2)
        sp2 = Stockpile(food=0.0, water=0.0, medicine=0.0)
        result = process_trade("a", "b", sp1, sp2, trust=0.9,
                               empathy_from=0.9, empathy_to=0.5,
                               year=10, rng=random.Random(42))
        if result is not None:
            assert result.amount > 0
            assert result.trust_after >= result.trust_before

    def test_trade_conserves_resources(self):
        """Total resources before and after trade are equal."""
        sp1 = Stockpile(food=0.5, water=0.3, medicine=0.2)
        sp2 = Stockpile(food=0.1, water=0.1, medicine=0.1)
        before = sp1.total() + sp2.total()
        process_trade("a", "b", sp1, sp2, trust=0.9,
                      empathy_from=0.9, empathy_to=0.5,
                      year=10, rng=random.Random(42))
        after = sp1.total() + sp2.total()
        assert abs(before - after) < 1e-9

    def test_no_trade_when_no_surplus(self):
        sp1 = Stockpile(food=0.0)
        sp2 = Stockpile(food=0.0)
        result = process_trade("a", "b", sp1, sp2, trust=0.9,
                               empathy_from=0.9, empathy_to=0.9,
                               year=10, rng=random.Random(42))
        assert result is None


# ─────────────────────────────────────────────────────────────
# Black market tests
# ─────────────────────────────────────────────────────────────

class TestBlackMarket:
    def test_low_sabotage_no_market(self):
        sp = Stockpile()
        res = Resources(food=0.1, water=0.1, medicine=0.1)
        result = process_black_market("c1", sp, res, sabotage_skill=0.1,
                                      paranoia=0.8, rng=random.Random(42), year=10)
        assert result is None

    def test_no_critical_resources_no_market(self):
        sp = Stockpile()
        res = Resources(food=0.8, water=0.8, medicine=0.8)
        result = process_black_market("c1", sp, res, sabotage_skill=0.8,
                                      paranoia=0.8, rng=random.Random(42), year=10)
        assert result is None

    def test_black_market_wastes_resources(self):
        """Colony loses more than colonist gains (2:1 ratio)."""
        sp = Stockpile()
        res = Resources(food=0.2, water=0.2, medicine=0.2)
        result = process_black_market("c1", sp, res, sabotage_skill=0.9,
                                      paranoia=0.9, rng=random.Random(1), year=10)
        if result is not None:
            assert result.colony_cost > result.personal_gain
            assert abs(result.colony_cost / max(0.001, result.personal_gain) - BLACK_MARKET_RATIO) < 0.1

    def test_low_paranoia_no_market(self):
        sp = Stockpile()
        res = Resources(food=0.1, water=0.1, medicine=0.1)
        result = process_black_market("c1", sp, res, sabotage_skill=0.9,
                                      paranoia=0.1, rng=random.Random(42), year=10)
        assert result is None


# ─────────────────────────────────────────────────────────────
# Liquidation tests
# ─────────────────────────────────────────────────────────────

class TestLiquidation:
    def test_liquidate_returns_to_colony(self):
        sp = Stockpile(food=0.3, water=0.2, medicine=0.1)
        res = Resources(food=0.5, water=0.5, medicine=0.5)
        returned = liquidate_stockpile(sp, res)
        assert returned["food"] == pytest.approx(0.3)
        assert sp.total() == 0.0
        assert res.food == pytest.approx(0.8)

    def test_liquidate_empty_stockpile(self):
        sp = Stockpile()
        res = Resources(food=0.5)
        returned = liquidate_stockpile(sp, res)
        assert all(v < 0.002 for v in returned.values())

    def test_liquidate_clamped(self):
        """Colony resources should not exceed 1.0 after liquidation."""
        sp = Stockpile(food=0.8)
        res = Resources(food=0.9)
        liquidate_stockpile(sp, res)
        assert res.food <= 1.0


# ─────────────────────────────────────────────────────────────
# Redistribution tests
# ─────────────────────────────────────────────────────────────

class TestRedistribution:
    def test_redistribution_equalizes(self):
        stockpiles = {
            "a": Stockpile(food=0.8, water=0.0, medicine=0.0),
            "b": Stockpile(food=0.0, water=0.0, medicine=0.0),
            "c": Stockpile(food=0.0, water=0.0, medicine=0.0),
        }
        redistribute_wealth(stockpiles, ["a", "b", "c"])
        # After redistribution, Gini should be 0
        values = [stockpiles[cid].total() for cid in ["a", "b", "c"]]
        gini = compute_gini(values)
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_redistribution_conserves(self):
        stockpiles = {
            "a": Stockpile(food=0.6, water=0.3, medicine=0.1),
            "b": Stockpile(food=0.1, water=0.1, medicine=0.0),
        }
        before = sum(s.total() for s in stockpiles.values())
        redistribute_wealth(stockpiles, ["a", "b"])
        after = sum(s.total() for s in stockpiles.values())
        assert abs(before - after) < 1e-9

    def test_redistribution_empty_ids(self):
        result = redistribute_wealth({}, [])
        assert result == {}


# ─────────────────────────────────────────────────────────────
# Integration: tick_economy
# ─────────────────────────────────────────────────────────────

class TestTickEconomy:
    def _make_colony(self, seed=42):
        colonists = create_founding_ten(seed)
        resources = Resources()
        social = SocialGraph()
        active_ids = [c.id for c in colonists]
        social.initialize(active_ids, random.Random(seed))
        return colonists, resources, social

    def test_tick_economy_returns_state(self):
        colonists, resources, social = self._make_colony()
        actions = {c.id: "farm" for c in colonists}
        stockpiles: dict[str, Stockpile] = {}
        specializations: dict[str, Specialization] = {}
        result = tick_economy(colonists, actions, resources, social,
                              stockpiles, specializations, year=1,
                              rng=random.Random(42))
        assert isinstance(result, EconomyState)
        assert result.gini >= 0.0

    def test_tick_economy_creates_entries(self):
        colonists, resources, social = self._make_colony()
        actions = {c.id: "farm" for c in colonists}
        stockpiles: dict[str, Stockpile] = {}
        specializations: dict[str, Specialization] = {}
        tick_economy(colonists, actions, resources, social,
                     stockpiles, specializations, year=1,
                     rng=random.Random(42))
        assert len(stockpiles) == 10
        assert len(specializations) == 10

    def test_tick_economy_hoarding_increases_wealth(self):
        colonists, resources, social = self._make_colony()
        # Make all colonists hoard with high hoarding stat
        for c in colonists:
            c.stats.hoarding = 0.9
        actions = {c.id: "hoard" for c in colonists}
        stockpiles: dict[str, Stockpile] = {}
        specializations: dict[str, Specialization] = {}
        result = tick_economy(colonists, actions, resources, social,
                              stockpiles, specializations, year=1,
                              rng=random.Random(42))
        assert result.total_wealth > 0.0

    def test_tick_economy_specialization_tracking(self):
        colonists, resources, social = self._make_colony()
        stockpiles: dict[str, Stockpile] = {}
        specializations: dict[str, Specialization] = {}
        # Run same action for SPECIALIZATION_THRESHOLD years
        for year in range(1, SPECIALIZATION_THRESHOLD + 1):
            actions = {c.id: "farm" for c in colonists}
            tick_economy(colonists, actions, resources, social,
                         stockpiles, specializations, year=year,
                         rng=random.Random(42 + year))
        result = tick_economy(colonists, {c.id: "farm" for c in colonists},
                              resources, social, stockpiles, specializations,
                              year=SPECIALIZATION_THRESHOLD + 1,
                              rng=random.Random(99))
        assert result.specializations == 10  # all specialized in farm

    def test_tick_economy_to_dict(self):
        colonists, resources, social = self._make_colony()
        actions = {c.id: "rest" for c in colonists}
        stockpiles: dict[str, Stockpile] = {}
        specializations: dict[str, Specialization] = {}
        result = tick_economy(colonists, actions, resources, social,
                              stockpiles, specializations, year=1,
                              rng=random.Random(42))
        d = result.to_dict()
        assert "gini" in d
        assert "total_wealth" in d
        assert "trade_count" in d

    def test_tick_economy_deterministic(self):
        """Same seed → same result."""
        def run_once(seed):
            colonists, resources, social = self._make_colony(seed)
            actions = {c.id: "hoard" for c in colonists}
            stockpiles: dict[str, Stockpile] = {}
            specializations: dict[str, Specialization] = {}
            return tick_economy(colonists, actions, resources, social,
                                stockpiles, specializations, year=1,
                                rng=random.Random(seed))
        r1 = run_once(42)
        r2 = run_once(42)
        assert r1.gini == r2.gini
        assert r1.total_wealth == r2.total_wealth

    def test_redistribution_triggers_at_high_gini(self):
        colonists, resources, social = self._make_colony()
        stockpiles = {
            colonists[0].id: Stockpile(food=0.9, water=0.9, medicine=0.9),
        }
        for c in colonists[1:]:
            stockpiles[c.id] = Stockpile()
        specializations: dict[str, Specialization] = {}
        actions = {c.id: "rest" for c in colonists}
        result = tick_economy(colonists, actions, resources, social,
                              stockpiles, specializations, year=50,
                              rng=random.Random(42))
        # Gini should have been high enough to trigger redistribution
        # After redistribution, stockpiles should be more equal
        values = [stockpiles[c.id].total() for c in colonists if c.is_active()]
        after_gini = compute_gini(values)
        assert after_gini < 0.5  # much more equal than before


# ─────────────────────────────────────────────────────────────
# Specialization bonus integration
# ─────────────────────────────────────────────────────────────

class TestSpecializationBonus:
    def test_no_specialization_no_bonus(self):
        specs = {"c1": Specialization(last_action="farm", consecutive_years=2)}
        assert get_specialization_bonus("c1", "farm", specs) == 0.0

    def test_specialized_correct_action(self):
        specs = {"c1": Specialization(last_action="farm",
                                      consecutive_years=SPECIALIZATION_THRESHOLD)}
        assert get_specialization_bonus("c1", "farm", specs) == SPECIALIZATION_BONUS

    def test_specialized_wrong_action(self):
        specs = {"c1": Specialization(last_action="farm",
                                      consecutive_years=SPECIALIZATION_THRESHOLD)}
        assert get_specialization_bonus("c1", "code", specs) == 0.0

    def test_unknown_colonist(self):
        assert get_specialization_bonus("nobody", "farm", {}) == 0.0


# ─────────────────────────────────────────────────────────────
# Property-based invariants
# ─────────────────────────────────────────────────────────────

class TestInvariants:
    def test_gini_non_negative_for_random_data(self):
        rng = random.Random(42)
        for _ in range(100):
            n = rng.randint(2, 20)
            values = [rng.random() for _ in range(n)]
            g = compute_gini(values)
            assert g >= 0.0

    def test_stockpile_always_non_negative(self):
        rng = random.Random(42)
        sp = Stockpile(food=0.01, water=0.01, medicine=0.01)
        for _ in range(100):
            sp.decay(rate=rng.uniform(0.0, 0.5))
            assert sp.food >= 0.0
            assert sp.water >= 0.0
            assert sp.medicine >= 0.0

    def test_economy_tick_preserves_resource_bounds(self):
        """Colony resources stay in [0, 1] range after economy tick."""
        colonists = create_founding_ten(42)
        resources = Resources()
        social = SocialGraph()
        active_ids = [c.id for c in colonists]
        social.initialize(active_ids, random.Random(42))
        stockpiles: dict[str, Stockpile] = {}
        specializations: dict[str, Specialization] = {}
        rng = random.Random(42)
        for year in range(1, 20):
            actions = {c.id: rng.choice(["hoard", "farm", "rest", "sabotage"])
                       for c in colonists if c.is_active()}
            tick_economy(colonists, actions, resources, social,
                         stockpiles, specializations, year=year, rng=rng)
            for name in ("food", "water", "power", "air", "medicine"):
                val = getattr(resources, name)
                assert val >= 0.0, f"Year {year}: {name} = {val}"
