"""Tests for the Mars-100 economics organ."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.economics import (
    EconomicState,
    EconomicTickResult,
    compute_gini,
    compute_diversity,
    compute_efficiency_bonus,
    determine_policy,
    tick_economy,
    STARTING_CREDITS,
    ACTION_CREDIT_VALUES,
    POLICY_TAX_RATES,
)
from src.mars100.colony import Relationship


# ---------------------------------------------------------------------------
# Gini coefficient
# ---------------------------------------------------------------------------

class TestGini:
    def test_empty(self) -> None:
        assert compute_gini([]) == 0.0

    def test_single(self) -> None:
        assert compute_gini([5.0]) == 0.0

    def test_all_zero(self) -> None:
        assert compute_gini([0.0, 0.0, 0.0]) == 0.0

    def test_perfect_equality(self) -> None:
        assert compute_gini([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_maximum_inequality(self) -> None:
        # One person has everything, rest have zero
        gini = compute_gini([0.0, 0.0, 0.0, 100.0])
        assert 0.7 < gini <= 1.0  # finite-sample Gini < 1.0

    def test_moderate_inequality(self) -> None:
        gini = compute_gini([1.0, 2.0, 3.0, 4.0])
        assert 0.1 < gini < 0.4

    def test_always_non_negative(self) -> None:
        rng = random.Random(42)
        for _ in range(100):
            vals = [rng.random() * 10 for _ in range(rng.randint(2, 20))]
            assert compute_gini(vals) >= 0.0

    def test_always_at_most_one(self) -> None:
        rng = random.Random(99)
        for _ in range(100):
            vals = [rng.random() * 100 for _ in range(rng.randint(2, 20))]
            assert compute_gini(vals) <= 1.0


# ---------------------------------------------------------------------------
# Labor diversity
# ---------------------------------------------------------------------------

class TestDiversity:
    def test_empty_actions(self) -> None:
        assert compute_diversity({}, ["farm", "code"]) == 0.0

    def test_single_action(self) -> None:
        actions = {"a": "farm", "b": "farm", "c": "farm"}
        div = compute_diversity(actions, ["farm", "code", "rest"])
        assert div < 0.1

    def test_uniform_distribution(self) -> None:
        actions = {"a": "farm", "b": "code", "c": "rest"}
        div = compute_diversity(actions, ["farm", "code", "rest"])
        assert div > 0.9

    def test_bounded_zero_one(self) -> None:
        rng = random.Random(77)
        pool = ["farm", "code", "mediate", "terraform", "rest"]
        for _ in range(50):
            n = rng.randint(1, 15)
            acts = {str(i): rng.choice(pool) for i in range(n)}
            d = compute_diversity(acts, pool)
            assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# Efficiency bonus
# ---------------------------------------------------------------------------

class TestEfficiency:
    def test_high_diversity_low_inequality(self) -> None:
        bonus = compute_efficiency_bonus(gini=0.1, diversity=0.9)
        assert bonus > 0.05  # positive bonus

    def test_low_diversity_high_inequality(self) -> None:
        bonus = compute_efficiency_bonus(gini=0.8, diversity=0.1)
        assert bonus < 0.0  # net penalty

    def test_moderate_inequality_bonus(self) -> None:
        bonus = compute_efficiency_bonus(gini=0.25, diversity=0.5)
        assert bonus > 0.0  # moderate inequality + diversity = net positive

    def test_zero_zero(self) -> None:
        bonus = compute_efficiency_bonus(gini=0.0, diversity=0.0)
        assert bonus == 0.0

    def test_bounded(self) -> None:
        for g in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            for d in [0.0, 0.3, 0.5, 0.8, 1.0]:
                b = compute_efficiency_bonus(g, d)
                assert -0.10 < b < 0.15


# ---------------------------------------------------------------------------
# Policy determination
# ---------------------------------------------------------------------------

class TestPolicy:
    def test_council_is_communal(self) -> None:
        assert determine_policy("council") == "communal"

    def test_anarchy_is_free_market(self) -> None:
        assert determine_policy("anarchy") == "free_market"

    def test_unknown_defaults_mixed(self) -> None:
        assert determine_policy("something_new") == "mixed"


# ---------------------------------------------------------------------------
# tick_economy integration
# ---------------------------------------------------------------------------

def _make_social_get(trust: float = 0.5):
    """Return a mock social_get function."""
    def social_get(from_id: str, to_id: str) -> Relationship:
        return Relationship(trust=trust, affection=0.5, respect=0.5)
    return social_get


class TestTickEconomy:
    def test_basic_tick(self) -> None:
        eco = EconomicState()
        rng = random.Random(42)
        ids = ["c0", "c1", "c2"]
        actions = {"c0": "farm", "c1": "code", "c2": "mediate"}
        result = tick_economy(eco, year=1, actions=actions,
                              active_ids=ids, social_get=_make_social_get(),
                              gov_type="council", rng=rng)
        assert result.year == 1
        assert result.credits_awarded > 0
        assert result.policy == "communal"
        assert len(eco.gini_history) == 1
        # All colonists should have credits
        for cid in ids:
            assert eco.credits[cid] >= 0

    def test_credits_never_negative(self) -> None:
        eco = EconomicState()
        rng = random.Random(123)
        ids = [f"c{i}" for i in range(10)]
        for _ in range(50):
            actions = {cid: rng.choice(["farm", "code", "rest", "sabotage"])
                       for cid in ids}
            tick_economy(eco, year=_, actions=actions, active_ids=ids,
                         social_get=_make_social_get(), gov_type="council",
                         rng=rng)
            for cid in ids:
                assert eco.credits.get(cid, 0.0) >= 0.0, f"{cid} has negative credits"

    def test_dead_colonists_zeroed(self) -> None:
        eco = EconomicState()
        eco.credits["dead-1"] = 5.0
        rng = random.Random(42)
        ids = ["c0"]
        actions = {"c0": "farm"}
        tick_economy(eco, year=1, actions=actions, active_ids=ids,
                     social_get=_make_social_get(), gov_type="anarchy", rng=rng)
        assert eco.credits["dead-1"] == 0.0

    def test_new_colonist_gets_starting_credits(self) -> None:
        eco = EconomicState()
        rng = random.Random(42)
        ids = ["new-1"]
        actions = {"new-1": "rest"}
        tick_economy(eco, year=1, actions=actions, active_ids=ids,
                     social_get=_make_social_get(), gov_type="council", rng=rng)
        # Should have starting credits + earned (rest earns 0.2ish)
        assert eco.credits["new-1"] >= STARTING_CREDITS * 0.5

    def test_taxation_redistributes(self) -> None:
        eco = EconomicState()
        eco.credits = {"rich": 100.0, "poor": 1.0}
        rng = random.Random(42)
        ids = ["rich", "poor"]
        actions = {"rich": "farm", "poor": "farm"}
        tick_economy(eco, year=1, actions=actions, active_ids=ids,
                     social_get=_make_social_get(trust=0.0),
                     gov_type="council", rng=rng)
        # Under communal policy (40% tax), gap should narrow
        assert eco.credits["poor"] > 1.0
        ratio = eco.credits["rich"] / max(0.01, eco.credits["poor"])
        assert ratio < 100  # gap narrowed from 100:1

    def test_free_market_less_redistribution(self) -> None:
        eco = EconomicState()
        eco.credits = {"rich": 50.0, "poor": 1.0}
        rng = random.Random(42)
        ids = ["rich", "poor"]
        actions = {"rich": "farm", "poor": "farm"}
        result = tick_economy(eco, year=1, actions=actions, active_ids=ids,
                              social_get=_make_social_get(trust=0.0),
                              gov_type="anarchy", rng=rng)
        assert result.policy == "free_market"
        assert result.tax_collected < eco.credits["rich"]

    def test_high_trust_enables_trade(self) -> None:
        eco = EconomicState()
        eco.credits = {"a": 10.0, "b": 1.0}
        rng = random.Random(42)
        ids = ["a", "b"]
        actions = {"a": "farm", "b": "code"}
        result = tick_economy(eco, year=1, actions=actions, active_ids=ids,
                              social_get=_make_social_get(trust=0.9),
                              gov_type="anarchy", rng=rng)
        assert result.trades_executed >= 1

    def test_low_trust_blocks_trade(self) -> None:
        eco = EconomicState()
        eco.credits = {"a": 10.0, "b": 1.0}
        rng = random.Random(42)
        ids = ["a", "b"]
        actions = {"a": "farm", "b": "code"}
        result = tick_economy(eco, year=1, actions=actions, active_ids=ids,
                              social_get=_make_social_get(trust=0.1),
                              gov_type="anarchy", rng=rng)
        assert result.trades_executed == 0

    def test_serialization(self) -> None:
        eco = EconomicState(credits={"c0": 5.0}, total_trades=3,
                            gini_history=[0.2, 0.3])
        d = eco.to_dict()
        assert d["credits"]["c0"] == 5.0
        assert d["total_trades"] == 3
        assert len(d["gini_history"]) == 2

    def test_result_serialization(self) -> None:
        r = EconomicTickResult(year=5, trades_executed=2, gini=0.3,
                               diversity=0.7, efficiency_bonus=0.04)
        d = r.to_dict()
        assert d["year"] == 5
        assert d["gini"] == 0.3

    def test_deterministic_with_seed(self) -> None:
        """Same seed → same outcome."""
        results = []
        for _ in range(2):
            eco = EconomicState()
            rng = random.Random(777)
            ids = [f"c{i}" for i in range(5)]
            actions = {cid: ["farm", "code", "rest", "mediate", "explore"][i]
                       for i, cid in enumerate(ids)}
            r = tick_economy(eco, year=1, actions=actions, active_ids=ids,
                             social_get=_make_social_get(),
                             gov_type="council", rng=rng)
            results.append(r.to_dict())
        assert results[0] == results[1]

    def test_summary(self) -> None:
        eco = EconomicState(credits={"a": 3.0, "b": 7.0},
                            gini_history=[0.25], total_trades=4)
        s = eco.summary()
        assert s["active_accounts"] == 2
        assert s["gini"] == 0.25
        assert s["total_trades"] == 4


# ---------------------------------------------------------------------------
# Full engine integration smoke test
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_10_year_run_has_economy(self) -> None:
        """Engine produces economy data for each year."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=42, total_years=10)
        result = engine.run()
        for yr in result.years:
            d = yr.to_dict()
            assert "economy" in d, f"year {yr.year} missing economy"
            assert "gini" in d["economy"]
            assert "diversity" in d["economy"]
        final = result.to_dict()
        assert "final_economy" in final

    def test_50_year_gini_bounded(self) -> None:
        """Gini stays in [0, 1] over 50 years."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=99, total_years=50)
        result = engine.run()
        for yr in result.years:
            eco = yr.to_dict()["economy"]
            assert 0.0 <= eco["gini"] <= 1.0, f"year {yr.year}: gini={eco['gini']}"

    def test_economy_survives_deaths(self) -> None:
        """Economy continues functioning after colonist deaths."""
        from src.mars100.engine import Mars100Engine
        engine = Mars100Engine(seed=123, total_years=100)
        result = engine.run()
        # At least some years should have deaths
        years_with_deaths = sum(1 for y in result.years if y.deaths)
        # Economy should still produce results for all years
        for yr in result.years:
            assert yr.economy is not None or "economy" in yr.to_dict()
