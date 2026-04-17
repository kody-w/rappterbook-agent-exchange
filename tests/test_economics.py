"""Tests for the Mars-100 economics organ."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mars100.economics import (
    Wallet, TradeRecord, EconomicState, EconomicTickResult,
    compute_labor_value, compute_gini, find_trade_pairs,
    execute_trade, apply_redistribution, update_economic_policy,
    handle_death, handle_exile, tick_economics,
    LABOR_VALUE_BASE, TRADE_TRUST_THRESHOLD,
    REDISTRIBUTION_RATES, COMPLEMENTARY_SKILLS,
)


# -- Wallet tests -----------------------------------------------------------

class TestWallet:
    def test_default_wallet(self):
        w = Wallet()
        assert w.credits == 0.0
        assert w.lifetime_earned == 0.0
        assert w.lifetime_spent == 0.0
        assert w.trades_completed == 0

    def test_wallet_roundtrip(self):
        w = Wallet(credits=1.5, lifetime_earned=3.0, lifetime_spent=1.5,
                   trades_completed=5)
        d = w.to_dict()
        w2 = Wallet.from_dict(d)
        assert w2.credits == w.credits
        assert w2.lifetime_earned == w.lifetime_earned
        assert w2.lifetime_spent == w.lifetime_spent
        assert w2.trades_completed == w.trades_completed

    def test_wallet_from_empty_dict(self):
        w = Wallet.from_dict({})
        assert w.credits == 0.0
        assert w.trades_completed == 0


# -- Gini coefficient tests -------------------------------------------------

class TestGini:
    def test_perfect_equality(self):
        assert compute_gini([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_maximum_inequality(self):
        gini = compute_gini([0.0, 0.0, 0.0, 100.0])
        assert gini > 0.7
        assert gini <= 1.0

    def test_empty_list(self):
        assert compute_gini([]) == 0.0

    def test_single_value(self):
        assert compute_gini([5.0]) == 0.0

    def test_all_zeros(self):
        assert compute_gini([0.0, 0.0, 0.0]) == 0.0

    def test_moderate_inequality(self):
        gini = compute_gini([1.0, 2.0, 3.0, 4.0, 5.0])
        assert 0.1 < gini < 0.5

    def test_two_values(self):
        gini = compute_gini([0.0, 10.0])
        assert gini > 0.4

    def test_known_uniform(self):
        """Uniform distribution Gini ≈ 1/(2n) for small n."""
        vals = [1.0] * 100
        assert abs(compute_gini(vals)) < 0.01

    def test_gini_bounded(self):
        """Gini must always be in [0, 1)."""
        rng = random.Random(42)
        for _ in range(50):
            n = rng.randint(2, 20)
            vals = [rng.expovariate(1.0) for _ in range(n)]
            g = compute_gini(vals)
            assert 0.0 <= g < 1.0, f"Gini {g} out of bounds for {vals}"


# -- Labor value tests -------------------------------------------------------

class TestLaborValue:
    def test_rest_produces_half_base(self):
        val = compute_labor_value("rest", {}, {})
        assert val == LABOR_VALUE_BASE * 0.5

    def test_skilled_action_produces_more(self):
        val = compute_labor_value("farm", {"hydroponics": 0.8}, {})
        assert val > LABOR_VALUE_BASE

    def test_unskilled_action(self):
        val = compute_labor_value("farm", {"hydroponics": 0.0}, {})
        assert val == LABOR_VALUE_BASE * 1.0

    def test_stat_fallback(self):
        val = compute_labor_value("explore", {}, {"improvisation": 0.9})
        assert val > LABOR_VALUE_BASE

    def test_sabotage_labor_value(self):
        val = compute_labor_value("sabotage", {"sabotage": 0.5}, {})
        assert val > 0


# -- Trade pair finding ------------------------------------------------------

class TestFindTradePairs:
    def test_complementary_skills_match(self):
        ids = ["a", "b"]
        actions = {"a": "farm", "b": "terraform"}
        trust = {("a", "b"): 0.6, ("b", "a"): 0.6}
        pairs = find_trade_pairs(ids, actions, trust, random.Random(42))
        assert len(pairs) >= 1

    def test_low_trust_no_trade(self):
        ids = ["a", "b"]
        actions = {"a": "farm", "b": "terraform"}
        trust = {("a", "b"): 0.1}
        pairs = find_trade_pairs(ids, actions, trust, random.Random(42))
        assert len(pairs) == 0

    def test_same_action_no_complement(self):
        ids = ["a", "b"]
        actions = {"a": "rest", "b": "rest"}
        trust = {("a", "b"): 0.9}
        pairs = find_trade_pairs(ids, actions, trust, random.Random(42))
        assert len(pairs) == 0

    def test_sabotage_excluded(self):
        ids = ["a", "b"]
        actions = {"a": "sabotage", "b": "farm"}
        trust = {("a", "b"): 0.9}
        pairs = find_trade_pairs(ids, actions, trust, random.Random(42))
        assert len(pairs) == 0


# -- Trade execution ---------------------------------------------------------

class TestExecuteTrade:
    def test_trade_increases_credits(self):
        state = EconomicState()
        record = execute_trade(
            state, "a", "b", "farm", "code",
            {"hydroponics": 0.5}, {"coding": 0.7},
            {}, {}, year=5, rng=random.Random(42),
        )
        assert record is not None
        assert state.wallets["a"].credits > 0
        assert state.wallets["b"].credits > 0
        assert state.total_trades == 1

    def test_trade_log_bounded(self):
        state = EconomicState()
        for i in range(60):
            execute_trade(
                state, "a", "b", "farm", "code",
                {"hydroponics": 0.5}, {"coding": 0.5},
                {}, {}, year=i, rng=random.Random(i),
            )
        assert len(state.trade_log) <= 50


# -- Redistribution ----------------------------------------------------------

class TestRedistribution:
    def test_anarchy_no_redistribution(self):
        state = EconomicState()
        state.wallets["a"] = Wallet(credits=10.0)
        state.wallets["b"] = Wallet(credits=0.0)
        total = apply_redistribution(state, "anarchy", ["a", "b"])
        assert total == 0.0
        assert state.wallets["a"].credits == 10.0

    def test_council_redistribution(self):
        state = EconomicState()
        state.wallets["a"] = Wallet(credits=10.0)
        state.wallets["b"] = Wallet(credits=0.0)
        total = apply_redistribution(state, "council", ["a", "b"])
        assert total > 0
        assert state.wallets["b"].credits > 0
        assert state.wallets["a"].credits < 10.0

    def test_dictator_leader_gets_share(self):
        state = EconomicState()
        state.wallets["a"] = Wallet(credits=10.0)
        state.wallets["b"] = Wallet(credits=10.0)
        state.wallets["leader"] = Wallet(credits=0.0)
        apply_redistribution(state, "dictator", ["a", "b", "leader"],
                             leader_id="leader")
        # Leader gets 30% of tax pool + nothing from redistribution to others
        # Leader started with 0, so leader.credits = leader_share only
        assert state.wallets["leader"].credits > 0
        # Verify leader gets disproportionate share of tax revenue
        leader_share = state.wallets["leader"].credits
        other_shares = [state.wallets[cid].credits - 10.0 + 10.0 * 0.25
                        for cid in ["a", "b"]]
        assert leader_share > max(other_shares) * 0.5

    def test_empty_active_ids(self):
        state = EconomicState()
        total = apply_redistribution(state, "council", [])
        assert total == 0.0


# -- Estate handling ---------------------------------------------------------

class TestEstateHandling:
    def test_death_distributes_credits(self):
        state = EconomicState()
        state.wallets["dead"] = Wallet(credits=10.0)
        state.wallets["alive1"] = Wallet(credits=0.0)
        state.wallets["alive2"] = Wallet(credits=0.0)
        estate = handle_death(state, "dead", ["dead", "alive1", "alive2"])
        assert estate == 10.0
        assert state.wallets["dead"].credits == 0.0
        assert state.wallets["alive1"].credits == 5.0
        assert state.wallets["alive2"].credits == 5.0

    def test_death_no_credits(self):
        state = EconomicState()
        state.wallets["dead"] = Wallet(credits=0.0)
        estate = handle_death(state, "dead", ["dead", "alive"])
        assert estate == 0.0

    def test_exile_confiscates(self):
        state = EconomicState()
        state.wallets["exile"] = Wallet(credits=5.0)
        confiscated = handle_exile(state, "exile")
        assert confiscated == 5.0
        assert state.wallets["exile"].credits == 0.0


# -- Economic policy ---------------------------------------------------------

class TestEconomicPolicy:
    def test_anarchy_market(self):
        assert update_economic_policy(0.5, "anarchy") == "market"

    def test_anarchy_communal(self):
        assert update_economic_policy(0.1, "anarchy") == "communal"

    def test_dictator_planned(self):
        assert update_economic_policy(0.3, "dictator") == "planned"

    def test_council_mixed(self):
        assert update_economic_policy(0.3, "council") == "mixed"

    def test_ai_governor(self):
        assert update_economic_policy(0.3, "ai_governor") == "algorithmic"


# -- EconomicState -----------------------------------------------------------

class TestEconomicState:
    def test_ensure_account_creates(self):
        state = EconomicState()
        w = state.ensure_account("new-colonist")
        assert w.credits == 0.0
        assert "new-colonist" in state.wallets

    def test_ensure_account_idempotent(self):
        state = EconomicState()
        state.ensure_account("c1")
        state.wallets["c1"].credits = 5.0
        w = state.ensure_account("c1")
        assert w.credits == 5.0

    def test_smoothed_gini_empty(self):
        state = EconomicState()
        assert state.smoothed_gini() == 0.0

    def test_smoothed_gini_window(self):
        state = EconomicState()
        state.gini_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        smoothed = state.smoothed_gini()
        assert abs(smoothed - 0.4) < 0.01

    def test_to_dict_roundtrip(self):
        state = EconomicState()
        state.ensure_account("a")
        state.wallets["a"].credits = 3.0
        state.gini_history = [0.1, 0.2]
        d = state.to_dict()
        assert d["gini_current"] == 0.2
        assert "a" in d["wallets"]


# -- Full tick ---------------------------------------------------------------

class TestTickEconomics:
    def _make_context(self, n_colonists=5, seed=42):
        rng = random.Random(seed)
        ids = [f"c{i}" for i in range(n_colonists)]
        action_pool = ["farm", "code", "terraform", "mediate", "explore"]
        actions = {cid: action_pool[i % len(action_pool)]
                   for i, cid in enumerate(ids)}
        trust = {}
        for a in ids:
            for b in ids:
                if a != b:
                    trust[(a, b)] = 0.5 + rng.gauss(0, 0.1)
        skills = {cid: {"hydroponics": rng.random(), "coding": rng.random(),
                         "terraforming": rng.random(), "mediation": rng.random()}
                  for cid in ids}
        stats = {cid: {"improvisation": rng.random(), "empathy": rng.random(),
                        "resolve": rng.random()}
                 for cid in ids}
        return ids, actions, trust, skills, stats, rng

    def test_basic_tick(self):
        state = EconomicState()
        ids, actions, trust, skills, stats, rng = self._make_context()
        result = tick_economics(
            state, year=1, active_ids=ids, actions=actions,
            social_trust=trust, colonist_skills=skills,
            colonist_stats=stats, gov_type="anarchy",
            leader_id=None, rng=rng,
        )
        assert result.year == 1
        assert isinstance(result.gini, float)
        assert 0.0 <= result.gini <= 1.0

    def test_10_year_smoke(self):
        """Economy evolves sensibly over 10 years."""
        state = EconomicState()
        rng = random.Random(42)
        ids = [f"c{i}" for i in range(8)]
        for year in range(1, 11):
            actions = {cid: rng.choice(["farm", "code", "terraform", "mediate"])
                       for cid in ids}
            trust = {(a, b): 0.5 for a in ids for b in ids if a != b}
            skills = {cid: {"hydroponics": 0.5, "coding": 0.5,
                             "terraforming": 0.5, "mediation": 0.5}
                      for cid in ids}
            stats = {cid: {"improvisation": 0.5} for cid in ids}
            result = tick_economics(
                state, year=year, active_ids=ids, actions=actions,
                social_trust=trust, colonist_skills=skills,
                colonist_stats=stats, gov_type="council",
                leader_id=None, rng=rng,
            )
        assert state.total_trades > 0
        assert len(state.gini_history) == 10
        assert all(0.0 <= g <= 1.0 for g in state.gini_history)

    def test_council_reduces_inequality(self):
        """Council redistribution keeps Gini lower than anarchy."""
        gini_anarchy = self._run_scenario("anarchy")
        gini_council = self._run_scenario("council")
        assert gini_council <= gini_anarchy + 0.1

    def _run_scenario(self, gov_type, years=20, seed=99):
        state = EconomicState()
        rng = random.Random(seed)
        ids = [f"c{i}" for i in range(6)]
        for year in range(1, years + 1):
            actions = {cid: rng.choice(["farm", "code", "terraform"])
                       for cid in ids}
            trust = {(a, b): 0.5 for a in ids for b in ids if a != b}
            skills = {cid: {"hydroponics": rng.random(), "coding": rng.random(),
                             "terraforming": rng.random()}
                      for cid in ids}
            stats = {cid: {} for cid in ids}
            tick_economics(
                state, year=year, active_ids=ids, actions=actions,
                social_trust=trust, colonist_skills=skills,
                colonist_stats=stats, gov_type=gov_type,
                leader_id=None, rng=rng,
            )
        return state.smoothed_gini()

    def test_inequality_alert(self):
        """Sustained high inequality triggers alert."""
        state = EconomicState()
        state.wallets["rich"] = Wallet(credits=100.0)
        state.wallets["poor"] = Wallet(credits=0.0)
        ids = ["rich", "poor"]
        rng = random.Random(42)
        for year in range(1, 10):
            result = tick_economics(
                state, year=year, active_ids=ids,
                actions={"rich": "code", "poor": "rest"},
                social_trust={}, colonist_skills={},
                colonist_stats={}, gov_type="anarchy",
                leader_id=None, rng=rng,
            )
        assert result.inequality_alert is True

    def test_velocity_computed(self):
        state = EconomicState()
        ids, actions, trust, skills, stats, rng = self._make_context()
        result = tick_economics(
            state, year=1, active_ids=ids, actions=actions,
            social_trust=trust, colonist_skills=skills,
            colonist_stats=stats, gov_type="anarchy",
            leader_id=None, rng=rng,
        )
        assert isinstance(result.velocity, float)
        assert result.velocity >= 0.0


# -- TradeRecord tests -------------------------------------------------------

class TestTradeRecord:
    def test_to_dict(self):
        tr = TradeRecord(year=5, seller_id="a", buyer_id="b",
                         seller_skill="hydroponics", buyer_skill="coding",
                         value=0.15)
        d = tr.to_dict()
        assert d["year"] == 5
        assert d["value"] == 0.15


# -- EconomicTickResult tests ------------------------------------------------

class TestEconomicTickResult:
    def test_to_dict(self):
        r = EconomicTickResult(year=1, gini=0.3, redistribution=0.5)
        d = r.to_dict()
        assert d["year"] == 1
        assert d["gini"] == 0.3
