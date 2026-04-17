"""Tests for the Mars-100 economics organ."""
from __future__ import annotations

import random
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mars100.economics import (
    EconomicAgent, EconomicState,
    compute_gini, produce, trade, tax_and_redistribute,
    sync_agents, tick_economics, economic_governance_pressure,
    DEFAULT_TAX_RATE, PRODUCTION_TABLE, TRADE_TRUST_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRelationship:
    def __init__(self, trust: float = 0.6):
        self.trust = trust


def make_social_get(trust: float = 0.6):
    """Return a callable that mimics SocialGraph.get(a, b)."""
    def _get(a: str, b: str) -> FakeRelationship:
        return FakeRelationship(trust)
    return _get


def make_state(n: int = 5, wealth: float = 1.0) -> EconomicState:
    """Create an EconomicState with n agents at equal wealth."""
    agents = {f"c-{i}": EconomicAgent(f"c-{i}", wealth=wealth) for i in range(n)}
    return EconomicState(agents=agents)


def make_ids(n: int = 5) -> list[str]:
    return [f"c-{i}" for i in range(n)]


def make_skills(cid: str) -> dict[str, float]:
    return {"terraforming": 0.5, "hydroponics": 0.5, "mediation": 0.5,
            "coding": 0.5, "prayer": 0.5, "sabotage": 0.2}


def make_stats(cid: str) -> dict[str, float]:
    return {"resolve": 0.5, "improvisation": 0.5, "empathy": 0.5,
            "hoarding": 0.5, "faith": 0.5, "paranoia": 0.3}


# ===========================================================================
# compute_gini tests
# ===========================================================================

class TestComputeGini:
    def test_empty_list(self):
        assert compute_gini([]) == 0.0

    def test_single_value(self):
        assert compute_gini([100.0]) == 0.0

    def test_all_zeros(self):
        assert compute_gini([0.0, 0.0, 0.0]) == 0.0

    def test_perfect_equality(self):
        assert compute_gini([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_extreme_inequality(self):
        """One person has everything."""
        gini = compute_gini([0.0, 0.0, 0.0, 100.0])
        assert 0.7 < gini <= 1.0

    def test_moderate_inequality(self):
        gini = compute_gini([1.0, 2.0, 3.0, 4.0, 5.0])
        assert 0.1 < gini < 0.5

    def test_two_agents_unequal(self):
        gini = compute_gini([0.0, 1.0])
        assert 0.4 < gini <= 0.6  # should be 0.5 for [0, 1]

    def test_result_bounds(self):
        """Gini should always be in [0, 1)."""
        rng = random.Random(42)
        for _ in range(50):
            values = [rng.random() * 10 for _ in range(rng.randint(2, 20))]
            gini = compute_gini(values)
            assert 0.0 <= gini < 1.0, f"Gini {gini} out of bounds for {values}"

    def test_negative_wealth_handled(self):
        """Negative wealth shouldn't crash."""
        # compute_gini handles total <= 0 case
        gini = compute_gini([-1.0, -2.0])
        assert gini == 0.0

    def test_deterministic(self):
        values = [0.5, 1.2, 3.4, 0.1, 2.0]
        assert compute_gini(values) == compute_gini(values)


# ===========================================================================
# produce tests
# ===========================================================================

class TestProduce:
    def test_all_actions_produce_positive(self):
        rng = random.Random(42)
        skills = make_skills("c-0")
        stats = make_stats("c-0")
        for action in PRODUCTION_TABLE:
            output = produce(action, skills, stats, 0.6, rng)
            assert output >= 0.0, f"Action {action} produced negative: {output}"

    def test_rest_produces_least(self):
        rng = random.Random(42)
        skills = make_skills("c-0")
        stats = make_stats("c-0")
        rest_outputs = [produce("rest", skills, stats, 0.6, random.Random(i))
                        for i in range(20)]
        farm_outputs = [produce("farm", skills, stats, 0.6, random.Random(i))
                        for i in range(20)]
        assert sum(rest_outputs) / 20 < sum(farm_outputs) / 20

    def test_higher_skill_produces_more(self):
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        low_skills = {"hydroponics": 0.1}
        high_skills = {"hydroponics": 0.9}
        stats = make_stats("c-0")
        low = produce("farm", low_skills, stats, 0.6, rng_a)
        high = produce("farm", high_skills, stats, 0.6, rng_b)
        assert high > low

    def test_scarcity_depresses_output(self):
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        skills = make_skills("c-0")
        stats = make_stats("c-0")
        rich = produce("farm", skills, stats, 0.8, rng_a)
        poor = produce("farm", skills, stats, 0.1, rng_b)
        assert rich > poor

    def test_output_non_negative_always(self):
        """Property: produce never returns negative."""
        for seed in range(100):
            rng = random.Random(seed)
            output = produce("sabotage", {"sabotage": 0.1}, {"paranoia": 0.9}, 0.1, rng)
            assert output >= 0.0

    def test_unknown_action_returns_small(self):
        rng = random.Random(42)
        output = produce("dance", {}, {}, 0.5, rng)
        assert 0.0 <= output < 0.05


# ===========================================================================
# trade tests
# ===========================================================================

class TestTrade:
    def test_no_trades_with_low_trust(self):
        state = make_state(3, wealth=1.0)
        social = make_social_get(trust=0.2)
        trades = trade(state, social, make_ids(3), random.Random(42))
        assert len(trades) == 0

    def test_trades_occur_with_high_trust(self):
        state = make_state(5, wealth=1.0)
        social = make_social_get(trust=0.8)
        trades = trade(state, social, make_ids(5), random.Random(42))
        assert len(trades) > 0

    def test_trade_conserves_wealth(self):
        state = make_state(5, wealth=2.0)
        total_before = sum(a.wealth for a in state.agents.values())
        social = make_social_get(trust=0.8)
        trade(state, social, make_ids(5), random.Random(42))
        total_after = sum(a.wealth for a in state.agents.values())
        assert abs(total_before - total_after) < 1e-10

    def test_empty_ids_no_crash(self):
        state = EconomicState()
        trades = trade(state, make_social_get(), [], random.Random(42))
        assert trades == []

    def test_single_agent_no_trades(self):
        state = make_state(1, wealth=5.0)
        trades = trade(state, make_social_get(), ["c-0"], random.Random(42))
        assert len(trades) == 0

    def test_zero_wealth_no_trade(self):
        state = make_state(3, wealth=0.0)
        trades = trade(state, make_social_get(0.9), make_ids(3), random.Random(42))
        assert len(trades) == 0

    def test_wealth_transfer_direction(self):
        """Wealthier gives to poorer."""
        state = EconomicState(agents={
            "rich": EconomicAgent("rich", wealth=10.0),
            "poor": EconomicAgent("poor", wealth=0.5),
        })
        social = make_social_get(trust=0.9)
        trades = trade(state, social, ["rich", "poor"], random.Random(42))
        if trades:
            assert trades[0]["from"] == "rich"
            assert trades[0]["to"] == "poor"


# ===========================================================================
# tax_and_redistribute tests
# ===========================================================================

class TestTaxAndRedistribute:
    def test_conservation_mixed(self):
        state = make_state(5, wealth=2.0)
        state.policy = "mixed"
        total_before = sum(a.wealth for a in state.agents.values()) + state.treasury
        tax_and_redistribute(state)
        total_after = sum(a.wealth for a in state.agents.values()) + state.treasury
        assert abs(total_before - total_after) < 1e-10

    def test_conservation_collectivist(self):
        state = make_state(5, wealth=2.0)
        state.policy = "collectivist"
        total_before = sum(a.wealth for a in state.agents.values()) + state.treasury
        tax_and_redistribute(state)
        total_after = sum(a.wealth for a in state.agents.values()) + state.treasury
        assert abs(total_before - total_after) < 1e-10

    def test_conservation_free_market(self):
        state = make_state(5, wealth=2.0)
        state.policy = "free_market"
        total_before = sum(a.wealth for a in state.agents.values()) + state.treasury
        tax_and_redistribute(state)
        total_after = sum(a.wealth for a in state.agents.values()) + state.treasury
        assert abs(total_before - total_after) < 1e-10

    def test_collectivist_empties_treasury(self):
        state = make_state(3, wealth=1.0)
        state.policy = "collectivist"
        tax_and_redistribute(state)
        assert abs(state.treasury) < 1e-10

    def test_free_market_accumulates_treasury(self):
        state = make_state(3, wealth=1.0)
        state.policy = "free_market"
        tax_and_redistribute(state)
        assert state.treasury > 0

    def test_zero_wealth_no_tax(self):
        state = make_state(3, wealth=0.0)
        result = tax_and_redistribute(state)
        assert result["total_tax"] == 0.0

    def test_tax_rate_applied(self):
        state = make_state(1, wealth=10.0)
        state.tax_rate = 0.20
        state.policy = "free_market"
        tax_and_redistribute(state)
        agent = state.agents["c-0"]
        assert abs(agent.tax_paid - 2.0) < 1e-10
        assert abs(agent.wealth - 8.0) < 1e-10


# ===========================================================================
# sync_agents tests
# ===========================================================================

class TestSyncAgents:
    def test_adds_new_colonists(self):
        state = make_state(3)
        sync_agents(state, make_ids(5))
        assert len(state.agents) == 5
        assert "c-3" in state.agents
        assert "c-4" in state.agents

    def test_removes_dead_colonists(self):
        state = make_state(5, wealth=2.0)
        sync_agents(state, ["c-0", "c-1", "c-2"])
        assert len(state.agents) == 3
        assert "c-3" not in state.agents

    def test_dead_wealth_goes_to_treasury(self):
        state = make_state(5, wealth=2.0)
        treasury_before = state.treasury
        sync_agents(state, ["c-0", "c-1", "c-2"])
        # 2 agents removed, each with 2.0 wealth
        assert abs(state.treasury - (treasury_before + 4.0)) < 1e-10

    def test_empty_active_removes_all(self):
        state = make_state(3, wealth=1.0)
        sync_agents(state, [])
        assert len(state.agents) == 0
        assert state.treasury == 3.0


# ===========================================================================
# tick_economics integration tests
# ===========================================================================

class TestTickEconomics:
    def _make_tick_args(self, n: int = 5, seed: int = 42):
        state = make_state(n)
        ids = make_ids(n)
        actions = {cid: "farm" for cid in ids}
        skills_map = {cid: make_skills(cid) for cid in ids}
        stats_map = {cid: make_stats(cid) for cid in ids}
        return state, ids, actions, skills_map, stats_map

    def test_basic_tick(self):
        state, ids, actions, skills, stats = self._make_tick_args()
        result = tick_economics(state, ids, actions, skills, stats,
                                0.6, make_social_get(), random.Random(42))
        assert result["gdp"] > 0
        assert 0.0 <= result["gini"] < 1.0
        assert result["treasury"] >= 0

    def test_gdp_is_production_only(self):
        """GDP should equal sum of labor outputs, not include trades."""
        state, ids, actions, skills, stats = self._make_tick_args()
        tick_economics(state, ids, actions, skills, stats,
                       0.6, make_social_get(0.9), random.Random(42))
        total_labor = sum(a.labor_output for a in state.agents.values())
        assert abs(state.gdp - total_labor) < 1e-10

    def test_wealth_non_negative(self):
        """No agent should end up with negative wealth."""
        state, ids, actions, skills, stats = self._make_tick_args(10)
        for _ in range(20):
            tick_economics(state, ids, actions, skills, stats,
                           0.6, make_social_get(), random.Random(_))
        for agent in state.agents.values():
            assert agent.wealth >= 0.0, f"Agent {agent.colonist_id} has negative wealth"

    def test_history_recorded(self):
        state, ids, actions, skills, stats = self._make_tick_args()
        tick_economics(state, ids, actions, skills, stats,
                       0.6, make_social_get(), random.Random(42))
        assert len(state.year_history) == 1
        assert "gdp" in state.year_history[0]

    def test_history_capped_at_20(self):
        state, ids, actions, skills, stats = self._make_tick_args()
        for i in range(25):
            tick_economics(state, ids, actions, skills, stats,
                           0.6, make_social_get(), random.Random(i))
        assert len(state.year_history) == 20

    def test_deterministic(self):
        """Same seed + inputs → same result."""
        def run_tick(seed):
            state, ids, actions, skills, stats = self._make_tick_args()
            return tick_economics(state, ids, actions, skills, stats,
                                  0.6, make_social_get(), random.Random(seed))
        a = run_tick(42)
        b = run_tick(42)
        assert a["gdp"] == b["gdp"]
        assert a["gini"] == b["gini"]

    def test_new_colonist_mid_sim(self):
        """Adding a new colonist mid-simulation works."""
        state, ids, actions, skills, stats = self._make_tick_args(5)
        tick_economics(state, ids, actions, skills, stats,
                       0.6, make_social_get(), random.Random(42))
        # Add a new colonist
        ids.append("c-5")
        actions["c-5"] = "code"
        skills["c-5"] = make_skills("c-5")
        stats["c-5"] = make_stats("c-5")
        tick_economics(state, ids, actions, skills, stats,
                       0.6, make_social_get(), random.Random(43))
        assert "c-5" in state.agents

    def test_colonist_death_mid_sim(self):
        """Removing a colonist mid-simulation works."""
        state, ids, actions, skills, stats = self._make_tick_args(5)
        tick_economics(state, ids, actions, skills, stats,
                       0.6, make_social_get(), random.Random(42))
        # Remove colonist c-4
        ids.remove("c-4")
        del actions["c-4"]
        tick_economics(state, ids, actions, skills, stats,
                       0.6, make_social_get(), random.Random(43))
        assert "c-4" not in state.agents
        assert state.treasury > 0  # dead colonist's wealth went to treasury


# ===========================================================================
# economic_governance_pressure tests
# ===========================================================================

class TestGovernancePressure:
    def test_low_gini_no_pressure(self):
        state = EconomicState(gini=0.1)
        assert economic_governance_pressure(state) == 0.0

    def test_high_gini_adds_pressure(self):
        state = EconomicState(gini=0.5)
        pressure = economic_governance_pressure(state)
        assert pressure > 0.0

    def test_pressure_capped(self):
        state = EconomicState(gini=1.0)
        pressure = economic_governance_pressure(state)
        assert pressure <= 0.3

    def test_monotonic(self):
        """Higher Gini → higher or equal pressure."""
        values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        pressures = [economic_governance_pressure(EconomicState(gini=g))
                     for g in values]
        for i in range(1, len(pressures)):
            assert pressures[i] >= pressures[i - 1]


# ===========================================================================
# serialization tests
# ===========================================================================

class TestSerialization:
    def test_agent_to_dict(self):
        agent = EconomicAgent("c-0", wealth=1.5, labor_output=0.3)
        d = agent.to_dict()
        assert d["colonist_id"] == "c-0"
        assert d["wealth"] == 1.5

    def test_state_to_dict(self):
        state = make_state(3, wealth=1.0)
        state.gdp = 0.5
        state.gini = 0.2
        d = state.to_dict()
        assert d["agent_count"] == 3
        assert d["gdp"] == 0.5
        assert d["total_wealth"] == 3.0

    def test_summary(self):
        state = make_state(2, wealth=2.0)
        state.gdp = 1.0
        state.gini = 0.3
        s = state.summary()
        assert "gdp" in s
        assert "gini" in s
        assert "total_wealth" in s


# ===========================================================================
# Property-based invariants
# ===========================================================================

class TestInvariants:
    def test_gini_bounds_random(self):
        """Gini stays in [0, 1) for random wealth distributions."""
        rng = random.Random(123)
        for _ in range(100):
            n = rng.randint(2, 30)
            values = [rng.random() * rng.random() * 10 for _ in range(n)]
            gini = compute_gini(values)
            assert 0.0 <= gini < 1.0

    def test_trade_conservation_random(self):
        """Trade always conserves total wealth."""
        for seed in range(20):
            rng = random.Random(seed)
            n = rng.randint(2, 10)
            state = EconomicState(agents={
                f"a-{i}": EconomicAgent(f"a-{i}", wealth=rng.random() * 5)
                for i in range(n)
            })
            total_before = sum(a.wealth for a in state.agents.values())
            ids = list(state.agents.keys())
            trade(state, make_social_get(0.8), ids, rng)
            total_after = sum(a.wealth for a in state.agents.values())
            assert abs(total_before - total_after) < 1e-10, \
                f"Seed {seed}: {total_before} != {total_after}"

    def test_tax_conservation_random(self):
        """Tax + redistribution always conserves total credits."""
        for seed in range(20):
            rng = random.Random(seed)
            n = rng.randint(1, 10)
            state = EconomicState(agents={
                f"a-{i}": EconomicAgent(f"a-{i}", wealth=rng.random() * 5)
                for i in range(n)
            })
            state.policy = rng.choice(["free_market", "mixed", "collectivist"])
            state.tax_rate = rng.uniform(0.05, 0.50)
            total_before = sum(a.wealth for a in state.agents.values()) + state.treasury
            tax_and_redistribute(state)
            total_after = sum(a.wealth for a in state.agents.values()) + state.treasury
            assert abs(total_before - total_after) < 1e-10, \
                f"Seed {seed}: {total_before} != {total_after}"

    def test_full_tick_10_years_no_crash(self):
        """Run 10 years of economic ticks without crashing."""
        state = EconomicState()
        ids = [f"c-{i}" for i in range(8)]
        for year in range(10):
            actions = {cid: random.choice(["farm", "code", "terraform", "rest"])
                       for cid in ids}
            skills = {cid: make_skills(cid) for cid in ids}
            stats = {cid: make_stats(cid) for cid in ids}
            result = tick_economics(state, ids, actions, skills, stats,
                                    0.6, make_social_get(), random.Random(year))
            assert result["gdp"] >= 0
            assert 0.0 <= result["gini"] < 1.0
