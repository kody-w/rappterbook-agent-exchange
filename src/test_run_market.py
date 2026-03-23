"""
Co-located tests for run_market.py — integration smoke tests.

Run: python -m pytest src/test_run_market.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_market import (
    build_agent_ids,
    run_terrarium,
    run_prediction_market,
    AGENT_ARCHETYPES,
)


class TestBuildAgentIds:
    def test_default_count(self) -> None:
        ids = build_agent_ids()
        assert len(ids) == len(AGENT_ARCHETYPES) * 3

    def test_custom_count(self) -> None:
        ids = build_agent_ids(agents_per_archetype=2)
        assert len(ids) == len(AGENT_ARCHETYPES) * 2

    def test_all_unique(self) -> None:
        ids = build_agent_ids()
        assert len(ids) == len(set(ids))

    def test_format(self) -> None:
        ids = build_agent_ids()
        for aid in ids:
            assert aid.startswith("zion-")
            parts = aid.split("-")
            assert len(parts) >= 3


class TestRunTerrarium:
    def test_smoke_30_sols(self) -> None:
        results = run_terrarium(sols=30, seed=42, quiet=True)
        assert "colonies" in results
        assert "summary" in results
        assert len(results["colonies"]) == 3

    def test_all_colonies_present(self) -> None:
        results = run_terrarium(sols=10, seed=42, quiet=True)
        names = {c["name"] for c in results["colonies"]}
        assert "Ares Prime" in names
        assert "Olympus Station" in names
        assert "Red Frontier" in names

    def test_deterministic(self) -> None:
        r1 = run_terrarium(sols=20, seed=42, quiet=True)
        r2 = run_terrarium(sols=20, seed=42, quiet=True)
        for i in range(3):
            assert r1["colonies"][i]["final_population"] == r2["colonies"][i]["final_population"]


class TestRunPredictionMarket:
    def test_smoke_integration(self) -> None:
        sim = run_terrarium(sols=30, seed=42, quiet=True)
        agent_ids = build_agent_ids(agents_per_archetype=2)
        full_results, engine = run_prediction_market(
            sim, agent_ids, rounds=5, seed=42,
        )
        assert "_meta" in full_results
        assert full_results["_meta"]["engine"] == "prediction-market"
        assert full_results["_meta"]["num_markets"] > 0
        assert full_results["_meta"]["num_trades"] > 0

    def test_all_markets_resolved(self) -> None:
        sim = run_terrarium(sols=50, seed=42, quiet=True)
        agent_ids = build_agent_ids()
        full_results, engine = run_prediction_market(
            sim, agent_ids, rounds=10, seed=42,
        )
        for m in engine.markets.values():
            assert m.resolved is True
            assert m.outcome is not None

    def test_leaderboard_present(self) -> None:
        sim = run_terrarium(sols=30, seed=42, quiet=True)
        agent_ids = build_agent_ids()
        full_results, engine = run_prediction_market(
            sim, agent_ids, rounds=5, seed=42,
        )
        board = engine.leaderboard()
        assert len(board) == len(agent_ids)
        # Verify sorted by PnL descending
        for i in range(len(board) - 1):
            assert board[i]["pnl"] >= board[i + 1]["pnl"]

    def test_prices_bounded(self) -> None:
        """LMSR prices must stay in (0, 1) at all times."""
        sim = run_terrarium(sols=100, seed=42, quiet=True)
        agent_ids = build_agent_ids()
        _, engine = run_prediction_market(sim, agent_ids, rounds=20, seed=42)
        for m in engine.markets.values():
            for snap in m.history:
                assert 0.0 < snap["price_yes"] < 1.0
                assert 0.0 < snap["price_no"] < 1.0
                assert abs(snap["price_yes"] + snap["price_no"] - 1.0) < 0.01

    def test_archetype_performance_keys(self) -> None:
        sim = run_terrarium(sols=30, seed=42, quiet=True)
        agent_ids = build_agent_ids()
        full_results, _ = run_prediction_market(
            sim, agent_ids, rounds=5, seed=42,
        )
        perf = full_results["archetype_performance"]
        assert len(perf) > 0
        for arch, data in perf.items():
            assert "mean_pnl" in data
            assert "best_pnl" in data
            assert "worst_pnl" in data
            assert "count" in data
