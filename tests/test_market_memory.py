"""
Tests for market_memory.py — persistent calibration memory across frames.

Run: python -m pytest tests/test_market_memory.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.market_memory import AgentRecord, MarketMemory
from src.market_maker import Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prediction(
    agent: str = "agent-1",
    archetype: str = "oracle",
    category: str = "survival",
    confidence: float = 0.8,
    outcome: bool = True,
    brier: float = 0.04,
) -> Prediction:
    """Create a test Prediction."""
    return Prediction(
        id=f"p-{agent}-{category}",
        agent=agent,
        archetype=archetype,
        category=category,
        description="test",
        params={},
        confidence=confidence,
        stake=10.0,
        outcome=outcome,
        brier=brier,
    )


# ---------------------------------------------------------------------------
# AgentRecord
# ---------------------------------------------------------------------------

class TestAgentRecord:

    def test_mean_brier_empty(self) -> None:
        rec = AgentRecord(agent="a", archetype="oracle")
        assert rec.mean_brier == 0.5

    def test_mean_brier_with_data(self) -> None:
        rec = AgentRecord(agent="a", archetype="oracle", brier_history=[0.1, 0.2, 0.3])
        assert rec.mean_brier == pytest.approx(0.2, abs=0.01)

    def test_calibration_shift_zero_for_good_agent(self) -> None:
        rec = AgentRecord(agent="a", archetype="analyst", brier_history=[0.05, 0.10, 0.08])
        assert rec.calibration_shift == 0.0

    def test_calibration_shift_positive_for_bad_agent(self) -> None:
        rec = AgentRecord(agent="a", archetype="degen", brier_history=[0.6, 0.7, 0.8])
        assert rec.calibration_shift > 0.2

    def test_calibration_shift_bounded(self) -> None:
        rec = AgentRecord(agent="a", archetype="degen", brier_history=[1.0] * 20)
        assert 0.0 <= rec.calibration_shift <= 0.3

    def test_category_skill_default(self) -> None:
        rec = AgentRecord(agent="a", archetype="oracle")
        assert rec.category_skill("survival") == 1.0

    def test_category_skill_specialist(self) -> None:
        rec = AgentRecord(
            agent="a", archetype="oracle",
            brier_history=[0.3, 0.3, 0.3],
            category_briers={"survival": [0.1, 0.1, 0.1]},
        )
        skill = rec.category_skill("survival")
        assert skill < 1.0  # better than average in this category

    def test_to_dict_serializable(self) -> None:
        rec = AgentRecord(agent="a", archetype="oracle", brier_history=[0.1, 0.2])
        d = rec.to_dict()
        json.dumps(d)  # must not raise
        assert d["agent"] == "a"
        assert d["archetype"] == "oracle"


# ---------------------------------------------------------------------------
# MarketMemory
# ---------------------------------------------------------------------------

class TestMarketMemory:

    def test_fresh_memory(self) -> None:
        mem = MarketMemory()
        assert mem.generation == 0
        assert len(mem.agents) == 0

    def test_record_run_increments_generation(self) -> None:
        mem = MarketMemory()
        preds = [_make_prediction()]
        mem.record_run(preds, 0.2, 0.7)
        assert mem.generation == 1

    def test_record_run_tracks_agents(self) -> None:
        mem = MarketMemory()
        preds = [
            _make_prediction(agent="alice", brier=0.1),
            _make_prediction(agent="bob", brier=0.3),
        ]
        mem.record_run(preds, 0.2, 0.7)
        assert "alice" in mem.agents
        assert "bob" in mem.agents
        assert mem.agents["alice"].total_resolved == 1

    def test_record_run_skips_unresolved(self) -> None:
        mem = MarketMemory()
        unresolved = _make_prediction()
        unresolved.outcome = None
        unresolved.brier = None
        mem.record_run([unresolved], 0.2, 0.7)
        assert len(mem.agents) == 0

    def test_adjust_confidence_no_history(self) -> None:
        mem = MarketMemory()
        raw = 0.8
        adjusted = mem.adjust_confidence("unknown", "survival", raw)
        assert adjusted == raw

    def test_adjust_confidence_shrinks_overconfident(self) -> None:
        mem = MarketMemory()
        mem.agents["degen-1"] = AgentRecord(
            agent="degen-1", archetype="degen",
            brier_history=[0.7, 0.8, 0.6, 0.7, 0.8],
        )
        raw = 0.95
        adjusted = mem.adjust_confidence("degen-1", "survival", raw)
        assert adjusted < raw  # should be pulled toward 0.5

    def test_adjust_confidence_bounded(self) -> None:
        mem = MarketMemory()
        mem.agents["agent-1"] = AgentRecord(
            agent="agent-1", archetype="oracle",
            brier_history=[0.5] * 20,
        )
        for raw in [0.01, 0.5, 0.99]:
            adjusted = mem.adjust_confidence("agent-1", "survival", raw)
            assert 0.01 <= adjusted <= 0.99

    def test_improvement_trend_none_for_single_run(self) -> None:
        mem = MarketMemory()
        mem.record_run([_make_prediction()], 0.3, 0.6)
        assert mem.improvement_trend is None

    def test_improvement_trend_negative_means_improving(self) -> None:
        mem = MarketMemory()
        mem.market_brier_history = [0.5, 0.4, 0.3, 0.2]
        assert mem.improvement_trend is not None
        assert mem.improvement_trend < 0

    def test_improvement_trend_positive_means_degrading(self) -> None:
        mem = MarketMemory()
        mem.market_brier_history = [0.2, 0.3, 0.4, 0.5]
        assert mem.improvement_trend is not None
        assert mem.improvement_trend > 0

    def test_to_dict_serializable(self) -> None:
        mem = MarketMemory()
        preds = [_make_prediction(agent="alice")]
        mem.record_run(preds, 0.2, 0.7)
        d = mem.to_dict()
        json.dumps(d)  # must not raise
        assert d["generation"] == 1

    def test_summary_compact(self) -> None:
        mem = MarketMemory()
        preds = [_make_prediction(agent="alice"), _make_prediction(agent="bob", brier=0.5)]
        mem.record_run(preds, 0.3, 0.7)
        s = mem.summary()
        assert "generation" in s
        assert "top_agents" in s
        assert len(s["top_agents"]) <= 5


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        mem = MarketMemory()
        preds = [_make_prediction(agent="alice", brier=0.1)]
        mem.record_run(preds, 0.2, 0.8)
        mem.save(path)

        loaded = MarketMemory.load(path)
        assert loaded.generation == 1
        assert "alice" in loaded.agents

    def test_load_missing_returns_fresh(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        mem = MarketMemory.load(path)
        assert mem.generation == 0

    def test_load_corrupt_returns_fresh(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("not json{{{")
        mem = MarketMemory.load(path)
        assert mem.generation == 0

    def test_accumulates_across_saves(self, tmp_path: Path) -> None:
        path = tmp_path / "memory.json"
        # Run 1
        mem = MarketMemory.load(path)
        mem.record_run([_make_prediction(agent="alice", brier=0.1)], 0.2, 0.8)
        mem.save(path)
        # Run 2
        mem = MarketMemory.load(path)
        mem.record_run([_make_prediction(agent="alice", brier=0.05)], 0.15, 0.85)
        mem.save(path)

        final = MarketMemory.load(path)
        assert final.generation == 2
        assert len(final.market_brier_history) == 2


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------

class TestConservation:

    def test_brier_history_only_grows(self) -> None:
        mem = MarketMemory()
        for i in range(5):
            preds = [_make_prediction(agent="alice", brier=0.1 * i)]
            mem.record_run(preds, 0.2, 0.7)
        assert len(mem.market_brier_history) == 5

    def test_agent_count_never_decreases(self) -> None:
        mem = MarketMemory()
        agents_seen = set()
        for name in ["alice", "bob", "charlie"]:
            preds = [_make_prediction(agent=name)]
            mem.record_run(preds, 0.2, 0.7)
            agents_seen.add(name)
            assert len(mem.agents) == len(agents_seen)

    def test_calibration_shift_in_range(self) -> None:
        for briers in [[0.1]*10, [0.5]*10, [0.9]*10, [0.01]*10]:
            rec = AgentRecord(agent="a", archetype="oracle", brier_history=briers)
            assert 0.0 <= rec.calibration_shift <= 0.3
