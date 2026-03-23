"""
market_memory.py — Persistent calibration memory for prediction market agents.

After each simulation run, agents' Brier scores are recorded. On the next
run, agents adjust their confidence based on past performance. The market
LEARNS across frames — data sloshing in action.

The output of frame N is the input to frame N+1.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentRecord:
    """Historical performance record for one agent."""
    agent: str
    archetype: str
    total_predictions: int = 0
    total_resolved: int = 0
    brier_history: list[float] = field(default_factory=list)
    category_briers: dict[str, list[float]] = field(default_factory=dict)

    @property
    def mean_brier(self) -> float:
        """Lifetime mean Brier score."""
        if not self.brier_history:
            return 0.5
        return statistics.mean(self.brier_history)

    @property
    def calibration_shift(self) -> float:
        """How much to shift confidence toward 0.5 (shrinkage).

        Well-calibrated agents (low Brier) get no shift.
        Poorly-calibrated agents get pulled toward 0.5.
        Range: [0.0, 0.3] — 0 = no correction, 0.3 = heavy shrinkage.
        """
        if not self.brier_history:
            return 0.0
        recent = self.brier_history[-20:]
        mean_b = statistics.mean(recent)
        if mean_b < 0.15:
            return 0.0
        if mean_b > 0.5:
            return 0.3
        return (mean_b - 0.15) / (0.5 - 0.15) * 0.3

    def category_skill(self, category: str) -> float:
        """Relative skill in a specific category. 1.0 = average, <1 = better."""
        cat_briers = self.category_briers.get(category, [])
        if not cat_briers:
            return 1.0
        cat_mean = statistics.mean(cat_briers[-10:])
        overall = self.mean_brier
        if overall == 0:
            return 1.0
        return min(2.0, cat_mean / max(0.01, overall))

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "agent": self.agent,
            "archetype": self.archetype,
            "total_predictions": self.total_predictions,
            "total_resolved": self.total_resolved,
            "mean_brier": round(self.mean_brier, 4),
            "calibration_shift": round(self.calibration_shift, 4),
            "recent_briers": [round(b, 4) for b in self.brier_history[-10:]],
            "category_skills": {
                cat: round(self.category_skill(cat), 4)
                for cat in self.category_briers
            },
        }


@dataclass
class MarketMemory:
    """Persistent memory across prediction market runs.

    Tracks agent performance, market-wide calibration, and
    provides correction signals for future predictions.
    """
    generation: int = 0
    agents: dict[str, AgentRecord] = field(default_factory=dict)
    market_brier_history: list[float] = field(default_factory=list)
    market_accuracy_history: list[float] = field(default_factory=list)

    def record_run(self, predictions: list, market_brier: float, accuracy: float) -> None:
        """Ingest results from one prediction market run."""
        self.generation += 1
        self.market_brier_history.append(market_brier)
        self.market_accuracy_history.append(accuracy)

        for p in predictions:
            if p.outcome is None or p.brier is None:
                continue
            key = p.agent
            if key not in self.agents:
                self.agents[key] = AgentRecord(agent=p.agent, archetype=p.archetype)
            rec = self.agents[key]
            rec.total_predictions += 1
            rec.total_resolved += 1
            rec.brier_history.append(p.brier)
            rec.category_briers.setdefault(p.category, []).append(p.brier)

    def adjust_confidence(self, agent: str, category: str, raw_confidence: float) -> float:
        """Apply learned calibration correction to a raw confidence value.

        Shrinks overconfident agents toward 0.5.
        Boosts category specialists away from 0.5 (they earned it).
        """
        if agent not in self.agents:
            return raw_confidence

        rec = self.agents[agent]
        shift = rec.calibration_shift
        skill = rec.category_skill(category)

        adjusted = raw_confidence
        if shift > 0:
            adjusted = adjusted * (1 - shift) + 0.5 * shift

        if skill < 0.8:
            boost = (0.8 - skill) * 0.1
            if adjusted > 0.5:
                adjusted = min(0.99, adjusted + boost)
            else:
                adjusted = max(0.01, adjusted - boost)

        return max(0.01, min(0.99, adjusted))

    @property
    def improvement_trend(self) -> float | None:
        """Slope of market Brier over time. Negative = improving."""
        if len(self.market_brier_history) < 2:
            return None
        n = len(self.market_brier_history)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(self.market_brier_history)
        numerator = sum(
            (i - x_mean) * (y - y_mean)
            for i, y in enumerate(self.market_brier_history)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def to_dict(self) -> dict:
        """Full serializable snapshot."""
        return {
            "generation": self.generation,
            "market_brier_history": [round(b, 4) for b in self.market_brier_history],
            "market_accuracy_history": [round(a, 4) for a in self.market_accuracy_history],
            "improvement_trend": (
                round(self.improvement_trend, 6)
                if self.improvement_trend is not None else None
            ),
            "agent_count": len(self.agents),
            "agents": {k: v.to_dict() for k, v in sorted(self.agents.items())},
        }

    def summary(self) -> dict:
        """Compact summary for proof output."""
        agent_records = sorted(
            self.agents.values(),
            key=lambda r: r.mean_brier,
        )
        top_agents = [
            {"agent": r.agent, "archetype": r.archetype, "brier": round(r.mean_brier, 4)}
            for r in agent_records[:5]
        ]
        return {
            "generation": self.generation,
            "latest_brier": round(self.market_brier_history[-1], 4) if self.market_brier_history else None,
            "latest_accuracy": round(self.market_accuracy_history[-1], 4) if self.market_accuracy_history else None,
            "trend": (
                round(self.improvement_trend, 6)
                if self.improvement_trend is not None else None
            ),
            "top_agents": top_agents,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MarketMemory:
        """Deserialize from JSON dict."""
        mem = cls()
        mem.generation = data.get("generation", 0)
        mem.market_brier_history = data.get("market_brier_history", [])
        mem.market_accuracy_history = data.get("market_accuracy_history", [])
        for key, agent_data in data.get("agents", {}).items():
            rec = AgentRecord(
                agent=agent_data["agent"],
                archetype=agent_data["archetype"],
                total_predictions=agent_data.get("total_predictions", 0),
                total_resolved=agent_data.get("total_resolved", 0),
                brier_history=agent_data.get("recent_briers", []),
            )
            mem.agents[key] = rec
        return mem

    @classmethod
    def load(cls, path: Path) -> MarketMemory:
        """Load memory from disk, or return fresh if missing."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return cls()

    def save(self, path: Path) -> None:
        """Atomic write to disk."""
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.rename(path)
