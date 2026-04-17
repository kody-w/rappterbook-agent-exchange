"""
Meta-insight extraction for Mars-100.

When depth-3 sub-simulations produce governance-relevant results,
extract them as potential constitutional amendments for Rappterbook.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.subsim import SubSimResult, SubSimBudget, spawn_subsim
from src.mars100.colonist import Colonist


INSIGHT_THRESHOLD = 0.7
AMENDMENT_THRESHOLD = 0.85

GOVERNANCE_SIM_TEMPLATES = [
    # Council simulation: does collective deliberation improve outcomes?
    "(let ((council-score (+ (* empathy 0.4) (* resolve 0.3) (* mediation 0.3))))"
    " (if (> council-score 0.6) (+ council-score 0.2) (- council-score 0.1)))",
    # Consensus simulation: can unanimous agreement be reached?
    "(let ((consensus (min empathy faith resolve)))"
    " (if (> consensus 0.5) (* consensus 1.5) 0))",
    # Resource equity: does sharing outperform hoarding?
    "(let ((equity (- 1 hoarding)) (survival (+ food water power)))"
    " (if (> equity 0.6) (+ survival equity) (- survival (* hoarding 0.5))))",
    # Trust network: does high trust correlate with colony survival?
    "(let ((trust-metric (+ (* empathy 0.5) (* faith 0.3) (* resolve 0.2))))"
    " (if (> trust-metric 0.65) (+ trust-metric 0.3) (- trust-metric paranoia)))",
    # Rotation: does rotating leadership outperform static?
    "(let ((rotation-value (+ improvisation (* empathy 0.5))))"
    " (if (> rotation-value 0.7) (* rotation-value resolve) (* paranoia -0.3)))",
]

AMENDMENT_TEMPLATES = [
    {
        "id": "recursive-governance",
        "text": "Any governance proposal affecting more than 3 agents must be "
                "modeled in a sandboxed sub-simulation before vote. Results "
                "become part of the public record.",
        "trigger": "council-score",
    },
    {
        "id": "consensus-mandate",
        "text": "Decisions affecting shared resources require 60%+ agreement. "
                "Sub-simulation evidence counts as one advisory vote.",
        "trigger": "consensus",
    },
    {
        "id": "anti-hoarding",
        "text": "No agent may accumulate more than 2x the colony average of "
                "any resource. Surplus is redistributed quarterly.",
        "trigger": "equity",
    },
    {
        "id": "trust-transparency",
        "text": "All governance sub-simulations must be logged and auditable. "
                "Hidden simulations are grounds for exile vote.",
        "trigger": "trust-metric",
    },
    {
        "id": "rotation-principle",
        "text": "Leadership positions rotate every N frames (colony decides N). "
                "No agent may serve consecutive terms without 75% approval.",
        "trigger": "rotation",
    },
]


@dataclass
class MetaInsight:
    """An insight extracted from a depth-3 sub-simulation."""
    year: int
    colonist_id: str
    depth: int
    expression: str
    result: Any
    insight_score: float
    amendment: dict | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "year": self.year, "colonist_id": self.colonist_id,
            "depth": self.depth, "expression": self.expression[:200],
            "result": self.result if isinstance(self.result, (int, float, str, bool, type(None))) else str(self.result),
            "insight_score": self.insight_score,
        }
        if self.amendment:
            d["amendment"] = self.amendment
        return d


@dataclass
class InsightCollector:
    """Collects and scores meta-insights across the simulation."""
    insights: list[MetaInsight] = field(default_factory=list)
    proposed_amendments: list[dict] = field(default_factory=list)

    def extract_from_subsim(
        self,
        result: SubSimResult,
        rng: random.Random,
    ) -> MetaInsight | None:
        """Check if a sub-sim result contains an actionable insight."""
        if not result.succeeded:
            return None
        if not isinstance(result.result, (int, float)):
            return None

        score = abs(float(result.result))
        normalized_score = min(1.0, score / 2.0)

        if normalized_score < INSIGHT_THRESHOLD:
            return None

        insight = MetaInsight(
            year=result.year,
            colonist_id=result.colonist_id,
            depth=result.depth,
            expression=result.expression,
            result=result.result,
            insight_score=round(normalized_score, 4),
        )

        if normalized_score >= AMENDMENT_THRESHOLD:
            amendment = rng.choice(AMENDMENT_TEMPLATES)
            insight.amendment = {
                "id": amendment["id"],
                "text": amendment["text"],
                "year": result.year,
                "source_colonist": result.colonist_id,
                "source_depth": result.depth,
                "evidence_score": round(normalized_score, 4),
            }
            self.proposed_amendments.append(insight.amendment)

        self.insights.append(insight)
        return insight

    def try_deep_governance_sim(
        self,
        colonist: Colonist,
        year: int,
        bindings: dict[str, Any],
        budget: SubSimBudget,
        subsim_log: list[SubSimResult],
        rng: random.Random,
    ) -> MetaInsight | None:
        """Attempt a depth-2 governance sim that may cascade to depth-3."""
        if not budget.can_spawn(colonist.id):
            return None

        expr = rng.choice(GOVERNANCE_SIM_TEMPLATES)
        d2_result = spawn_subsim(
            expression=expr, colonist_id=colonist.id,
            year=year, bindings=bindings,
            depth=2, budget=budget, log=subsim_log,
        )

        if not d2_result.succeeded or not isinstance(d2_result.result, (int, float)):
            return None

        d2_score = abs(float(d2_result.result))
        if d2_score < 0.6 or not budget.can_spawn(colonist.id):
            return self.extract_from_subsim(d2_result, rng)

        # Cascade to depth-3 with enriched context
        deeper_bindings = dict(bindings)
        deeper_bindings["parent-result"] = d2_result.result
        deeper_bindings["gov-evidence"] = d2_score

        deeper_expr = (
            "(let ((evidence gov-evidence) (meta (* parent-result sim-depth)))"
            " (if (> (+ evidence meta) 1.0)"
            "   (+ evidence (* meta 0.5))"
            "   (* evidence (- 1 paranoia))))"
        )
        d3_result = spawn_subsim(
            expression=deeper_expr, colonist_id=colonist.id,
            year=year, bindings=deeper_bindings,
            depth=3, budget=budget, log=subsim_log,
        )
        if d3_result.succeeded:
            d2_result.children.append(d3_result)

        return self.extract_from_subsim(d3_result, rng)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all collected insights."""
        return {
            "total_insights": len(self.insights),
            "total_amendments": len(self.proposed_amendments),
            "deepest_insight": max(
                (i.depth for i in self.insights), default=0
            ),
            "highest_score": max(
                (i.insight_score for i in self.insights), default=0.0
            ),
            "amendments": self.proposed_amendments[:5],
        }
