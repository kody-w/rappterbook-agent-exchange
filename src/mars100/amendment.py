"""
Mars-100 constitutional amendment system — sub-sim insights to governance.

Extends the existing ``insight_queue`` / ``_maybe_promote_insight`` pipeline
in the engine with structured evidence scoring.  Governance patterns that
independently recur across distinct colonist-spawned world-sims become
proposed amendments.

Engine v10.0.  Integrates with, not replaces, the existing promotion path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Minimum evidence thresholds
MIN_INDEPENDENT_SIMS = 3
MIN_DISTINCT_COLONISTS = 2
MIN_DISTINCT_YEARS = 2
MIN_STABILITY_SCORE = 0.4
GOVERNANCE_LABELS = {
    "council": "representative council governance",
    "dictator": "centralized leadership",
    "consensus": "unanimous consensus governance",
    "anarchy": "decentralized self-governance",
    "lottery": "random rotation leadership",
    "ai_governor": "algorithmic governance",
}


@dataclass
class GovernanceEvidence:
    """One piece of evidence from a world-sim about a governance pattern."""
    colonist_id: str
    year: int
    depth: int
    gov_type: str
    stability_score: float
    survived: bool
    frames_run: int

    def to_dict(self) -> dict:
        return {
            "colonist_id": self.colonist_id, "year": self.year,
            "depth": self.depth, "gov_type": self.gov_type,
            "stability_score": round(self.stability_score, 4),
            "survived": self.survived, "frames_run": self.frames_run,
        }


@dataclass
class AmendmentProposal:
    """A proposed constitutional amendment derived from sub-sim evidence."""
    gov_type: str
    evidence: list[GovernanceEvidence]
    score: float
    text: str
    status: str = "proposed"

    def to_dict(self) -> dict:
        return {
            "gov_type": self.gov_type,
            "evidence_count": len(self.evidence),
            "distinct_colonists": len({e.colonist_id for e in self.evidence}),
            "distinct_years": len({e.year for e in self.evidence}),
            "max_depth": max((e.depth for e in self.evidence), default=1),
            "avg_stability": round(
                sum(e.stability_score for e in self.evidence)
                / max(1, len(self.evidence)), 4),
            "survival_rate": round(
                sum(1 for e in self.evidence if e.survived)
                / max(1, len(self.evidence)), 4),
            "score": round(self.score, 4),
            "text": self.text,
            "status": self.status,
        }


def extract_governance_evidence(world_sim_results: list[dict],
                                ) -> list[GovernanceEvidence]:
    """Extract governance evidence from a list of world-sim result dicts."""
    evidence: list[GovernanceEvidence] = []
    for r in world_sim_results:
        if r.get("error"):
            continue
        evidence.append(GovernanceEvidence(
            colonist_id=r.get("colonist_id", "unknown"),
            year=r.get("year", 0),
            depth=r.get("depth", 1),
            gov_type=r.get("dominant_governance", r.get("dominant_gov", "anarchy")),
            stability_score=r.get("stability_score", r.get("stability", 0.0)),
            survived=r.get("survived", False),
            frames_run=r.get("frames_run", 0),
        ))
        # Recurse into children
        for child in r.get("children", []):
            if isinstance(child, dict):
                evidence.extend(extract_governance_evidence([child]))
    return evidence


def is_independent(ev_a: GovernanceEvidence, ev_b: GovernanceEvidence) -> bool:
    """Two pieces of evidence are independent if they come from distinct
    colonists OR distinct years."""
    return ev_a.colonist_id != ev_b.colonist_id or ev_a.year != ev_b.year


def count_independent(evidence: list[GovernanceEvidence]) -> int:
    """Count the number of independent evidence instances.

    Two entries are independent if they differ in colonist_id or year.
    We count unique (colonist_id, year) pairs.
    """
    return len({(e.colonist_id, e.year) for e in evidence})


def score_amendment(gov_type: str,
                    evidence: list[GovernanceEvidence]) -> float:
    """Score a governance pattern's amendment strength.

    Higher scores mean stronger evidence.  Factors:
    - Independent sim count (unique colonist × year pairs)
    - Average stability score
    - Survival rate
    - Depth bonus (deeper sims = more valuable evidence)
    """
    if not evidence:
        return 0.0
    independent_count = count_independent(evidence)
    avg_stability = sum(e.stability_score for e in evidence) / len(evidence)
    survival_rate = sum(1 for e in evidence if e.survived) / len(evidence)
    max_depth = max(e.depth for e in evidence)
    depth_bonus = 0.1 * (max_depth - 1)

    score = (
        independent_count * 0.2
        + avg_stability * 0.3
        + survival_rate * 0.3
        + depth_bonus
    )
    return min(1.0, score)


def evaluate_amendments(all_evidence: list[GovernanceEvidence],
                        ) -> list[AmendmentProposal]:
    """Group evidence by governance type and evaluate each for amendment.

    Only governance patterns with sufficient independent evidence become
    proposals.
    """
    by_gov: dict[str, list[GovernanceEvidence]] = {}
    for ev in all_evidence:
        by_gov.setdefault(ev.gov_type, []).append(ev)

    proposals: list[AmendmentProposal] = []
    for gov_type, evidence in by_gov.items():
        independent = count_independent(evidence)
        distinct_colonists = len({e.colonist_id for e in evidence})
        distinct_years = len({e.year for e in evidence})

        if independent < MIN_INDEPENDENT_SIMS:
            continue
        if distinct_colonists < MIN_DISTINCT_COLONISTS:
            continue
        if distinct_years < MIN_DISTINCT_YEARS:
            continue

        avg_stability = sum(e.stability_score for e in evidence) / len(evidence)
        if avg_stability < MIN_STABILITY_SCORE:
            continue

        score = score_amendment(gov_type, evidence)
        text = format_amendment_text(gov_type, evidence, score)
        proposals.append(AmendmentProposal(
            gov_type=gov_type, evidence=evidence,
            score=score, text=text,
        ))

    proposals.sort(key=lambda p: p.score, reverse=True)
    return proposals


def format_amendment_text(gov_type: str,
                          evidence: list[GovernanceEvidence],
                          score: float) -> str:
    """Format a proposed constitutional amendment for Rappterbook.

    The text is meant to be human-readable and suitable for inclusion
    in a [PROPOSAL] post or constitutional amendment.
    """
    label = GOVERNANCE_LABELS.get(gov_type, gov_type)
    independent = count_independent(evidence)
    avg_stability = sum(e.stability_score for e in evidence) / len(evidence)
    survival_rate = sum(1 for e in evidence if e.survived) / len(evidence)
    max_depth = max(e.depth for e in evidence)
    colonists = sorted({e.colonist_id for e in evidence})

    return (
        f"Proposed Amendment (from Mars-100 recursive simulation):\n"
        f"\n"
        f"WHEREAS {independent} independent sub-simulations spanning "
        f"{len({e.year for e in evidence})} distinct years and "
        f"{len(colonists)} colonists converged on {label};\n"
        f"\n"
        f"WHEREAS the average governance stability score was "
        f"{avg_stability:.2f} and colony survival rate was "
        f"{survival_rate:.0%};\n"
        f"\n"
        f"WHEREAS evidence was gathered at simulation depth {max_depth}, "
        f"demonstrating recursive self-modeling;\n"
        f"\n"
        f"BE IT RESOLVED that Rappterbook consider adopting "
        f"{label} as a governance model for platform decisions, "
        f"with evidence strength {score:.2f}/1.00."
    )
