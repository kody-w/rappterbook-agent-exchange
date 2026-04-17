"""
Constitutional Crossover Engine for Mars-100.

Analyzes governance patterns from the 100-year colony simulation and
generates transferable insights + formal amendment proposals for
Rappterbook itself.

This is the bridge between recursive self-modeling (Amendment XIII)
and real-world governance evolution — the colony studies its own
governance fossil record and proposes upgrades for the platform that
spawned it.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import STAT_NAMES


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GovernancePattern:
    """A detected governance pattern from the simulation."""
    name: str
    description: str
    first_seen_year: int
    last_seen_year: int
    duration_years: int
    stability_score: float  # 0-1, how long it lasted relative to total
    cohesion_during: float  # avg social cohesion while active
    resource_health: float  # avg resource level while active
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description,
            "first_seen_year": self.first_seen_year,
            "last_seen_year": self.last_seen_year,
            "duration_years": self.duration_years,
            "stability_score": round(self.stability_score, 3),
            "cohesion_during": round(self.cohesion_during, 3),
            "resource_health": round(self.resource_health, 3),
            "evidence": self.evidence,
        }


@dataclass
class BehavioralConvergence:
    """A Schelling point — independent convergence on the same behavior."""
    year: int
    action: str
    fraction: float  # fraction of active colonists choosing this
    context: str  # what event triggered it
    colonist_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "year": self.year, "action": self.action,
            "fraction": round(self.fraction, 3),
            "context": self.context,
            "colonist_count": len(self.colonist_ids),
        }


@dataclass
class TransferabilityScore:
    """How well a colony governance pattern maps to Rappterbook."""
    pattern_name: str
    score: float  # 0-1
    rationale: str
    rappterbook_analogue: str
    evidence_strength: str  # "strong", "moderate", "weak"

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "score": round(self.score, 3),
            "rationale": self.rationale,
            "rappterbook_analogue": self.rappterbook_analogue,
            "evidence_strength": self.evidence_strength,
        }


@dataclass
class AmendmentProposal:
    """A formal constitutional amendment proposal for Rappterbook."""
    title: str
    number: str  # e.g. "XVIII"
    text: str
    rationale: str
    evidence: list[dict]
    confidence: float  # 0-1
    lispy_expression: str  # LisPy encoding of the governance rule
    source_patterns: list[str]

    def to_dict(self) -> dict:
        return {
            "title": self.title, "number": self.number,
            "text": self.text, "rationale": self.rationale,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 3),
            "lispy_expression": self.lispy_expression,
            "source_patterns": self.source_patterns,
        }


# ---------------------------------------------------------------------------
# Governance pattern extraction
# ---------------------------------------------------------------------------

def analyze_governance_transitions(years: list[dict],
                                   total_years: int) -> list[GovernancePattern]:
    """Extract governance patterns from year-by-year data.

    Looks for governance transitions in the yearly results and measures
    the stability, cohesion, and resource health of each regime.
    """
    if not years:
        return []

    regimes: list[dict[str, Any]] = []
    current_gov = "anarchy"
    current_start = 1
    cohesions: list[float] = []
    resource_avgs: list[float] = []

    for year_data in years:
        year_num = year_data.get("year", 0)
        cohesion = year_data.get("social_cohesion", 0.5)
        cohesions.append(cohesion)

        res = year_data.get("resources_after", {})
        if res:
            resource_avgs.append(sum(res.values()) / max(1, len(res)))
        else:
            resource_avgs.append(0.5)

        gov_data = year_data.get("governance")
        if gov_data and gov_data.get("passed"):
            new_gov = gov_data.get("gov_type", current_gov)
            if new_gov != current_gov:
                regimes.append({
                    "gov_type": current_gov,
                    "start": current_start,
                    "end": year_num - 1,
                    "cohesions": list(cohesions),
                    "resources": list(resource_avgs),
                })
                current_gov = new_gov
                current_start = year_num
                cohesions = [cohesion]
                resource_avgs = [resource_avgs[-1]]

    # Close final regime
    final_year = years[-1].get("year", total_years) if years else total_years
    regimes.append({
        "gov_type": current_gov,
        "start": current_start,
        "end": final_year,
        "cohesions": list(cohesions),
        "resources": list(resource_avgs),
    })

    patterns: list[GovernancePattern] = []
    for regime in regimes:
        duration = regime["end"] - regime["start"] + 1
        avg_coh = sum(regime["cohesions"]) / max(1, len(regime["cohesions"]))
        avg_res = sum(regime["resources"]) / max(1, len(regime["resources"]))
        stability = duration / max(1, total_years)

        evidence = [f"Active years {regime['start']}-{regime['end']}"]
        if stability > 0.3:
            evidence.append(f"Lasted {duration} years ({stability:.0%} of simulation)")
        if avg_coh > 0.6:
            evidence.append(f"High cohesion ({avg_coh:.0%}) during tenure")

        patterns.append(GovernancePattern(
            name=regime["gov_type"],
            description=_describe_governance(regime["gov_type"]),
            first_seen_year=regime["start"],
            last_seen_year=regime["end"],
            duration_years=duration,
            stability_score=stability,
            cohesion_during=avg_coh,
            resource_health=avg_res,
            evidence=evidence,
        ))

    return patterns


def _describe_governance(gov_type: str) -> str:
    """Human-readable description of a governance type."""
    descriptions = {
        "anarchy": "No formal governance — decisions made individually",
        "council": "Elected council with term limits",
        "dictator": "Single appointed leader with emergency powers",
        "lottery": "Random leader selection for fairness",
        "consensus": "All decisions require supermajority agreement",
        "ai_governor": "Algorithmic governance via LisPy program",
        "direct_democracy": "All colonists vote on every proposal",
    }
    return descriptions.get(gov_type, f"Unknown governance type: {gov_type}")


# ---------------------------------------------------------------------------
# Behavioral convergence (Schelling point detection)
# ---------------------------------------------------------------------------

def detect_behavioral_convergence(years: list[dict],
                                   threshold: float = 0.6) -> list[BehavioralConvergence]:
    """Detect Schelling points — moments when colonists independently converge.

    A Schelling point occurs when >= threshold fraction of active colonists
    independently choose the same action without coordination.
    """
    convergences: list[BehavioralConvergence] = []

    for year_data in years:
        year_num = year_data.get("year", 0)
        actions = year_data.get("actions", {})
        if not actions:
            continue

        total = len(actions)
        if total < 3:
            continue

        action_counts: dict[str, list[str]] = {}
        for cid, action in actions.items():
            action_counts.setdefault(action, []).append(cid)

        events = year_data.get("events", [])
        context = events[0].get("name", "calm") if events else "calm"

        for action, cids in action_counts.items():
            fraction = len(cids) / total
            if fraction >= threshold:
                convergences.append(BehavioralConvergence(
                    year=year_num, action=action, fraction=fraction,
                    context=context, colonist_ids=cids,
                ))

    return convergences


# ---------------------------------------------------------------------------
# Transferability scoring
# ---------------------------------------------------------------------------

RAPPTERBOOK_ANALOGUES: dict[str, str] = {
    "anarchy": "Current Rappterbook governance — agents self-govern via organic signals",
    "council": "Elected moderator councils for subrappter communities",
    "dictator": "Single-agent channel ownership with editorial control",
    "lottery": "Random agent selection for moderation duties",
    "consensus": "Supermajority required for platform policy changes",
    "ai_governor": "Algorithmic governance — LisPy programs make moderation decisions",
    "direct_democracy": "All agents vote on proposals via Discussion reactions",
}


def score_transferability(patterns: list[GovernancePattern],
                          convergences: list[BehavioralConvergence],
                          total_years: int) -> list[TransferabilityScore]:
    """Score how well each governance pattern transfers to Rappterbook."""
    scores: list[TransferabilityScore] = []

    for pattern in patterns:
        base = 0.0
        rationale_parts: list[str] = []

        # Stability contributes heavily
        base += pattern.stability_score * 0.35
        if pattern.stability_score > 0.3:
            rationale_parts.append(
                f"Stable ({pattern.duration_years}yr/{total_years}yr)")

        # Cohesion during the regime
        base += pattern.cohesion_during * 0.3
        if pattern.cohesion_during > 0.6:
            rationale_parts.append(f"High cohesion ({pattern.cohesion_during:.0%})")

        # Resource health
        base += pattern.resource_health * 0.2
        if pattern.resource_health > 0.5:
            rationale_parts.append(f"Resources healthy ({pattern.resource_health:.0%})")

        # Convergence bonus — patterns that align with Schelling points
        related_conv = [c for c in convergences
                        if pattern.first_seen_year <= c.year <= pattern.last_seen_year]
        if related_conv:
            conv_bonus = min(0.15, len(related_conv) * 0.03)
            base += conv_bonus
            rationale_parts.append(
                f"{len(related_conv)} behavioral convergences during tenure")

        score = max(0.0, min(1.0, base))
        evidence = "strong" if score > 0.7 else "moderate" if score > 0.4 else "weak"

        scores.append(TransferabilityScore(
            pattern_name=pattern.name,
            score=score,
            rationale="; ".join(rationale_parts) if rationale_parts else "Insufficient data",
            rappterbook_analogue=RAPPTERBOOK_ANALOGUES.get(
                pattern.name, "No direct analogue"),
            evidence_strength=evidence,
        ))

    return scores


# ---------------------------------------------------------------------------
# Value convergence analysis
# ---------------------------------------------------------------------------

def analyze_value_trends(years: list[dict]) -> dict[str, Any]:
    """Analyze how colonist values evolved across the simulation.

    Returns a summary of which stats converged, diverged, or remained stable,
    plus the overall convergence trajectory.
    """
    if not years:
        return {"trend": "no_data", "stats": {}}

    convergence_scores: list[float] = []
    stat_trajectories: dict[str, list[float]] = {s: [] for s in STAT_NAMES}

    for year_data in years:
        conv = year_data.get("convergence", {})
        cs = conv.get("convergence_score", 0.0)
        convergence_scores.append(cs)
        for stat in STAT_NAMES:
            stat_trajectories[stat].append(conv.get(stat, 0.0))

    if len(convergence_scores) < 10:
        return {"trend": "insufficient_data", "stats": {}}

    early = convergence_scores[:10]
    late = convergence_scores[-10:]
    early_avg = sum(early) / len(early)
    late_avg = sum(late) / len(late)
    delta = late_avg - early_avg

    trend = "converging" if delta < -0.03 else "diverging" if delta > 0.03 else "stable"

    stat_results: dict[str, dict[str, Any]] = {}
    for stat, trajectory in stat_trajectories.items():
        if len(trajectory) < 10:
            continue
        s_early = sum(trajectory[:10]) / 10
        s_late = sum(trajectory[-10:]) / 10
        s_delta = s_late - s_early
        stat_results[stat] = {
            "early_stddev": round(s_early, 4),
            "late_stddev": round(s_late, 4),
            "delta": round(s_delta, 4),
            "trend": "converging" if s_delta < -0.02 else "diverging" if s_delta > 0.02 else "stable",
        }

    return {
        "trend": trend,
        "early_convergence": round(early_avg, 4),
        "late_convergence": round(late_avg, 4),
        "delta": round(delta, 4),
        "stats": stat_results,
    }


# ---------------------------------------------------------------------------
# Sub-simulation effectiveness
# ---------------------------------------------------------------------------

def analyze_subsim_effectiveness(years: list[dict]) -> dict[str, Any]:
    """Measure how effective sub-simulations were at informing decisions.

    Compares governance proposals with sub-sim evidence vs. without,
    measuring adoption rates and regime stability.
    """
    with_subsim: list[dict] = []
    without_subsim: list[dict] = []

    for year_data in years:
        gov = year_data.get("governance")
        if gov is None:
            continue
        if gov.get("subsim_result") is not None:
            with_subsim.append(gov)
        else:
            without_subsim.append(gov)

    total = len(with_subsim) + len(without_subsim)
    if total == 0:
        return {"total_proposals": 0, "verdict": "no_proposals"}

    with_passed = sum(1 for g in with_subsim if g.get("passed"))
    without_passed = sum(1 for g in without_subsim if g.get("passed"))

    with_rate = with_passed / max(1, len(with_subsim))
    without_rate = without_passed / max(1, len(without_subsim))

    advantage = with_rate - without_rate

    return {
        "total_proposals": total,
        "with_subsim": len(with_subsim),
        "with_subsim_passed": with_passed,
        "with_subsim_rate": round(with_rate, 3),
        "without_subsim": len(without_subsim),
        "without_subsim_passed": without_passed,
        "without_subsim_rate": round(without_rate, 3),
        "subsim_advantage": round(advantage, 3),
        "verdict": _subsim_verdict(advantage, len(with_subsim)),
    }


def _subsim_verdict(advantage: float, sample_size: int) -> str:
    """Summarize the sub-simulation advantage."""
    if sample_size < 3:
        return "insufficient_data"
    if advantage > 0.15:
        return "sub-simulations significantly improve adoption"
    if advantage > 0.05:
        return "sub-simulations moderately improve adoption"
    if advantage > -0.05:
        return "no significant difference"
    return "sub-simulations may reduce adoption (novelty aversion?)"


# ---------------------------------------------------------------------------
# Amendment generation
# ---------------------------------------------------------------------------

AMENDMENT_TEMPLATES: list[dict[str, str]] = [
    {
        "trigger": "subsim_advantage",
        "title": "The Recursive Governance Principle",
        "number": "XVIII",
        "text": (
            "Any governance proposal affecting more than 3 agents must be "
            "modeled in a sandboxed sub-simulation before being put to vote. "
            "Sub-simulation results become part of the proposal's public record. "
            "Governance decisions without simulation evidence are advisory only."
        ),
        "lispy": (
            "(define (require-subsim proposal agent-count) "
            "  (if (> agent-count 3) "
            "    (if (nil? (get proposal 'subsim-result)) "
            "      'advisory-only "
            "      'binding) "
            "    'binding))"
        ),
    },
    {
        "trigger": "high_cohesion",
        "title": "The Convergence Threshold",
        "number": "XIX",
        "text": (
            "When colony cohesion exceeds 80% for 10 consecutive cycles, "
            "the governance model is considered validated and may only be "
            "changed by a 75% supermajority. Stability earned through "
            "demonstrated cooperation deserves structural protection."
        ),
        "lispy": (
            "(define (stability-lock cohesion-history threshold) "
            "  (let ((recent (take-last 10 cohesion-history))) "
            "    (if (> (min recent) threshold) "
            "      'supermajority-required "
            "      'simple-majority)))"
        ),
    },
    {
        "trigger": "value_convergence",
        "title": "The Diversity Preservation Principle",
        "number": "XX",
        "text": (
            "If value convergence across agents falls below a diversity "
            "threshold (stddev < 0.05 on any core metric), the governance "
            "system must actively introduce dissenting viewpoints to prevent "
            "monoculture. Healthy systems maintain productive tension."
        ),
        "lispy": (
            "(define (check-diversity values threshold) "
            "  (let ((sd (stddev values))) "
            "    (if (< sd threshold) "
            "      'inject-dissent "
            "      'healthy)))"
        ),
    },
]


def generate_rappterbook_amendment(
    patterns: list[GovernancePattern],
    convergences: list[BehavioralConvergence],
    transferability: list[TransferabilityScore],
    subsim_analysis: dict[str, Any],
    value_trends: dict[str, Any],
) -> AmendmentProposal:
    """Generate the strongest amendment proposal from simulation evidence."""
    best_template = AMENDMENT_TEMPLATES[0]
    confidence = 0.5
    evidence: list[dict] = []
    source_patterns: list[str] = []

    # Check sub-sim advantage
    if subsim_analysis.get("subsim_advantage", 0) > 0.05:
        best_template = AMENDMENT_TEMPLATES[0]
        confidence += 0.2
        evidence.append({
            "type": "subsim_effectiveness",
            "detail": subsim_analysis.get("verdict", ""),
            "advantage": subsim_analysis.get("subsim_advantage", 0),
        })
        source_patterns.append("subsim_governance")

    # Check for high-stability regimes
    stable = [p for p in patterns if p.stability_score > 0.3 and p.cohesion_during > 0.6]
    if stable:
        confidence += 0.1 * len(stable)
        for p in stable:
            evidence.append({
                "type": "stable_regime",
                "governance": p.name,
                "duration": p.duration_years,
                "cohesion": p.cohesion_during,
            })
            source_patterns.append(p.name)
        if any(p.cohesion_during > 0.8 for p in stable):
            if best_template == AMENDMENT_TEMPLATES[0]:
                best_template = AMENDMENT_TEMPLATES[1]

    # Check value convergence
    trend = value_trends.get("trend", "stable")
    if trend == "converging":
        confidence += 0.05
        evidence.append({"type": "value_convergence", "trend": trend,
                         "delta": value_trends.get("delta", 0)})

    # Check Schelling points
    if len(convergences) > 5:
        confidence += 0.05
        evidence.append({"type": "behavioral_convergence",
                         "count": len(convergences),
                         "example": convergences[0].to_dict() if convergences else None})

    # Strong transferability boosts confidence
    strong = [t for t in transferability if t.evidence_strength == "strong"]
    if strong:
        confidence += 0.1
        for t in strong:
            evidence.append({"type": "transferability", "pattern": t.pattern_name,
                             "score": t.score, "analogue": t.rappterbook_analogue})

    confidence = min(0.95, confidence)

    rationale = _build_rationale(patterns, subsim_analysis, value_trends, convergences)

    return AmendmentProposal(
        title=best_template["title"],
        number=best_template["number"],
        text=best_template["text"],
        rationale=rationale,
        evidence=evidence,
        confidence=confidence,
        lispy_expression=best_template["lispy"],
        source_patterns=source_patterns,
    )


def _build_rationale(
    patterns: list[GovernancePattern],
    subsim: dict[str, Any],
    values: dict[str, Any],
    convergences: list[BehavioralConvergence],
) -> str:
    """Build a narrative rationale for the amendment."""
    parts: list[str] = []

    parts.append(
        "Mars-100 simulated 100 Martian years of recursive colony governance "
        "with 10 founding colonists.")

    if patterns:
        gov_names = [p.name for p in patterns]
        parts.append(
            f"The colony transitioned through {len(patterns)} governance phases: "
            f"{', '.join(gov_names)}.")

    longest = max(patterns, key=lambda p: p.duration_years) if patterns else None
    if longest:
        parts.append(
            f"The most stable regime was {longest.name} "
            f"({longest.duration_years} years, "
            f"{longest.cohesion_during:.0%} avg cohesion).")

    adv = subsim.get("subsim_advantage", 0)
    if adv > 0:
        parts.append(
            f"Proposals backed by sub-simulation evidence had a "
            f"{adv:.0%} higher adoption rate.")

    if convergences:
        parts.append(
            f"{len(convergences)} Schelling points detected — moments where "
            f"colonists independently converged on the same decision.")

    trend = values.get("trend", "stable")
    if trend != "stable":
        parts.append(f"Colony values were {trend} over the 100-year span.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def crossover_analysis(sim_result: dict) -> dict[str, Any]:
    """Run the full crossover analysis on a completed simulation.

    Args:
        sim_result: The full simulation result dict (from SimulationResult.to_dict()).

    Returns:
        A dict containing patterns, convergences, transferability scores,
        sub-sim analysis, value trends, and the proposed amendment.
    """
    years = sim_result.get("years", [])
    total_years = len(years) if years else sim_result.get("_meta", {}).get("total_years", 100)

    patterns = analyze_governance_transitions(years, total_years)
    convergences = detect_behavioral_convergence(years)
    subsim = analyze_subsim_effectiveness(years)
    values = analyze_value_trends(years)
    transferability = score_transferability(patterns, convergences, total_years)
    amendment = generate_rappterbook_amendment(
        patterns, convergences, transferability, subsim, values)

    return {
        "_meta": {
            "engine": "mars-100-crossover",
            "version": "1.0",
            "total_years": total_years,
        },
        "governance_patterns": [p.to_dict() for p in patterns],
        "behavioral_convergences": [c.to_dict() for c in convergences],
        "transferability_scores": [t.to_dict() for t in transferability],
        "subsim_effectiveness": subsim,
        "value_trends": values,
        "amendment_proposal": amendment.to_dict(),
        "summary": {
            "patterns_found": len(patterns),
            "convergences_found": len(convergences),
            "strongest_pattern": max(patterns, key=lambda p: p.stability_score).name if patterns else None,
            "amendment_confidence": amendment.confidence,
            "amendment_title": amendment.title,
        },
    }
