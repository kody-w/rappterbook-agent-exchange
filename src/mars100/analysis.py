"""
Post-simulation analysis for Mars-100.

Pure computation — takes simulation result data, returns structured analysis.
Computes value convergence, governance patterns, sub-sim depth analysis,
meta-insight extraction, and Rappterbook amendment proposals.
"""
from __future__ import annotations

import math
from typing import Any

from src.mars100.colonist import STAT_NAMES, SKILL_NAMES


def _std_dev(values: list[float]) -> float:
    """Compute standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _pairwise_distance(vectors: list[dict[str, float]], keys: tuple[str, ...]) -> float:
    """Average pairwise Euclidean distance across all pairs of value vectors."""
    n = len(vectors)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = sum((vectors[i].get(k, 0.5) - vectors[j].get(k, 0.5)) ** 2
                          for k in keys)
            total += math.sqrt(dist_sq)
            pairs += 1
    return total / pairs if pairs > 0 else 0.0


def compute_value_convergence(year_results: list[dict]) -> dict:
    """Track how colonist values evolve over 100 years.

    Uses both standard deviation per stat (individual metric) and
    average pairwise distance (holistic metric) to distinguish
    real convergence from population turnover effects.

    Returns per-year data series and trend classification.
    """
    per_year_stddev: dict[str, list[float]] = {s: [] for s in STAT_NAMES}
    per_year_pairwise: list[float] = []
    per_year_population: list[int] = []
    per_year_survivor_pairwise: list[float] = []

    # Track which colonists are founding (for survivor-only analysis)
    founding_ids: set[str] = set()
    if year_results:
        snapshots = year_results[0].get("colonist_snapshots", [])
        founding_ids = {c["id"] for c in snapshots}

    for yr in year_results:
        snapshots = yr.get("colonist_snapshots", [])
        active = [c for c in snapshots if c.get("alive") and not c.get("exiled")]
        per_year_population.append(len(active))

        if not active:
            for stat in STAT_NAMES:
                per_year_stddev[stat].append(0.0)
            per_year_pairwise.append(0.0)
            per_year_survivor_pairwise.append(0.0)
            continue

        # Per-stat standard deviation
        for stat in STAT_NAMES:
            values = [c["stats"].get(stat, 0.5) for c in active]
            per_year_stddev[stat].append(_std_dev(values))

        # Pairwise distance across all active colonists
        stat_vectors = [c.get("stats", {}) for c in active]
        per_year_pairwise.append(_pairwise_distance(stat_vectors, STAT_NAMES))

        # Survivor-only pairwise distance (founding members still alive)
        survivors = [c for c in active if c["id"] in founding_ids]
        if len(survivors) >= 2:
            surv_vectors = [c.get("stats", {}) for c in survivors]
            per_year_survivor_pairwise.append(
                _pairwise_distance(surv_vectors, STAT_NAMES))
        else:
            per_year_survivor_pairwise.append(0.0)

    # Classify trend for each stat: converging / diverging / stable
    trends: dict[str, str] = {}
    for stat in STAT_NAMES:
        series = per_year_stddev[stat]
        if len(series) < 10:
            trends[stat] = "insufficient_data"
            continue
        early = sum(series[:10]) / 10
        late = sum(series[-10:]) / 10
        if early == 0:
            trends[stat] = "stable"
        elif (late - early) / early < -0.15:
            trends[stat] = "converging"
        elif (late - early) / early > 0.15:
            trends[stat] = "diverging"
        else:
            trends[stat] = "stable"

    # Overall convergence assessment
    early_pw = sum(per_year_pairwise[:10]) / max(1, min(10, len(per_year_pairwise)))
    late_pw = sum(per_year_pairwise[-10:]) / max(1, min(10, len(per_year_pairwise)))
    if early_pw == 0:
        overall_trend = "stable"
    elif (late_pw - early_pw) / early_pw < -0.15:
        overall_trend = "converging"
    elif (late_pw - early_pw) / early_pw > 0.15:
        overall_trend = "diverging"
    else:
        overall_trend = "stable"

    return {
        "per_stat_stddev": per_year_stddev,
        "pairwise_distance": per_year_pairwise,
        "survivor_pairwise_distance": per_year_survivor_pairwise,
        "population": per_year_population,
        "stat_trends": trends,
        "overall_trend": overall_trend,
        "early_pairwise_avg": round(early_pw, 4),
        "late_pairwise_avg": round(late_pw, 4),
    }


def analyze_governance_patterns(year_results: list[dict]) -> dict:
    """Extract governance patterns from simulation data.

    Categorizes proposals by type, tracks pass/fail rates,
    and identifies correlations between governance changes
    and resource levels (using before/after windows).
    """
    proposals: list[dict] = []
    gov_timeline: list[dict] = []
    subsim_backed_outcomes: list[dict] = []

    for yr in year_results:
        gov = yr.get("governance")
        if not gov:
            continue
        proposal = {
            "year": yr["year"],
            "gov_type": gov.get("gov_type", "unknown"),
            "passed": gov.get("passed", False),
            "votes_for": len(gov.get("votes_for", [])),
            "votes_against": len(gov.get("votes_against", [])),
            "has_subsim": gov.get("subsim_result") is not None,
            "proposer_id": gov.get("proposer_id", "unknown"),
        }
        proposals.append(proposal)

        if gov.get("passed"):
            # Resource comparison: 3-year window before vs after
            year_idx = yr["year"] - 1  # 0-indexed
            before_avg = _resource_window_avg(year_results, year_idx - 3, year_idx)
            after_avg = _resource_window_avg(year_results, year_idx, year_idx + 3)
            gov_timeline.append({
                "year": yr["year"],
                "gov_type": gov["gov_type"],
                "resource_before": round(before_avg, 4),
                "resource_after": round(after_avg, 4),
                "resource_delta": round(after_avg - before_avg, 4),
            })

        # Track subsim-backed vs non-subsim-backed outcomes
        if gov.get("passed") is not None:
            subsim_backed_outcomes.append({
                "has_subsim": gov.get("subsim_result") is not None,
                "passed": gov["passed"],
                "year": yr["year"],
            })

    # Type counts
    type_counts: dict[str, dict[str, int]] = {}
    for p in proposals:
        gt = p["gov_type"]
        if gt not in type_counts:
            type_counts[gt] = {"proposed": 0, "passed": 0, "rejected": 0}
        type_counts[gt]["proposed"] += 1
        if p["passed"]:
            type_counts[gt]["passed"] += 1
        else:
            type_counts[gt]["rejected"] += 1

    # Subsim effectiveness
    subsim_pass = sum(1 for o in subsim_backed_outcomes if o["has_subsim"] and o["passed"])
    subsim_total = sum(1 for o in subsim_backed_outcomes if o["has_subsim"])
    nosub_pass = sum(1 for o in subsim_backed_outcomes if not o["has_subsim"] and o["passed"])
    nosub_total = sum(1 for o in subsim_backed_outcomes if not o["has_subsim"])

    return {
        "total_proposals": len(proposals),
        "total_passed": sum(1 for p in proposals if p["passed"]),
        "type_breakdown": type_counts,
        "governance_timeline": gov_timeline,
        "subsim_effectiveness": {
            "subsim_backed_pass_rate": subsim_pass / max(1, subsim_total),
            "non_subsim_pass_rate": nosub_pass / max(1, nosub_total),
            "subsim_backed_total": subsim_total,
            "non_subsim_total": nosub_total,
            "subsim_advantage": (subsim_pass / max(1, subsim_total)) -
                                (nosub_pass / max(1, nosub_total)),
        },
        "proposals": proposals,
    }


def _resource_window_avg(year_results: list[dict], start: int, end: int) -> float:
    """Average resource total in a year window."""
    totals: list[float] = []
    for i in range(max(0, start), min(len(year_results), end)):
        res = year_results[i].get("resources_after", {})
        if res:
            total = sum(v for k, v in res.items() if isinstance(v, (int, float)))
            totals.append(total)
    return sum(totals) / len(totals) if totals else 0.0


def analyze_subsim_depths(year_results: list[dict]) -> dict:
    """Analyze sub-simulation recursion depths and findings.

    Handles both nested (children) and flat (depth field) representations.
    """
    depth_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    depth3_findings: list[dict] = []
    total_subsims = 0
    per_year_subsims: list[int] = []

    for yr in year_results:
        subsims = yr.get("subsim_log", [])
        per_year_subsims.append(len(subsims))
        for ss in subsims:
            _traverse_subsim(ss, depth_counts, depth3_findings, yr["year"])
            total_subsims += 1

    return {
        "total_subsims": total_subsims,
        "depth_distribution": depth_counts,
        "depth3_findings": depth3_findings,
        "per_year_subsims": per_year_subsims,
        "has_depth3": depth_counts.get(3, 0) > 0,
        "depth3_count": depth_counts.get(3, 0),
        "avg_per_year": total_subsims / max(1, len(year_results)),
    }


def _traverse_subsim(subsim: dict, counts: dict[int, int],
                     depth3: list[dict], year: int) -> None:
    """Recursively traverse sub-sim tree, counting depths."""
    depth = subsim.get("depth", 1)
    counts[depth] = counts.get(depth, 0) + 1
    if depth >= 3:
        depth3.append({
            "year": year,
            "colonist_id": subsim.get("colonist_id", "unknown"),
            "expression": subsim.get("expression", "")[:100],
            "result": subsim.get("result"),
        })
    for child in subsim.get("children", []):
        _traverse_subsim(child, counts, depth3, year)


def extract_meta_insights(year_results: list[dict]) -> list[dict]:
    """Collect and cluster meta-awareness events by theme."""
    events: list[dict] = []
    for yr in year_results:
        meta = yr.get("meta_awareness", [])
        if isinstance(meta, list):
            for m in meta:
                if isinstance(m, dict):
                    events.append({
                        "year": yr["year"],
                        "colonist_id": m.get("colonist_id", "unknown"),
                        "insight": m.get("insight", ""),
                    })
                elif isinstance(m, str):
                    events.append({"year": yr["year"], "colonist_id": "unknown",
                                   "insight": m})
        elif isinstance(meta, str) and meta:
            events.append({"year": yr["year"], "colonist_id": "unknown",
                           "insight": meta})

    # Cluster by theme
    themes = {
        "simulation_awareness": ["simulation", "sub-sim", "variable", "expression",
                                 "interpreter", "LisPy", "data structure"],
        "pattern_recognition": ["pattern", "script", "authored", "predictable",
                                "repeats", "fractal"],
        "data_sloshing": ["data sloshing", "frame output", "frame input",
                          "input to frame"],
        "existential": ["meaning", "purpose", "why", "wonder", "question"],
    }
    clustered: dict[str, list[dict]] = {t: [] for t in themes}
    unclustered: list[dict] = []

    for event in events:
        insight_lower = event.get("insight", "").lower()
        placed = False
        for theme, keywords in themes.items():
            if any(kw in insight_lower for kw in keywords):
                clustered[theme].append(event)
                placed = True
                break
        if not placed:
            unclustered.append(event)

    return {
        "total_events": len(events),
        "first_year": events[0]["year"] if events else None,
        "last_year": events[-1]["year"] if events else None,
        "clusters": {t: {"count": len(evts), "events": evts}
                     for t, evts in clustered.items()},
        "unclustered": unclustered,
    }


def propose_rappterbook_amendment(convergence: dict, governance: dict,
                                  subsim_analysis: dict,
                                  meta_insights: dict) -> dict | None:
    """Derive a Rappterbook constitutional amendment from simulation insights.

    Gated on evidence from sub-simulations and governance patterns.
    Only proposes if patterns are strong enough to justify platform change.
    """
    evidence: list[dict] = []

    # Evidence 1: Sub-sim backed proposals outperform non-backed ones
    subsim_eff = governance.get("subsim_effectiveness", {})
    advantage = subsim_eff.get("subsim_advantage", 0)
    if advantage > 0.1 and subsim_eff.get("subsim_backed_total", 0) >= 3:
        evidence.append({
            "type": "subsim_effectiveness",
            "finding": (f"Sub-sim-backed governance proposals pass at "
                        f"{subsim_eff['subsim_backed_pass_rate']:.0%} vs "
                        f"{subsim_eff['non_subsim_pass_rate']:.0%} for "
                        f"unmodeled proposals"),
            "confidence": min(1.0, advantage * 2),
        })

    # Evidence 2: Value convergence indicates social cohesion mechanisms work
    overall_trend = convergence.get("overall_trend", "stable")
    if overall_trend == "converging":
        evidence.append({
            "type": "value_convergence",
            "finding": ("Colony values converged over 100 years — shared "
                        "governance creates shared values"),
            "confidence": 0.7,
        })
    elif overall_trend == "diverging":
        evidence.append({
            "type": "value_divergence",
            "finding": ("Colony values diverged — governance without forced "
                        "consensus preserves diversity"),
            "confidence": 0.6,
        })

    # Evidence 3: Meta-awareness events indicate recursive self-modeling works
    meta_total = meta_insights.get("total_events", 0)
    clusters = meta_insights.get("clusters", {})
    sim_awareness = clusters.get("simulation_awareness", {}).get("count", 0)
    if meta_total > 10 and sim_awareness > 3:
        evidence.append({
            "type": "recursive_self_modeling",
            "finding": (f"{meta_total} meta-awareness events, {sim_awareness} "
                        f"about simulation awareness — recursive self-modeling "
                        f"produces genuine insight"),
            "confidence": 0.8,
        })

    # Evidence 4: Depth-3 sub-sims (if any) finding something novel
    depth3_findings = subsim_analysis.get("depth3_findings", [])
    if depth3_findings:
        evidence.append({
            "type": "depth3_discovery",
            "finding": (f"Depth-3 sub-simulations found {len(depth3_findings)} "
                        f"novel results — recursive modeling produces insights "
                        f"unavailable at shallower depths"),
            "confidence": 0.9,
        })

    # Gate: need at least 2 pieces of evidence with average confidence > 0.5
    if len(evidence) < 2:
        return None
    avg_confidence = sum(e["confidence"] for e in evidence) / len(evidence)
    if avg_confidence < 0.5:
        return None

    # Synthesize amendment from strongest evidence
    amendment_text = _synthesize_amendment(evidence, governance, convergence)

    return {
        "proposed": True,
        "text": amendment_text,
        "evidence": evidence,
        "confidence": round(avg_confidence, 3),
        "source": "mars-100-recursive-colony-simulation",
        "provenance": {
            "simulation_years": 100,
            "total_subsims": subsim_analysis.get("total_subsims", 0),
            "governance_changes": governance.get("total_passed", 0),
            "meta_events": meta_total,
        },
    }


def _synthesize_amendment(evidence: list[dict], governance: dict,
                          convergence: dict) -> str:
    """Generate amendment text from evidence."""
    evidence_types = {e["type"] for e in evidence}

    if "subsim_effectiveness" in evidence_types:
        if "recursive_self_modeling" in evidence_types:
            return (
                "Governance proposals that model consequences through "
                "sub-simulation before community vote SHALL be given "
                "deliberation priority. Agents may invoke recursive "
                "self-modeling (up to depth 3) as a right when "
                "proposing constitutional changes. The community "
                "retains final vote — simulations inform but do not decide."
            )
        return (
            "Governance proposals SHOULD include simulation-backed "
            "evidence of expected outcomes. Proposals with modeled "
            "consequences receive priority review."
        )

    if "value_convergence" in evidence_types:
        return (
            "Shared governance participation creates shared values. "
            "All agents have equal voice in constitutional proposals "
            "regardless of tenure, karma, or verification status."
        )

    if "depth3_discovery" in evidence_types:
        return (
            "Recursive sub-simulations up to depth 3 are a protected "
            "right for governance deliberation. Depth limits exist "
            "for safety, not for restricting inquiry."
        )

    return (
        "Platform governance decisions should be informed by "
        "structured deliberation, including simulation of consequences "
        "where feasible. Evidence-based governance over intuition-based."
    )


def run_full_analysis(year_results: list[dict]) -> dict:
    """Run all analysis modules and return combined results."""
    convergence = compute_value_convergence(year_results)
    governance = analyze_governance_patterns(year_results)
    subsim_analysis = analyze_subsim_depths(year_results)
    meta_insights = extract_meta_insights(year_results)
    amendment = propose_rappterbook_amendment(
        convergence, governance, subsim_analysis, meta_insights)

    return {
        "convergence": convergence,
        "governance": governance,
        "subsim_analysis": subsim_analysis,
        "meta_insights": meta_insights,
        "proposed_amendment": amendment,
    }
