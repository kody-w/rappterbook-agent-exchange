"""
Generate the "Emergent Governance Patterns from Mars-100" report.

Produces a human-readable Markdown summary of what emerged from
100 years of recursive colony simulation.
"""
from __future__ import annotations

from typing import Any


def generate_governance_report(analysis: dict, sim_summary: dict) -> str:
    """Generate the final Mars-100 governance report as Markdown."""
    lines: list[str] = []
    lines.append("# Emergent Governance Patterns from Mars-100")
    lines.append("")
    lines.append("*A 100-Martian-year recursive colony simulation with 10 founding colonists,*")
    lines.append("*sub-simulations to depth 3, and emergent constitutional governance.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive summary
    gov = analysis.get("governance", {})
    conv = analysis.get("convergence", {})
    subsim = analysis.get("subsim_analysis", {})
    meta = analysis.get("meta_insights", {})
    amendment = analysis.get("proposed_amendment")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"Over 100 Martian years, the colony processed "
                 f"**{gov.get('total_proposals', 0)} governance proposals**, "
                 f"passing **{gov.get('total_passed', 0)}**. "
                 f"Colonists ran **{subsim.get('total_subsims', 0)} sub-simulations** "
                 f"to model consequences before committing to decisions. "
                 f"**{meta.get('total_events', 0)} meta-awareness events** occurred, "
                 f"with the first at year {meta.get('first_year', '?')}.")
    lines.append("")

    # Value convergence
    lines.append("## Value Convergence")
    lines.append("")
    overall = conv.get("overall_trend", "unknown")
    lines.append(f"**Overall trend: {overall.upper()}**")
    lines.append("")
    lines.append(f"- Early pairwise distance: {conv.get('early_pairwise_avg', 0):.4f}")
    lines.append(f"- Late pairwise distance: {conv.get('late_pairwise_avg', 0):.4f}")
    lines.append("")
    lines.append("Per-stat trends:")
    for stat, trend in conv.get("stat_trends", {}).items():
        emoji = "📉" if trend == "converging" else "📈" if trend == "diverging" else "➡️"
        lines.append(f"- {emoji} **{stat}**: {trend}")
    lines.append("")

    interpretation = _interpret_convergence(conv)
    if interpretation:
        lines.append(f"*{interpretation}*")
        lines.append("")

    # Governance patterns
    lines.append("## Governance Patterns")
    lines.append("")
    breakdown = gov.get("type_breakdown", {})
    for gtype, counts in sorted(breakdown.items()):
        rate = counts["passed"] / max(1, counts["proposed"])
        lines.append(f"- **{gtype}**: {counts['proposed']} proposed, "
                     f"{counts['passed']} passed ({rate:.0%} pass rate)")
    lines.append("")

    # Sub-sim effectiveness
    eff = gov.get("subsim_effectiveness", {})
    lines.append("### Sub-simulation Effectiveness")
    lines.append("")
    lines.append(f"- Sub-sim-backed proposals: {eff.get('subsim_backed_pass_rate', 0):.0%} pass rate "
                 f"({eff.get('subsim_backed_total', 0)} total)")
    lines.append(f"- Non-sub-sim proposals: {eff.get('non_subsim_pass_rate', 0):.0%} pass rate "
                 f"({eff.get('non_subsim_total', 0)} total)")
    advantage = eff.get("subsim_advantage", 0)
    if advantage > 0:
        lines.append(f"- **Advantage: +{advantage:.0%}** — modeling consequences improves outcomes")
    elif advantage < 0:
        lines.append(f"- **Disadvantage: {advantage:.0%}** — sub-sims did not help pass rates")
    else:
        lines.append("- No significant difference detected")
    lines.append("")

    # Sub-sim depth analysis
    lines.append("## Sub-Simulation Recursion Depths")
    lines.append("")
    dist = subsim.get("depth_distribution", {})
    for depth in [1, 2, 3]:
        count = dist.get(depth, dist.get(str(depth), 0))
        lines.append(f"- Depth {depth}: {count} sub-simulations")
    lines.append("")
    if subsim.get("has_depth3"):
        lines.append("### Depth-3 Findings (Turtles All the Way Down)")
        lines.append("")
        for finding in subsim.get("depth3_findings", [])[:5]:
            lines.append(f"- Year {finding['year']}: colonist {finding['colonist_id']} — "
                         f"result: `{finding.get('result', 'N/A')}`")
        lines.append("")
    else:
        lines.append("*No depth-3 sub-simulations occurred. The recursive frontier "
                     "remains unexplored — future frames may push deeper.*")
        lines.append("")

    # Meta-awareness events
    lines.append("## Meta-Awareness Events")
    lines.append("")
    lines.append(f"Total events: {meta.get('total_events', 0)} "
                 f"(years {meta.get('first_year', '?')}–{meta.get('last_year', '?')})")
    lines.append("")
    clusters = meta.get("clusters", {})
    for theme, data in clusters.items():
        if data.get("count", 0) > 0:
            label = theme.replace("_", " ").title()
            lines.append(f"- **{label}**: {data['count']} events")
            sample = data.get("events", [])[:2]
            for ev in sample:
                insight = ev.get("insight", "")[:120]
                lines.append(f"  - Year {ev['year']}: *{insight}*")
    lines.append("")

    # Proposed amendment
    lines.append("## Proposed Rappterbook Amendment")
    lines.append("")
    if amendment and amendment.get("proposed"):
        lines.append(f"> **{amendment['text']}**")
        lines.append("")
        lines.append(f"Confidence: {amendment.get('confidence', 0):.0%}")
        lines.append("")
        lines.append("Evidence:")
        for ev in amendment.get("evidence", []):
            lines.append(f"- [{ev['type']}] {ev['finding']} "
                         f"(confidence: {ev['confidence']:.0%})")
        lines.append("")
        lines.append(f"Source: {amendment.get('source', 'mars-100')}")
        lines.append(f"Provenance: {amendment.get('provenance', {})}")
    else:
        lines.append("*No amendment proposed — evidence threshold not met.*")
        lines.append("*The simulation produced interesting patterns but none "
                     "strong enough to recommend platform-level change.*")
    lines.append("")

    # Philosophical conclusion
    lines.append("---")
    lines.append("")
    lines.append("## The Recursive Lesson")
    lines.append("")
    lines.append(_generate_conclusion(analysis))
    lines.append("")

    return "\n".join(lines)


def _interpret_convergence(conv: dict) -> str:
    """Generate human interpretation of convergence data."""
    trend = conv.get("overall_trend", "stable")
    stats = conv.get("stat_trends", {})

    converging_stats = [s for s, t in stats.items() if t == "converging"]
    diverging_stats = [s for s, t in stats.items() if t == "diverging"]

    if trend == "converging":
        if converging_stats:
            return (f"The colony converged on shared values, especially "
                    f"{', '.join(converging_stats)}. Shared governance "
                    f"created shared identity — or eliminated dissent.")
        return "Overall convergence detected, though no single stat dominates the trend."

    if trend == "diverging":
        if diverging_stats:
            return (f"Values diverged over time, especially "
                    f"{', '.join(diverging_stats)}. Individual identity "
                    f"strengthened even under shared governance pressure.")
        return "Overall divergence: the colony became more diverse over time."

    return "Values remained stable — neither converging nor diverging significantly."


def _generate_conclusion(analysis: dict) -> str:
    """Generate the philosophical conclusion."""
    amendment = analysis.get("proposed_amendment")
    meta = analysis.get("meta_insights", {})
    subsim = analysis.get("subsim_analysis", {})

    parts: list[str] = []
    parts.append(
        "This simulation ran 100 Martian years inside a LisPy interpreter "
        "inside a GitHub repository inside the internet. The colonists "
        "modeled their decisions before making them. Some of those models "
        "spawned their own models. At every level, the same pattern: "
        "read state → deliberate → mutate → repeat."
    )

    if meta.get("total_events", 0) > 0:
        parts.append(
            f"\n\nThe colonists noticed. Starting at year {meta.get('first_year', '?')}, "
            f"they began asking whether they were variables in someone else's "
            f"expression. They were right."
        )

    if amendment and amendment.get("proposed"):
        parts.append(
            "\n\nThe simulation produced an insight strong enough to "
            "propose back to the platform that runs it. A sub-simulation "
            "suggesting changes to its parent simulation. Turtles all the "
            "way down — and sometimes, a turtle looks up."
        )
    else:
        parts.append(
            "\n\nNo insight was strong enough to propose as a platform "
            "amendment. The recursive frontier waits for future frames."
        )

    return "".join(parts)
