"""
Human-readable governance report for Mars-100.

Turns the structured analysis dict into a Markdown document:
"Emergent Governance Patterns from Mars-100".
"""
from __future__ import annotations

from typing import Any


def _interpret_convergence(conv: dict[str, Any]) -> str:
    """Turn convergence data into a prose paragraph."""
    verdict = conv.get("verdict", "stable")
    overall = conv.get("overall_convergence", 0.0)
    scores = conv.get("convergence_scores", {})

    if not scores:
        return "Insufficient data to assess value convergence."

    top = max(scores, key=scores.get)
    bottom = min(scores, key=scores.get)

    if verdict == "converging":
        return (
            f"Colonist values **converged** over the simulation (overall score "
            f"{overall:+.2f}). The strongest convergence was in **{top}** "
            f"({scores[top]:+.2f}), suggesting shared hardship selects for this "
            f"trait. The weakest was **{bottom}** ({scores[bottom]:+.2f})."
        )
    if verdict == "diverging":
        return (
            f"Colonist values **diverged** over the simulation (overall score "
            f"{overall:+.2f}). **{bottom}** diverged the most "
            f"({scores[bottom]:+.2f}), indicating the colony's social fabric "
            f"frayed along this axis."
        )
    return (
        f"Colonist values remained **stable** (overall score {overall:+.2f}). "
        f"Neither strong convergence nor divergence was observed."
    )


def _generate_conclusion(analysis: dict[str, Any]) -> str:
    """Synthesize a brief philosophical conclusion."""
    fitness = analysis.get("fitness", {})
    composite = fitness.get("composite", 0.0)
    amendment = analysis.get("amendment_proposal", {})
    meta = analysis.get("meta_emergence", {})

    lines = []

    if composite > 0.6:
        lines.append(
            "The colony thrived. Despite environmental crises and political upheaval, "
            "the recursive simulation substrate produced a resilient community."
        )
    elif composite > 0.3:
        lines.append(
            "The colony survived but struggled. Governance transitions and resource "
            "scarcity tested its adaptive capacity."
        )
    else:
        lines.append(
            "The colony barely survived. This run demonstrated the fragility of "
            "isolated communities under persistent stress."
        )

    if amendment.get("proposed"):
        best = amendment["amendment"]
        lines.append(
            f"\nThe strongest emergent pattern — **{best['title']}** — is "
            f"worth considering as a real governance principle for platforms "
            f"built on recursive self-modeling."
        )

    if meta.get("total_events", 0) > 0:
        lines.append(
            f"\nMeta-awareness emerged at year {meta.get('first_year', '?')}, "
            f"confirming that recursive simulations naturally produce "
            f"self-referential knowledge. This is the Turtles All the Way Down "
            f"doctrine validated experimentally."
        )

    return "\n".join(lines)


def generate_governance_report(
    analysis: dict[str, Any],
    sim_summary: dict[str, Any] | None = None,
) -> str:
    """Generate the full Markdown governance report.

    Args:
        analysis: Output of ``full_analysis(sim_dict)``.
        sim_summary: Optional ``sim_dict["summary"]`` for population stats.

    Returns:
        A Markdown string.
    """
    lines: list[str] = []

    lines.append("# Emergent Governance Patterns from Mars-100\n")
    lines.append(
        "*A 100-Martian-year recursive colony simulation with 10 founding colonists, "
        "sub-simulations, and emergent governance.*\n"
    )

    # --- Executive Summary ---
    lines.append("## Executive Summary\n")
    fitness = analysis.get("fitness", {})
    composite = fitness.get("composite", 0.0)

    if sim_summary:
        lines.append(
            f"Over {sim_summary.get('total_years', 100)} Martian years, the colony "
            f"experienced {sim_summary.get('total_deaths', 0)} deaths, "
            f"{sim_summary.get('total_births', 0)} births, and "
            f"{sim_summary.get('governance_changes', 0)} governance transitions. "
            f"Colonists spawned {sim_summary.get('total_subsims', 0)} sub-simulations "
            f"to model proposals before committing.\n"
        )

    lines.append(f"**Colony fitness score: {composite:.2f}** (0–1 scale)\n")

    for component in ("survival_rate", "resource_health", "social_cohesion",
                       "governance_stability", "cultural_richness"):
        val = fitness.get(component, 0.0)
        lines.append(f"- {component.replace('_', ' ').title()}: {val:.2f}")
    lines.append("")

    # --- Value Convergence ---
    lines.append("## Value Convergence\n")
    conv = analysis.get("value_convergence", {})
    lines.append(_interpret_convergence(conv))
    lines.append("")

    # --- Governance Stability ---
    lines.append("## Governance Stability\n")
    gov = analysis.get("governance_stability", {})
    periods = gov.get("periods", [])
    attractor = gov.get("attractor", "none")
    transitions = gov.get("transitions", 0)

    lines.append(
        f"The colony went through **{transitions} governance transitions** "
        f"across {len(periods)} distinct periods. The dominant governance type "
        f"was **{attractor}**.\n"
    )

    if periods:
        lines.append("| Period | Type | Years | Duration |")
        lines.append("|--------|------|-------|----------|")
        for p in periods:
            lines.append(
                f"| {p['start_year']}–{p['end_year']} | {p['type']} | "
                f"{p['start_year']}–{p['end_year']} | {p['duration']} years |"
            )
        lines.append("")

    lp = gov.get("longest_period")
    if lp:
        lines.append(
            f"The longest stable period was **{lp['type']}** "
            f"({lp['duration']} years, years {lp['start_year']}–{lp['end_year']}).\n"
        )

    # --- Sub-Simulation Effectiveness ---
    lines.append("## Sub-Simulation Effectiveness\n")
    subsim = analysis.get("subsim_effectiveness", {})
    lines.append(
        f"Total sub-simulations run: **{subsim.get('total_subsims', 0)}**\n"
    )
    lines.append(
        f"- Subsim-backed proposals: {subsim.get('backed_proposals', 0)} "
        f"(pass rate: {subsim.get('backed_pass_rate', 0):.0%})"
    )
    lines.append(
        f"- Unbacked proposals: {subsim.get('unbacked_proposals', 0)} "
        f"(pass rate: {subsim.get('unbacked_pass_rate', 0):.0%})"
    )
    delta = subsim.get("effectiveness_delta", 0.0)
    lines.append(f"- **Effectiveness delta: {delta:+.0%}**\n")

    depth_counts = subsim.get("depth_counts", {})
    if depth_counts:
        lines.append("Depth distribution:")
        for depth in sorted(depth_counts):
            lines.append(f"- Depth {depth}: {depth_counts[depth]} sub-sims")
        lines.append("")

    max_depth = subsim.get("max_depth_reached", 0)
    if max_depth >= 3:
        lines.append(
            "**Depth-3 sub-simulations were observed** — colonists recursed "
            "to the constitutional maximum depth, modeling simulations within "
            "simulations within simulations. Turtles all the way down.\n"
        )
    elif max_depth == 2:
        lines.append(
            "Sub-simulations reached depth 2 — colonists modeled scenarios "
            "that themselves spawned further simulations.\n"
        )

    # --- Meta-Awareness ---
    lines.append("## Meta-Awareness Events\n")
    meta = analysis.get("meta_emergence", {})
    total_meta = meta.get("total_events", 0)

    if total_meta > 0:
        lines.append(
            f"**{total_meta} meta-awareness events** were recorded, first "
            f"appearing at year {meta.get('first_year', '?')}. "
            f"{meta.get('unique_colonists_aware', 0)} unique colonists "
            f"developed awareness of their simulated nature.\n"
        )
    else:
        lines.append("No meta-awareness events were observed in this run.\n")

    # --- Amendment Proposal ---
    lines.append("## Proposed Constitutional Amendment\n")
    amendment = analysis.get("amendment_proposal", {})

    if amendment.get("proposed"):
        best = amendment["amendment"]
        lines.append(f"### {best['title']}\n")
        lines.append(f"> {best['text']}\n")
        lines.append(f"**Rationale:** {best['rationale']}\n")
        lines.append(
            f"**Strength:** {best['strength']:.2f} | "
            f"**Evidence:** {best['evidence']}\n"
        )

        all_candidates = amendment.get("all_candidates", [])
        if len(all_candidates) > 1:
            lines.append("### Other Candidates\n")
            for c in all_candidates:
                if c["id"] != best["id"]:
                    lines.append(
                        f"- **{c['title']}** (strength: {c['strength']:.2f}) — "
                        f"{c['rationale']}"
                    )
            lines.append("")
    else:
        reason = amendment.get("reason", "No strong patterns emerged.")
        lines.append(f"*{reason}*\n")

    # --- Conclusion ---
    lines.append("## Conclusion\n")
    lines.append(_generate_conclusion(analysis))
    lines.append("")

    return "\n".join(lines)
