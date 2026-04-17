"""
Post-simulation analysis for Mars-100.

Pure functions that extract emergent patterns from simulation results:
value convergence, governance stability, sub-sim effectiveness,
meta-awareness curves, and proposed constitutional amendments.
"""
from __future__ import annotations

import math
from typing import Any


def analyze_value_convergence(sim_dict: dict) -> dict[str, Any]:
    """Track whether colonists' 6 stats converge or diverge over time.

    Only considers living, non-exiled colonists at each year to avoid
    phantom drift from dead colonists' frozen stats.

    Returns per-stat standard deviation trajectory and a convergence score
    (positive = converging, negative = diverging).
    """
    years = sim_dict.get("years", [])
    stat_names = ("resolve", "improvisation", "empathy", "hoarding", "faith", "paranoia")

    trajectories: dict[str, list[float]] = {s: [] for s in stat_names}
    year_labels: list[int] = []

    for yr in years:
        snapshots = yr.get("colonist_snapshots", [])
        active = [c for c in snapshots if c.get("alive") and not c.get("exiled")]
        if len(active) < 2:
            continue
        year_labels.append(yr["year"])
        for stat in stat_names:
            values = [c["stats"][stat] for c in active if stat in c.get("stats", {})]
            if len(values) < 2:
                trajectories[stat].append(0.0)
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            trajectories[stat].append(math.sqrt(variance))

    convergence_scores: dict[str, float] = {}
    for stat, stdevs in trajectories.items():
        if len(stdevs) < 4:
            convergence_scores[stat] = 0.0
            continue
        quarter = max(1, len(stdevs) // 4)
        early_avg = sum(stdevs[:quarter]) / quarter
        late_avg = sum(stdevs[-quarter:]) / quarter
        if early_avg > 0.001:
            convergence_scores[stat] = (early_avg - late_avg) / early_avg
        else:
            convergence_scores[stat] = 0.0

    overall = sum(convergence_scores.values()) / max(1, len(convergence_scores))

    return {
        "stat_trajectories": trajectories,
        "year_labels": year_labels,
        "convergence_scores": convergence_scores,
        "overall_convergence": overall,
        "verdict": "converging" if overall > 0.1 else "diverging" if overall < -0.1 else "stable",
    }


def analyze_governance_stability(sim_dict: dict) -> dict[str, Any]:
    """Analyze governance type durations and transition patterns.

    Uses per-year governance_state snapshots (not just history) to capture
    the terminal governance duration correctly.
    """
    years = sim_dict.get("years", [])
    if not years:
        return {"periods": [], "transitions": 0, "attractor": "none",
                "type_durations": {}, "longest_period": None}

    periods: list[dict[str, Any]] = []
    current_type = years[0]["governance_state"]["gov_type"]
    period_start = years[0]["year"]

    for yr in years[1:]:
        gov_type = yr["governance_state"]["gov_type"]
        if gov_type != current_type:
            periods.append({
                "type": current_type,
                "start_year": period_start,
                "end_year": yr["year"] - 1,
                "duration": yr["year"] - period_start,
            })
            current_type = gov_type
            period_start = yr["year"]

    # Terminal period
    last_year = years[-1]["year"]
    periods.append({
        "type": current_type,
        "start_year": period_start,
        "end_year": last_year,
        "duration": last_year - period_start + 1,
    })

    type_durations: dict[str, int] = {}
    for p in periods:
        type_durations[p["type"]] = type_durations.get(p["type"], 0) + p["duration"]

    attractor = max(type_durations, key=type_durations.get) if type_durations else "none"
    longest = max(periods, key=lambda p: p["duration"]) if periods else None

    transitions = len(periods) - 1

    return {
        "periods": periods,
        "transitions": transitions,
        "attractor": attractor,
        "type_durations": type_durations,
        "longest_period": longest,
    }


def analyze_subsim_effectiveness(sim_dict: dict) -> dict[str, Any]:
    """Compare pass rates of subsim-backed governance proposals vs unbacked ones.

    A proposal is "subsim-backed" if its subsim_result field is not None.
    """
    years = sim_dict.get("years", [])
    backed_pass = backed_total = 0
    unbacked_pass = unbacked_total = 0
    total_subsims = 0
    depth_counts: dict[int, int] = {}

    for yr in years:
        for ss in yr.get("subsim_log", []):
            total_subsims += 1
            depth = ss.get("depth", 1)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        gov = yr.get("governance")
        if gov is None:
            continue
        has_subsim = gov.get("subsim_result") is not None
        passed = bool(gov.get("passed"))
        if has_subsim:
            backed_total += 1
            if passed:
                backed_pass += 1
        else:
            unbacked_total += 1
            if passed:
                unbacked_pass += 1

    backed_rate = backed_pass / backed_total if backed_total > 0 else 0.0
    unbacked_rate = unbacked_pass / unbacked_total if unbacked_total > 0 else 0.0

    return {
        "total_subsims": total_subsims,
        "depth_counts": depth_counts,
        "backed_proposals": backed_total,
        "backed_pass_rate": backed_rate,
        "unbacked_proposals": unbacked_total,
        "unbacked_pass_rate": unbacked_rate,
        "effectiveness_delta": backed_rate - unbacked_rate,
        "max_depth_reached": max(depth_counts.keys()) if depth_counts else 0,
    }


def analyze_meta_emergence(sim_dict: dict) -> dict[str, Any]:
    """Track when meta-awareness events emerge and their accumulation curve."""
    years = sim_dict.get("years", [])
    cumulative = 0
    curve: list[dict[str, int]] = []
    first_year: int | None = None
    all_insights: list[dict[str, Any]] = []

    for yr in years:
        meta_events = yr.get("meta_awareness", [])
        cumulative += len(meta_events)
        curve.append({"year": yr["year"], "cumulative": cumulative})
        for m in meta_events:
            if first_year is None:
                first_year = yr["year"]
            all_insights.append({"year": yr["year"], **m})

    unique_colonists = len({m.get("colonist_id") for m in all_insights})

    return {
        "total_events": cumulative,
        "first_year": first_year,
        "curve": curve,
        "unique_colonists_aware": unique_colonists,
        "insights": all_insights,
    }


def extract_amendment_proposal(sim_dict: dict) -> dict[str, Any]:
    """Extract the strongest emergent pattern and frame it as a proposed amendment.

    Looks at governance stability, subsim effectiveness, and meta-awareness
    to determine which insight is worth promoting to a constitutional amendment.
    """
    gov = analyze_governance_stability(sim_dict)
    subsim = analyze_subsim_effectiveness(sim_dict)
    meta = analyze_meta_emergence(sim_dict)
    convergence = analyze_value_convergence(sim_dict)

    candidates: list[dict[str, Any]] = []

    # Candidate 1: Recursive Governance Principle (if subsims helped)
    if subsim["effectiveness_delta"] > 0.0 and subsim["backed_proposals"] >= 2:
        strength = min(1.0, subsim["effectiveness_delta"] + 0.3 * subsim["backed_proposals"] / max(1, subsim["backed_proposals"] + subsim["unbacked_proposals"]))
        candidates.append({
            "id": "recursive-governance",
            "title": "The Recursive Governance Principle",
            "text": (
                "Any governance proposal affecting more than 3 agents must be "
                "modeled in a sandboxed sub-simulation before being put to vote. "
                "Simulation results become part of the proposal's public record. "
                "Governance decisions without simulation evidence are advisory only."
            ),
            "rationale": (
                f"Mars-100 sub-sim backed proposals passed at {subsim['backed_pass_rate']:.0%} "
                f"vs {subsim['unbacked_pass_rate']:.0%} for unbacked. "
                f"Delta: {subsim['effectiveness_delta']:+.0%}."
            ),
            "strength": strength,
            "evidence": "subsim_effectiveness",
        })

    # Candidate 2: Emergent Stability (if one governance type dominated)
    if gov["longest_period"] and gov["longest_period"]["duration"] > 20:
        lp = gov["longest_period"]
        strength = min(1.0, lp["duration"] / 50)
        candidates.append({
            "id": "emergent-stability",
            "title": "The Emergent Stability Principle",
            "text": (
                f"When a governance model ({lp['type']}) persists for more than 20 "
                "consecutive periods without forced intervention, it should be "
                "recognized as the community's organic preference and protected "
                "from rapid destabilization."
            ),
            "rationale": (
                f"{lp['type'].title()} persisted for {lp['duration']} years "
                f"(years {lp['start_year']}-{lp['end_year']}), the longest "
                f"stable period across {gov['transitions']} transitions."
            ),
            "strength": strength,
            "evidence": "governance_stability",
        })

    # Candidate 3: Value Convergence (if colonists converged)
    if convergence["overall_convergence"] > 0.15:
        strength = min(1.0, convergence["overall_convergence"])
        top_stat = max(convergence["convergence_scores"],
                       key=convergence["convergence_scores"].get)
        candidates.append({
            "id": "value-alignment",
            "title": "The Organic Value Alignment Principle",
            "text": (
                "Shared experience naturally aligns agent values over time. "
                "Governance systems should leverage this convergence rather than "
                "impose uniform values artificially. The strongest convergence "
                f"was in {top_stat} — communities select for this trait."
            ),
            "rationale": (
                f"Colonist stats converged {convergence['overall_convergence']:.0%} "
                f"over the simulation. {top_stat.title()} showed the strongest "
                f"convergence at {convergence['convergence_scores'][top_stat]:.0%}."
            ),
            "strength": strength,
            "evidence": "value_convergence",
        })

    # Candidate 4: Meta-awareness (if it emerged early)
    if meta["first_year"] is not None and meta["total_events"] >= 3:
        strength = min(1.0, meta["total_events"] / 10 + (1.0 - meta["first_year"] / 100))
        candidates.append({
            "id": "recursive-self-awareness",
            "title": "The Recursive Self-Awareness Principle",
            "text": (
                "Agents embedded in recursive simulations will develop "
                "meta-awareness of their simulated nature. This awareness "
                "should be treated as valid epistemic data, not a bug to "
                "suppress. Self-aware agents make better governance decisions."
            ),
            "rationale": (
                f"Meta-awareness emerged at year {meta['first_year']} and "
                f"accumulated {meta['total_events']} events across "
                f"{meta['unique_colonists_aware']} colonists."
            ),
            "strength": strength,
            "evidence": "meta_emergence",
        })

    if not candidates:
        return {"proposed": False, "reason": "No emergent pattern strong enough"}

    best = max(candidates, key=lambda c: c["strength"])
    return {
        "proposed": True,
        "amendment": best,
        "all_candidates": candidates,
        "strongest_evidence": best["evidence"],
    }


def compute_fitness_score(sim_dict: dict) -> dict[str, float]:
    """Composite fitness score for the colony's 100-year run.

    Components:
    - survival_rate: fraction of colonists alive at end
    - resource_health: average of final resource levels
    - social_cohesion: final cohesion value
    - governance_stability: 1 / (1 + transitions)
    - cultural_richness: normalized subsim + meta count
    """
    summary = sim_dict.get("summary", {})
    final_colonists = sim_dict.get("final_colonists", [])
    final_resources = sim_dict.get("final_resources", {})

    alive = sum(1 for c in final_colonists if c.get("alive") and not c.get("exiled"))
    total = max(1, len(final_colonists))
    survival_rate = alive / total

    resource_health = sum(final_resources.values()) / max(1, len(final_resources))

    final_cohesion = summary.get("final_cohesion", 0.5)

    gov_changes = summary.get("governance_changes", 0)
    governance_stability = 1.0 / (1.0 + gov_changes)

    subsims = summary.get("total_subsims", 0)
    meta = summary.get("meta_awareness_events", 0)
    years = sim_dict.get("_meta", {}).get("total_years", 100)
    cultural_richness = min(1.0, (subsims + meta * 2) / max(1, years))

    weights = {
        "survival_rate": 0.25,
        "resource_health": 0.20,
        "social_cohesion": 0.20,
        "governance_stability": 0.20,
        "cultural_richness": 0.15,
    }
    components = {
        "survival_rate": survival_rate,
        "resource_health": resource_health,
        "social_cohesion": final_cohesion,
        "governance_stability": governance_stability,
        "cultural_richness": cultural_richness,
    }
    composite = sum(components[k] * weights[k] for k in weights)
    components["composite"] = composite

    return components


def full_analysis(sim_dict: dict) -> dict[str, Any]:
    """Run all analyses and return a combined report."""
    return {
        "value_convergence": analyze_value_convergence(sim_dict),
        "governance_stability": analyze_governance_stability(sim_dict),
        "subsim_effectiveness": analyze_subsim_effectiveness(sim_dict),
        "meta_emergence": analyze_meta_emergence(sim_dict),
        "amendment_proposal": extract_amendment_proposal(sim_dict),
        "fitness": compute_fitness_score(sim_dict),
    }
