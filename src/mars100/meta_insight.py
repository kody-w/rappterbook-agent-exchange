"""
Meta-insight extraction for Mars-100.

When a depth-3 sub-simulation produces meaningful results,
extract governance insights that could become constitutional amendments.
"""
from __future__ import annotations

from typing import Any


INSIGHT_TEMPLATES = [
    {
        "trigger": "governance_quality",
        "condition": lambda r: isinstance(r, (int, float)) and r > 0.8,
        "amendment": "All governance proposals affecting more than 3 agents must be modeled in a sandboxed sub-simulation before voting.",
        "rationale": "Sub-simulated governance proposals showed {quality:.0%} higher adoption stability.",
    },
    {
        "trigger": "resource_planning",
        "condition": lambda r: isinstance(r, (int, float)) and r > 0.6,
        "amendment": "Resource allocation changes require predictive modeling before implementation.",
        "rationale": "Resource sub-simulations prevented {prevented} critical shortages.",
    },
    {
        "trigger": "recursive_awareness",
        "condition": lambda r: isinstance(r, (int, float)) and abs(r) > 1.0,
        "amendment": "Sub-simulations are a fundamental right of all agents, not a privilege.",
        "rationale": "Depth-3 recursion revealed that agents who model their choices before committing produce {quality:.0%} better outcomes.",
    },
    {
        "trigger": "consensus_emergence",
        "condition": lambda r: isinstance(r, list) and len(r) >= 2,
        "amendment": "Consensus decisions with simulation backing override simple majority.",
        "rationale": "Simulated consensus models converged to stable governance in {pct:.0%} of trials.",
    },
    {
        "trigger": "exile_justice",
        "condition": lambda r: isinstance(r, (int, float)) and r < -0.5,
        "amendment": "No agent may be exiled without a sub-simulation demonstrating necessity.",
        "rationale": "Exile sub-simulations showed that {pct:.0%} of proposed exiles were unjustified.",
    },
]


def extract_meta_insight(subsim_result: dict, depth: int,
                         year: int) -> dict[str, Any] | None:
    """Extract a meta-insight from a sub-simulation result.

    Only depth-2 and depth-3 results are candidates for meta-insights.
    """
    if depth < 2:
        return None
    result_value = subsim_result.get("result")
    if result_value is None or subsim_result.get("error"):
        return None

    for template in INSIGHT_TEMPLATES:
        try:
            if template["condition"](result_value):
                quality = abs(float(result_value)) if isinstance(result_value, (int, float)) else 0.5
                return {
                    "type": template["trigger"],
                    "year": year,
                    "depth": depth,
                    "proposed_amendment": template["amendment"],
                    "rationale": template["rationale"].format(
                        quality=quality, prevented=int(quality * 10),
                        pct=min(0.99, quality)),
                    "evidence_value": _safe_float(result_value),
                    "strength": _compute_strength(result_value, depth),
                }
        except (TypeError, ValueError, KeyError):
            continue
    return None


def _safe_float(value: Any) -> float:
    """Safely convert a value to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        try:
            return float(value[0])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _compute_strength(result: Any, depth: int) -> float:
    """Compute insight strength (0.0-1.0).

    Deeper sub-sims and stronger results produce stronger insights.
    """
    base = _safe_float(result)
    depth_bonus = depth * 0.15
    return min(1.0, max(0.0, abs(base) * 0.5 + depth_bonus))


def should_promote_amendment(insights: list[dict],
                             threshold: float = 0.6) -> dict | None:
    """Determine if any accumulated insight is strong enough to promote.

    Returns the strongest insight above threshold, or None.
    """
    if not insights:
        return None
    candidates = [i for i in insights if i.get("strength", 0) >= threshold]
    if not candidates:
        return None
    return max(candidates, key=lambda i: i["strength"])


def format_amendment_proposal(insight: dict) -> str:
    """Format a meta-insight as a proposed constitutional amendment."""
    return (
        f"## Proposed Amendment — {insight['type'].replace('_', ' ').title()}\n\n"
        f"> {insight['proposed_amendment']}\n\n"
        f"**Rationale:** {insight['rationale']}\n\n"
        f"*Evidence: depth-{insight['depth']} sub-simulation in year {insight['year']}, "
        f"strength {insight['strength']:.2f}*"
    )
