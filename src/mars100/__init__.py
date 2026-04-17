"""Mars-100: A 100-Martian-year recursive colony simulation with 10 agent-colonists."""
from __future__ import annotations

from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, MemoryEntry,
    ELEMENTS, STAT_NAMES, SKILL_NAMES, COLONIST_NAMES,
    create_founding_ten, create_child,
)
from src.mars100.colony import (
    Resources, SocialGraph, Relationship, RESOURCE_NAMES,
    compute_value_convergence, tick_resources,
)
from src.mars100.engine import Mars100Engine, YearResult, SimulationResult
from src.mars100.events import Event, generate_events, EVENT_TEMPLATES
from src.mars100.governance import (
    GovernanceProposal, GovernanceState,
    generate_proposal, resolve_vote, should_propose, apply_governance,
)
from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim
from src.mars100.lispy_vm import LispyError, LispyRuntimeError, run as lispy_run, make_env
from src.mars100.narrator import narrate_year, generate_diary_entries, generate_final_report
from src.mars100.crossover import (
    crossover_analysis,
    analyze_governance_transitions,
    detect_behavioral_convergence,
    score_transferability,
    analyze_value_trends,
    analyze_subsim_effectiveness,
    generate_rappterbook_amendment,
    GovernancePattern,
    BehavioralConvergence,
    TransferabilityScore,
    AmendmentProposal,
)

__all__ = [
    "Mars100Engine", "YearResult", "SimulationResult",
    "Colonist", "ColonistStats", "ColonistSkills", "MemoryEntry",
    "ELEMENTS", "STAT_NAMES", "SKILL_NAMES", "COLONIST_NAMES",
    "create_founding_ten", "create_child",
    "Resources", "SocialGraph", "Relationship", "RESOURCE_NAMES",
    "compute_value_convergence", "tick_resources",
    "Event", "generate_events", "EVENT_TEMPLATES",
    "GovernanceProposal", "GovernanceState",
    "generate_proposal", "resolve_vote", "should_propose", "apply_governance",
    "SubSimBudget", "SubSimResult", "spawn_subsim",
    "LispyError", "LispyRuntimeError", "lispy_run", "make_env",
    "narrate_year", "generate_diary_entries", "generate_final_report",
    "crossover_analysis",
    "analyze_governance_transitions", "detect_behavioral_convergence",
    "score_transferability", "analyze_value_trends",
    "analyze_subsim_effectiveness", "generate_rappterbook_amendment",
    "GovernancePattern", "BehavioralConvergence",
    "TransferabilityScore", "AmendmentProposal",
]


def run_simulation(years: int = 100, seed: int = 42) -> dict:
    """Compatibility shim — runs the engine and returns legacy dict format.

    This bridges the package API (Mars100Engine) with the legacy dict format
    expected by the runner script (src/mars100_engine.py).
    """
    engine = Mars100Engine(seed=seed, total_years=years)
    result = engine.run()
    rd = result.to_dict()

    # Build legacy-compatible deltas from year results
    deltas: list[dict] = []
    for yr in result.years:
        delta: dict = {
            "year": yr.year,
            "population": len([c for c in yr.colonist_snapshots
                               if c.get("alive") and not c.get("exiled")]),
            "event": yr.events[0] if yr.events else None,
            "event_effects": [f"{e['name']}: {e['description'][:60]}" for e in yr.events],
            "colonist_actions": [
                {"colonist": cid, "action": f"{cid} works on {act}"}
                for cid, act in yr.actions.items()
            ],
            "sub_sims": yr.subsim_log,
            "governance_results": (
                [f"{yr.governance['gov_type']}:{'PASSED' if yr.governance.get('passed') else 'REJECTED'}"]
                if yr.governance else []
            ),
            "births": yr.births,
            "diary_entries": [],
            "meta_awareness": (yr.meta_awareness[0]["insight"]
                               if yr.meta_awareness else ""),
            "resources_snapshot": yr.resources_after,
        }
        deltas.append(delta)

    # Build legacy colony dict
    final_alive = [c for c in rd.get("final_colonists", [])
                   if c.get("alive") and not c.get("exiled")]
    dead_souls = [c for c in rd.get("final_colonists", [])
                  if not c.get("alive")]

    colony = {
        "colonists": final_alive,
        "dead_souls": dead_souls,
        "resources": rd.get("final_resources", {}),
        "governance": rd.get("final_governance", {}),
    }

    summary = rd.get("summary", {})

    return {
        "colony": colony,
        "deltas": deltas,
        "summary": summary,
        "full_result": rd,
    }
