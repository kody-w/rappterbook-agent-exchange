"""Mars-100: A 100-Martian-year recursive colony simulation with 10 agent-colonists."""
from __future__ import annotations

from typing import Any

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
from src.mars100.synthesis import synthesize, SynthesisResult


def run_simulation(years: int = 100, seed: int = 42) -> dict[str, Any]:
    """Compatibility shim: run Mars100Engine and return result in legacy dict format.

    Bridges the new package engine API with the dict format expected by
    mars100_engine.py and the published data.json schema.
    """
    engine = Mars100Engine(total_years=years, seed=seed)
    sim_result = engine.run()

    colonists_dicts = sim_result.final_colonists
    deltas: list[dict] = []
    for yr in sim_result.years:
        d = yr.to_dict()
        deltas.append({
            "year": d["year"],
            "population": len([c for c in d["colonist_snapshots"] if c.get("alive")]),
            "event": d["events"][0] if d["events"] else None,
            "event_effects": [],
            "colonist_actions": [
                {"colonist": cid, "action": f"{cid} {act}"}
                for cid, act in d["actions"].items()
            ],
            "sub_sims": [s for s in d["subsim_log"]],
            "governance_results": [d["governance"]["gov_type"]] if d["governance"] else [],
            "resource_effects": [],
            "births": d.get("births", []),
            "diary_entries": [],
            "meta_awareness": (d["meta_awareness"][0]["insight"]
                               if d["meta_awareness"] else ""),
            "resources_snapshot": d["resources_after"],
        })

    governance = sim_result.final_governance
    return {
        "colony": {
            "year": years,
            "colonists": colonists_dicts,
            "resources": sim_result.final_resources,
            "governance": governance,
            "proposals_pending": [],
            "dead_souls": [c for c in colonists_dicts if not c.get("alive", True)],
            "sub_sim_log": [],
            "event_history": [],
            "_meta": {"engine": "mars-100-package", "version": "2.0"},
        },
        "deltas": deltas,
        "summary": {
            "years_survived": len(sim_result.years),
            "final_population": len([c for c in colonists_dicts if c.get("alive")]),
            "peak_population": max(
                (len([c for c in yr.colonist_snapshots if c.get("alive")])
                 for yr in sim_result.years), default=0),
            "total_births": sim_result.total_births,
            "total_deaths": sim_result.total_deaths,
            "total_sub_simulations": sim_result.total_subsims,
            "total_proposals": sim_result.governance_changes,
            "governance_system": governance.get("gov_type", "unknown"),
            "constitutional_amendments": governance.get("constitution", []),
            "meta_awareness_events": [],
            "population_curve": [
                len([c for c in yr.colonist_snapshots if c.get("alive")])
                for yr in sim_result.years
            ],
            "morale_curve": [],
        },
    }


__all__ = [
    "Mars100Engine", "YearResult", "SimulationResult",
    "run_simulation", "synthesize", "SynthesisResult",
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
]
