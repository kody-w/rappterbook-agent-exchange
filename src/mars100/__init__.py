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
from src.mars100.infrastructure import (
    TechSpec, TECH_TREE, TECH_BY_ID,
    ActiveProject, InfrastructureState,
    available_techs, can_afford, choose_project, start_project,
    tick_infrastructure, compute_resource_modifiers, compute_operating_costs,
    validate_tech_tree,
)
from src.mars100.diplomacy import (
    Faction, Pact, DiplomacyState, DiplomacyYearResult,
    detect_factions, manage_pacts, compute_action_modifiers,
    tick_diplomacy, check_crisis, STAT_TO_FACTION,
    CRISIS_THRESHOLD,
)
from src.mars100.counterfactual import (
    CounterfactualResult, TimelineDelta,
    branch_engine, collect_checkpoints, apply_intervention,
    run_forward, compare_timelines, run_counterfactual,
    generate_counterfactuals, run_all_counterfactuals,
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
    "TechSpec", "TECH_TREE", "TECH_BY_ID",
    "ActiveProject", "InfrastructureState",
    "available_techs", "can_afford", "choose_project", "start_project",
    "tick_infrastructure", "compute_resource_modifiers", "compute_operating_costs",
    "validate_tech_tree",
    "CounterfactualResult", "TimelineDelta",
    "branch_engine", "collect_checkpoints", "apply_intervention",
    "run_forward", "compare_timelines", "run_counterfactual",
    "generate_counterfactuals", "run_all_counterfactuals",
    "Faction", "Pact", "DiplomacyState", "DiplomacyYearResult",
    "detect_factions", "manage_pacts", "compute_action_modifiers",
    "tick_diplomacy", "check_crisis", "STAT_TO_FACTION",
    "CRISIS_THRESHOLD",
]
