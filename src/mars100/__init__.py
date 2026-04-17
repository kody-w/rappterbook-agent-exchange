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
from src.mars100.economy import (
    ColonyEconomy, EconomySnapshot, Wallet,
    compute_income, process_theft, collect_taxes,
    process_inheritance, spend_treasury, compute_gini, tick_economy,
)
from src.mars100.culture import (
    OralHistory, Tradition,
    tradition_from_death, tradition_from_governance,
    tradition_from_subsim, tradition_from_crisis, tradition_from_meta,
    CATEGORY_ACTION_BIAS,
)
from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim
from src.mars100.lispy_vm import LispyError, LispyRuntimeError, run as lispy_run, make_env
from src.mars100.narrator import narrate_year, generate_diary_entries, generate_final_report

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
    "ColonyEconomy", "EconomySnapshot", "Wallet",
    "compute_income", "process_theft", "collect_taxes",
    "process_inheritance", "spend_treasury", "compute_gini", "tick_economy",
    "OralHistory", "Tradition",
    "tradition_from_death", "tradition_from_governance",
    "tradition_from_subsim", "tradition_from_crisis", "tradition_from_meta",
    "CATEGORY_ACTION_BIAS",
    "SubSimBudget", "SubSimResult", "spawn_subsim",
    "LispyError", "LispyRuntimeError", "lispy_run", "make_env",
    "narrate_year", "generate_diary_entries", "generate_final_report",
]
