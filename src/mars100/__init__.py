"""Mars-100: A 100-Martian-year recursive colony simulation with 10 agent-colonists."""
from __future__ import annotations

from src.mars100.engine import Mars100Engine, YearResult, SimulationResult, ENGINE_VERSION
from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, create_founding_ten,
    ELEMENTS, STAT_NAMES, SKILL_NAMES, MemoryEntry,
)
from src.mars100.colony import Resources, SocialGraph, tick_resources, RESOURCE_NAMES
from src.mars100.events import Event, generate_events
from src.mars100.governance import (
    GovernanceProposal, GovernanceState, apply_governance,
    generate_proposal, resolve_vote, should_propose, GOVERNANCE_TYPES,
)
from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim
from src.mars100.lispy_vm import run as lispy_run, LispyError, LispyBudgetExceeded
from src.mars100.births import maybe_birth, reset_birth_counter
from src.mars100.convergence import (
    compute_convergence_score, convergence_trend, convergence_summary,
    per_stat_convergence,
)
from src.mars100.meta_insight import (
    extract_meta_insight, should_promote_amendment, format_amendment_proposal,
)
from src.mars100.narrator import narrate_year, generate_diary_entries, generate_final_report


def run_simulation(years: int = 100, seed: int = 42) -> SimulationResult:
    """Run the full Mars-100 simulation and return results."""
    engine = Mars100Engine(seed=seed, total_years=years)
    return engine.run()


__all__ = [
    "Mars100Engine", "YearResult", "SimulationResult", "ENGINE_VERSION",
    "Colonist", "ColonistStats", "ColonistSkills", "create_founding_ten",
    "ELEMENTS", "STAT_NAMES", "SKILL_NAMES", "MemoryEntry",
    "Resources", "SocialGraph", "tick_resources", "RESOURCE_NAMES",
    "Event", "generate_events",
    "GovernanceProposal", "GovernanceState", "apply_governance",
    "generate_proposal", "resolve_vote", "should_propose", "GOVERNANCE_TYPES",
    "SubSimBudget", "SubSimResult", "spawn_subsim",
    "lispy_run", "LispyError", "LispyBudgetExceeded",
    "maybe_birth", "reset_birth_counter",
    "compute_convergence_score", "convergence_trend", "convergence_summary",
    "per_stat_convergence",
    "extract_meta_insight", "should_promote_amendment", "format_amendment_proposal",
    "narrate_year", "generate_diary_entries", "generate_final_report",
    "run_simulation",
]
