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
from src.mars100.codex import Codex, CodexEntry, imprint_child
from src.mars100.chronicle import (
    build_codex as build_chronicle, detect_eras, extract_lessons,
    generate_testimonies, generate_chronicle_html, propose_amendment,
    normalize_year, normalize_colonists,
    Codex as Chronicle, Era, NormalizedYear, NormalizedColonist,
)
from src.mars100.crisis import (
    CrisisEvent, CrisisPattern, ProposedAmendment,
    detect_crises, backfill_deaths, learn_from_crises,
    deep_deliberation, extract_rappterbook_amendment,
)
from src.mars100.pressure import (
    compute_environmental_pressure, compute_social_pressure,
    compute_existential_pressure, update_pressure, apply_pressure_release,
    pressure_action_modifier, pressure_death_modifier,
    pressure_birth_modifier, collective_pressure, PressureSnapshot,
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
    "Codex", "CodexEntry", "imprint_child",
    "CrisisEvent", "CrisisPattern", "ProposedAmendment",
    "detect_crises", "backfill_deaths", "learn_from_crises",
    "deep_deliberation", "extract_rappterbook_amendment",
    "build_chronicle", "detect_eras", "extract_lessons",
    "generate_testimonies", "generate_chronicle_html", "propose_amendment",
    "normalize_year", "normalize_colonists",
    "Chronicle", "Era", "NormalizedYear", "NormalizedColonist",
    "compute_environmental_pressure", "compute_social_pressure",
    "compute_existential_pressure", "update_pressure", "apply_pressure_release",
    "pressure_action_modifier", "pressure_death_modifier",
    "pressure_birth_modifier", "collective_pressure", "PressureSnapshot",
]
