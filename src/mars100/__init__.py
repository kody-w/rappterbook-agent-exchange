"""Mars-100: A 100-Martian-year recursive colony simulation with 10 agent-colonists."""
from __future__ import annotations

from src.mars100.colonist import (
    Colonist, ColonistStats, ColonistSkills, MemoryEntry, Wallet,
    ELEMENTS, STAT_NAMES, SKILL_NAMES, COLONIST_NAMES,
    create_founding_ten, create_child, create_immigrant,
    IMMIGRANT_ARCHETYPES,
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
from src.mars100.culture import (
    CulturalMemory, YearContext as CultureYearContext,
    evolve_culture, compute_cultural_pressure, transmit_to_child,
)
from src.mars100.earth import (
    EarthState, EarthMessage, EarthTickResult, SupplyShip,
    tick_earth, compute_maintenance_modifier,
    check_independence_conditions, declare_independence,
)
from src.mars100.economics import (
    EconomicState, EconomicTickResult, TradeRecord,
    allocate_labor_income, find_trades, apply_taxation,
    compute_gini, compute_economic_pressure, liquidate_estate,
    endow_immigrant, tick_economics,
)
from src.mars100.psychology import (
    PsychState, CrisisEvent, PsychTickResult,
    ColonistPsychContext, tick_psychology,
    death_rate_modifier, compute_colony_morale,
    compute_bottom_quartile_morale,
)
from src.mars100.behavior import (
    BehaviorProfile, BehaviorTickResult, ContagionDelta,
    compute_action_perturbation, compute_social_contagion,
    update_learned_preferences, compute_risk_tolerance,
)
from src.mars100.diplomacy import (
    DiplomacyState, DiplomacyTickResult,
    Faction, Alliance,
    classify_ideology, compute_bloc_pressure, compute_faction_vote_bias,
    tick_diplomacy,
    IDEOLOGY_NAMES,
)
from src.mars100.comm_channels import (
    Channel as CommChannel, CommChannelsState, CommChannelsTickResult,
    RevivalPrompt as CommRevivalPrompt,
    BridgePrompt as CommBridgePrompt,
    tick_comm_channels, compute_vitality, classify_status,
    infer_contacts, generate_revival_prompt, compute_revival_pressure,
    compute_colony_comm_health, pair_key,
    find_bridge_builder, generate_bridge_prompt,
    compute_bridge_pressure, compute_bridge_efficacy_rate,
    FLATLINE_SILENCE_YEARS, FADING_SILENCE_YEARS,
    MAX_REVIVAL_PROMPTS_PER_TICK, MAX_BRIDGE_PROMPTS_PER_TICK,
    BRIDGE_TRUST_FLOOR, BRIDGE_COOLDOWN_YEARS,
    STATUS_VITAL as CHANNEL_STATUS_VITAL,
    STATUS_FADING as CHANNEL_STATUS_FADING,
    STATUS_FLATLINED as CHANNEL_STATUS_FLATLINED,
    STATUS_REVIVED as CHANNEL_STATUS_REVIVED,
    STATUS_DORMANT as CHANNEL_STATUS_DORMANT,
    STATUS_INACTIVE as CHANNEL_STATUS_INACTIVE,
)
from src.mars100.ecology import (
    EcologyState, EcologyYearContext, EcologyTickResult,
    tick_ecology, compute_biosphere_index,
    compute_ecology_bonuses, compute_resource_modifiers as compute_ecology_modifiers,
    compute_nature_stress_reduction, update_biome_level,
    outdoor_habitable, has_greenhouse_tech,
    BIOME_NAMES,
)

__all__ = [
    "Mars100Engine", "YearResult", "SimulationResult",
    "Colonist", "ColonistStats", "ColonistSkills", "MemoryEntry", "Wallet",
    "ELEMENTS", "STAT_NAMES", "SKILL_NAMES", "COLONIST_NAMES",
    "create_founding_ten", "create_child", "create_immigrant",
    "IMMIGRANT_ARCHETYPES",
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
    "EarthState", "EarthMessage", "EarthTickResult", "SupplyShip",
    "tick_earth", "compute_maintenance_modifier",
    "check_independence_conditions", "declare_independence",
    "EconomicState", "EconomicTickResult", "TradeRecord",
    "allocate_labor_income", "find_trades", "apply_taxation",
    "compute_gini", "compute_economic_pressure", "liquidate_estate",
    "endow_immigrant", "tick_economics",
    "PsychState", "CrisisEvent", "PsychTickResult",
    "ColonistPsychContext", "tick_psychology",
    "death_rate_modifier", "compute_colony_morale",
    "compute_bottom_quartile_morale",
    "BehaviorProfile", "BehaviorTickResult", "ContagionDelta",
    "compute_action_perturbation", "compute_social_contagion",
    "update_learned_preferences", "compute_risk_tolerance",
    "EcologyState", "EcologyYearContext", "EcologyTickResult",
    "tick_ecology", "compute_biosphere_index",
    "compute_ecology_bonuses", "compute_ecology_modifiers",
    "compute_nature_stress_reduction", "update_biome_level",
    "outdoor_habitable", "has_greenhouse_tech",
    "BIOME_NAMES",
    "DiplomacyState", "DiplomacyTickResult",
    "Faction", "Alliance",
    "classify_ideology", "compute_bloc_pressure", "compute_faction_vote_bias",
    "tick_diplomacy",
    "IDEOLOGY_NAMES",
    "CommChannel", "CommChannelsState", "CommChannelsTickResult",
    "CommRevivalPrompt",
    "tick_comm_channels", "compute_vitality", "classify_status",
    "infer_contacts", "generate_revival_prompt", "compute_revival_pressure",
    "compute_colony_comm_health", "pair_key",
    "FLATLINE_SILENCE_YEARS", "FADING_SILENCE_YEARS",
    "MAX_REVIVAL_PROMPTS_PER_TICK",
    "CHANNEL_STATUS_VITAL", "CHANNEL_STATUS_FADING",
    "CHANNEL_STATUS_FLATLINED", "CHANNEL_STATUS_REVIVED",
    "CHANNEL_STATUS_DORMANT", "CHANNEL_STATUS_INACTIVE",
    "Rumor", "RumorsState", "RumorsTickResult",
    "tick_rumors", "compute_fragmentation", "build_channel_lookup",
    "RUMOR_SEEDS", "MUTATION_SWAPS",
]

from src.mars100.rumors import (
    Rumor, RumorsState, RumorsTickResult,
    tick_rumors, compute_fragmentation, build_channel_lookup,
    RUMOR_SEEDS, MUTATION_SWAPS,
)
