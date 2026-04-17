"""Mars-100: A 100-Martian-year recursive colony simulation with 10 agent-colonists.

This package provides both the structured engine (mars100.engine, mars100.colonist, etc.)
and a compatibility facade re-exporting the standalone simulation API from mars100_sim.
"""
from __future__ import annotations

# New structured engine API
from src.mars100.engine import Mars100Engine, SimulationResult, YearResult  # noqa: F401
from src.mars100.colonist import Colonist, create_founding_ten  # noqa: F401
from src.mars100.analysis import full_analysis  # noqa: F401

# Re-export the standalone simulation API for backward compatibility.
# Tests and external consumers import from src.mars100 and expect these names.
from src.mars100_sim import (  # noqa: F401
    COLONIST_NAMES, ELEMENTS, EVENTS, GOVERNANCE_TYPES,
    INITIAL_RESOURCES, SKILL_NAMES, STAT_NAMES,
    apply_event_effects, check_births, check_meta_awareness,
    colonist_to_env, consume_resources, create_colony,
    create_colonist, evolve_relationships, generate_action_expr,
    generate_subsim_expr, init_relationships, pick_event,
    process_action, resolve_proposals, run_simulation, tick_year,
)
