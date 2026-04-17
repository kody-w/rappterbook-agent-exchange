"""Mars-100: A 100-Martian-year recursive colony simulation with 10 agent-colonists."""
from __future__ import annotations

# Re-export the flat dictionary-based simulation API.
# The canonical tested interface lives in sim.py (formerly src/mars100.py).
from src.mars100.sim import (  # noqa: F401
    COLONIST_NAMES,
    ELEMENTS,
    EVENTS,
    GOVERNANCE_TYPES,
    INITIAL_RESOURCES,
    SKILL_NAMES,
    STAT_NAMES,
    apply_event_effects,
    check_births,
    check_meta_awareness,
    colonist_to_env,
    compute_value_convergence,
    consume_resources,
    create_colony,
    create_colonist,
    evolve_relationships,
    generate_action_expr,
    generate_diary_entries,
    generate_subsim_expr,
    init_relationships,
    pick_event,
    process_action,
    resolve_proposals,
    run_simulation,
    spawn_deep_subsim,
    tick_year,
)
