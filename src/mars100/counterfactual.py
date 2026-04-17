"""
Counterfactual engine for Mars-100.

Branches from exact engine snapshots to ask "what if?" questions:
  - What if governance changed earlier?
  - What if a different tech was prioritised?
  - What if a resource shock hit at year N?

Every counterfactual runs a *paired baseline* from the same snapshot
(same RNG state, no intervention) so that deltas isolate the
intervention effect from random drift.

Pure computation — returns data, no I/O.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from src.mars100.engine import Mars100Engine, YearResult
from src.mars100.colony import RESOURCE_NAMES
from src.mars100.infrastructure import TECH_BY_ID
from src.mars100.governance import GovernanceState


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TimelineDelta:
    """Differences between baseline and intervention timelines."""
    population_delta: int = 0
    total_deaths_delta: int = 0
    resource_deltas: dict = field(default_factory=dict)
    governance_changes_delta: int = 0
    cohesion_delta: float = 0.0
    subsim_delta: int = 0
    births_delta: int = 0
    tech_count_delta: int = 0


@dataclass
class CounterfactualResult:
    """Result of one what-if question."""
    question: str
    intervention_type: str
    branch_year: int
    horizon: int
    baseline_summary: dict = field(default_factory=dict)
    intervention_summary: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)
    verdict: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "intervention_type": self.intervention_type,
            "branch_year": self.branch_year,
            "horizon": self.horizon,
            "baseline": self.baseline_summary,
            "intervention": self.intervention_summary,
            "delta": self.delta,
            "verdict": self.verdict,
        }


# ---------------------------------------------------------------------------
# Engine branching
# ---------------------------------------------------------------------------

def branch_engine(engine: Mars100Engine) -> Mars100Engine:
    """Create an exact deep-copy of the engine (preserves RNG state)."""
    return copy.deepcopy(engine)


def collect_checkpoints(
    seed: int = 42,
    total_years: int = 100,
    checkpoint_years: list = None,
) -> dict:
    """Run a full simulation, saving engine snapshots at specified years.

    Returns a dict mapping year -> engine snapshot *before* that year's tick.
    """
    if checkpoint_years is None:
        checkpoint_years = [10, 15, 20, 30, 50]
    cp_set = set(checkpoint_years)
    snapshots: dict = {}

    engine = Mars100Engine(seed=seed, total_years=total_years)
    for _ in range(total_years):
        if not engine._active_colonists():
            break
        if engine.year + 1 in cp_set:
            snapshots[engine.year + 1] = branch_engine(engine)
        engine.tick()
    return snapshots


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------

def apply_intervention(
    engine: Mars100Engine,
    intervention_type: str,
    params: dict,
) -> None:
    """Mutate an engine in-place to apply a hypothetical intervention.

    Supported types:
      force_governance  -- set governance type (params: gov_type)
      force_tech        -- instantly complete a tech (params: tech_id)
      resource_boost    -- add to a resource (params: resource, amount)
    """
    if intervention_type == "force_governance":
        gov_type = params["gov_type"]
        engine.governance.gov_type = gov_type
        engine.governance.leader = None
        engine.governance.council = []
    elif intervention_type == "force_tech":
        tech_id = params["tech_id"]
        if tech_id not in TECH_BY_ID:
            raise ValueError(f"Unknown tech: {tech_id}")
        if tech_id not in engine.infra.completed:
            engine.infra.completed.append(tech_id)
        engine.infra.project = None
    elif intervention_type == "resource_boost":
        resource = params["resource"]
        amount = params.get("amount", 0.2)
        current = getattr(engine.resources, resource, None)
        if current is None:
            raise ValueError(f"Unknown resource: {resource}")
        new_val = min(1.0, max(0.0, current + amount))
        setattr(engine.resources, resource, new_val)
    else:
        raise ValueError(f"Unknown intervention: {intervention_type}")


# ---------------------------------------------------------------------------
# Timeline running & comparison
# ---------------------------------------------------------------------------

def run_forward(engine: Mars100Engine, horizon: int) -> dict:
    """Run an engine forward for *horizon* years and return a summary dict."""
    years = []
    deaths = subsims = births = gov_changes = 0
    for _ in range(horizon):
        if not engine._active_colonists():
            break
        result = engine.tick()
        years.append(result)
        deaths += len(result.deaths)
        subsims += len(result.subsim_log)
        births += len(result.births)
        if result.governance and result.governance.get("passed"):
            gov_changes += 1

    final_pop = len(engine._active_colonists())
    return {
        "years_run": len(years),
        "final_population": final_pop,
        "total_deaths": deaths,
        "total_births": births,
        "total_subsims": subsims,
        "governance_changes": gov_changes,
        "final_resources": engine.resources.to_dict(),
        "final_governance": engine.governance.to_dict(),
        "final_cohesion": engine.social.colony_cohesion(engine._active_ids()),
        "tech_completed": list(engine.infra.completed),
    }


def compare_timelines(
    baseline: dict,
    intervention: dict,
) -> TimelineDelta:
    """Compute deltas between baseline and intervention summaries."""
    rd = {}
    for r in RESOURCE_NAMES:
        b_val = baseline["final_resources"].get(r, 0.0)
        i_val = intervention["final_resources"].get(r, 0.0)
        rd[r] = round(i_val - b_val, 4)

    return TimelineDelta(
        population_delta=intervention["final_population"] - baseline["final_population"],
        total_deaths_delta=intervention["total_deaths"] - baseline["total_deaths"],
        resource_deltas=rd,
        governance_changes_delta=intervention["governance_changes"] - baseline["governance_changes"],
        cohesion_delta=round(
            intervention["final_cohesion"] - baseline["final_cohesion"], 4
        ),
        subsim_delta=intervention["total_subsims"] - baseline["total_subsims"],
        births_delta=intervention["total_births"] - baseline["total_births"],
        tech_count_delta=(
            len(intervention["tech_completed"]) - len(baseline["tech_completed"])
        ),
    )


def _verdict(delta: TimelineDelta, intervention_type: str) -> str:
    """Generate a one-line verdict from the delta."""
    pop = delta.population_delta
    deaths = delta.total_deaths_delta
    avg_res = sum(delta.resource_deltas.values()) / max(len(delta.resource_deltas), 1)

    parts = []
    if pop > 0:
        parts.append(f"+{pop} survivors")
    elif pop < 0:
        parts.append(f"{pop} survivors")

    if deaths < 0:
        parts.append(f"{deaths} deaths (fewer)")
    elif deaths > 0:
        parts.append(f"+{deaths} deaths (more)")

    if abs(avg_res) > 0.01:
        direction = "higher" if avg_res > 0 else "lower"
        parts.append(f"resources {direction} ({avg_res:+.3f} avg)")

    if abs(delta.cohesion_delta) > 0.01:
        direction = "higher" if delta.cohesion_delta > 0 else "lower"
        parts.append(f"cohesion {direction} ({delta.cohesion_delta:+.3f})")

    if not parts:
        return "negligible effect"
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Counterfactual execution
# ---------------------------------------------------------------------------

def run_counterfactual(
    snapshot: Mars100Engine,
    question: str,
    intervention_type: str,
    params: dict,
    horizon: int = 20,
) -> CounterfactualResult:
    """Run one counterfactual question against a snapshot.

    Both baseline and intervention branch from the *same* snapshot
    (same RNG state) so deltas isolate the intervention effect.
    """
    # Baseline branch -- no intervention, same RNG
    baseline_engine = branch_engine(snapshot)
    baseline_summary = run_forward(baseline_engine, horizon)

    # Intervention branch
    intervention_engine = branch_engine(snapshot)
    apply_intervention(intervention_engine, intervention_type, params)
    intervention_summary = run_forward(intervention_engine, horizon)

    delta = compare_timelines(baseline_summary, intervention_summary)
    verdict = _verdict(delta, intervention_type)

    return CounterfactualResult(
        question=question,
        intervention_type=intervention_type,
        branch_year=snapshot.year,
        horizon=horizon,
        baseline_summary=baseline_summary,
        intervention_summary=intervention_summary,
        delta={
            "population": delta.population_delta,
            "deaths": delta.total_deaths_delta,
            "resources": delta.resource_deltas,
            "governance_changes": delta.governance_changes_delta,
            "cohesion": delta.cohesion_delta,
            "subsims": delta.subsim_delta,
            "births": delta.births_delta,
            "tech_count": delta.tech_count_delta,
        },
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Scenario generation from emergence data
# ---------------------------------------------------------------------------

def generate_counterfactuals(emergence: dict) -> list:
    """Generate counterfactual scenarios from emergence analysis data.

    Returns a list of dicts with keys: question, intervention_type, params,
    branch_year, horizon.
    """
    scenarios = []

    # 1. Governance phase transitions
    phases = emergence.get("governance_phases", [])
    gov_types = ["anarchy", "direct_democracy", "council", "dictator"]
    for phase in phases:
        transition_year = phase.get("start_year", 1)
        current_gov = phase.get("gov_type", "anarchy")
        for alt_gov in gov_types:
            if alt_gov == current_gov:
                continue
            scenarios.append({
                "question": (
                    f"What if {alt_gov} replaced {current_gov} "
                    f"at year {transition_year}?"
                ),
                "intervention_type": "force_governance",
                "params": {"gov_type": alt_gov},
                "branch_year": max(1, transition_year),
                "horizon": 20,
            })

    # 2. Earlier tech unlocks
    for tech_id, tech in TECH_BY_ID.items():
        scenarios.append({
            "question": f"What if {tech.name} was completed by year 10?",
            "intervention_type": "force_tech",
            "params": {"tech_id": tech_id},
            "branch_year": 10,
            "horizon": 30,
        })

    # 3. Resource shocks from mortality patterns
    mortality = emergence.get("mortality", {})
    deadliest = mortality.get("deadliest_decade", 50)
    shock_year = max(1, deadliest - 5)
    for resource in ("air", "food", "water"):
        scenarios.append({
            "question": (
                f"What if a +0.3 {resource} boost arrived "
                f"at year {shock_year} (before peak mortality)?"
            ),
            "intervention_type": "resource_boost",
            "params": {"resource": resource, "amount": 0.3},
            "branch_year": shock_year,
            "horizon": 20,
        })

    return scenarios


def run_all_counterfactuals(
    emergence: dict,
    seed: int = 42,
    total_years: int = 100,
) -> list:
    """Generate and run all counterfactuals.  Returns list of result dicts."""
    scenarios = generate_counterfactuals(emergence)

    # Collect checkpoint years needed
    branch_years = sorted({s["branch_year"] for s in scenarios})
    snapshots = collect_checkpoints(seed, total_years, branch_years)

    results = []
    for scenario in scenarios:
        by = scenario["branch_year"]
        # Find the closest available snapshot at or before branch_year
        available = [y for y in snapshots if y <= by]
        if not available:
            continue
        snap_year = max(available)
        snapshot = snapshots[snap_year]

        result = run_counterfactual(
            snapshot=snapshot,
            question=scenario["question"],
            intervention_type=scenario["intervention_type"],
            params=scenario["params"],
            horizon=scenario["horizon"],
        )
        results.append(result.to_dict())

    return results
