"""Tests for the Mars-100 counterfactual engine."""
from __future__ import annotations

import copy
import pytest

from src.mars100.engine import Mars100Engine
from src.mars100.colony import RESOURCE_NAMES
from src.mars100.infrastructure import TECH_BY_ID
from src.mars100.counterfactual import (
    CounterfactualResult,
    TimelineDelta,
    apply_intervention,
    branch_engine,
    collect_checkpoints,
    compare_timelines,
    generate_counterfactuals,
    run_all_counterfactuals,
    run_counterfactual,
    run_forward,
)


# -- Helpers ---------------------------------------------------------------

SEED = 42
SHORT_YEARS = 30  # shorter sim for speed in tests

MINIMAL_EMERGENCE = {
    "governance_phases": [
        {"gov_type": "anarchy", "start_year": 1, "end_year": 16, "duration": 15},
        {"gov_type": "direct_democracy", "start_year": 16, "end_year": None, "duration": 85},
    ],
    "mortality": {
        "total_deaths": 14,
        "causes": {"asphyxiation": 14},
        "deadliest_decade": 91,
    },
    "convergence": [],
    "factions": [],
    "crisis_resilience": [],
    "subsim_accuracy": 0.75,
    "total_subsims": 1304,
    "insights": [],
}


def _quick_engine(years: int = 15) -> Mars100Engine:
    """Build a small engine and run it partway."""
    eng = Mars100Engine(seed=SEED, total_years=years)
    for _ in range(10):
        if not eng._active_colonists():
            break
        eng.tick()
    return eng


# -- Data structure tests --------------------------------------------------

class TestTimelineDelta:
    def test_defaults(self) -> None:
        td = TimelineDelta()
        assert td.population_delta == 0
        assert td.cohesion_delta == 0.0
        assert td.resource_deltas == {}

    def test_custom_values(self) -> None:
        td = TimelineDelta(population_delta=5, total_deaths_delta=-3)
        assert td.population_delta == 5
        assert td.total_deaths_delta == -3


class TestCounterfactualResult:
    def test_to_dict_keys(self) -> None:
        cr = CounterfactualResult(
            question="test?",
            intervention_type="force_governance",
            branch_year=10,
            horizon=20,
        )
        d = cr.to_dict()
        assert set(d.keys()) == {
            "question", "intervention_type", "branch_year", "horizon",
            "baseline", "intervention", "delta", "verdict",
        }

    def test_to_dict_roundtrip(self) -> None:
        cr = CounterfactualResult(
            question="what if?",
            intervention_type="resource_boost",
            branch_year=5,
            horizon=10,
            delta={"population": 2},
            verdict="helpful",
        )
        d = cr.to_dict()
        assert d["question"] == "what if?"
        assert d["delta"] == {"population": 2}


# -- Branch engine ---------------------------------------------------------

class TestBranchEngine:
    def test_branch_preserves_year(self) -> None:
        eng = _quick_engine()
        branch = branch_engine(eng)
        assert branch.year == eng.year

    def test_branch_isolates_mutations(self) -> None:
        eng = _quick_engine()
        branch = branch_engine(eng)
        branch.resources.food = 0.0
        assert eng.resources.food > 0.0

    def test_branch_preserves_rng_state(self) -> None:
        eng = _quick_engine()
        b1 = branch_engine(eng)
        b2 = branch_engine(eng)
        # Both branches should produce the same random number
        assert b1.rng.random() == b2.rng.random()

    def test_branch_preserves_colonists(self) -> None:
        eng = _quick_engine()
        branch = branch_engine(eng)
        orig_ids = {c.id for c in eng.colonists}
        branch_ids = {c.id for c in branch.colonists}
        assert orig_ids == branch_ids


# -- Checkpoints -----------------------------------------------------------

class TestCheckpoints:
    def test_checkpoint_years_captured(self) -> None:
        cps = collect_checkpoints(SEED, SHORT_YEARS, [5, 10, 15])
        assert 5 in cps
        assert 10 in cps

    def test_checkpoint_is_frozen_state(self) -> None:
        cps = collect_checkpoints(SEED, SHORT_YEARS, [10])
        snap = cps[10]
        # The snapshot should be at year 9 (before tick 10)
        assert snap.year == 9

    def test_checkpoint_independent(self) -> None:
        cps = collect_checkpoints(SEED, SHORT_YEARS, [5, 10])
        snap5 = cps[5]
        snap10 = cps[10]
        assert snap5.year < snap10.year


# -- Interventions ---------------------------------------------------------

class TestInterventions:
    def test_force_governance(self) -> None:
        eng = _quick_engine()
        apply_intervention(eng, "force_governance", {"gov_type": "council"})
        assert eng.governance.gov_type == "council"

    def test_force_governance_clears_leader(self) -> None:
        eng = _quick_engine()
        eng.governance.leader = "someone"
        apply_intervention(eng, "force_governance", {"gov_type": "anarchy"})
        assert eng.governance.leader is None

    def test_force_tech(self) -> None:
        eng = _quick_engine()
        apply_intervention(eng, "force_tech", {"tech_id": "med_bay"})
        assert "med_bay" in eng.infra.completed

    def test_force_tech_no_duplicate(self) -> None:
        eng = _quick_engine()
        eng.infra.completed.append("med_bay")
        apply_intervention(eng, "force_tech", {"tech_id": "med_bay"})
        assert eng.infra.completed.count("med_bay") == 1

    def test_force_tech_unknown_raises(self) -> None:
        eng = _quick_engine()
        with pytest.raises(ValueError, match="Unknown tech"):
            apply_intervention(eng, "force_tech", {"tech_id": "warp_drive"})

    def test_resource_boost(self) -> None:
        eng = _quick_engine()
        before = eng.resources.food
        apply_intervention(eng, "resource_boost", {"resource": "food", "amount": 0.2})
        assert eng.resources.food == pytest.approx(min(1.0, before + 0.2), abs=1e-9)

    def test_resource_boost_clamped(self) -> None:
        eng = _quick_engine()
        apply_intervention(eng, "resource_boost", {"resource": "food", "amount": 5.0})
        assert eng.resources.food == 1.0

    def test_resource_boost_unknown_raises(self) -> None:
        eng = _quick_engine()
        with pytest.raises(ValueError, match="Unknown resource"):
            apply_intervention(eng, "resource_boost", {"resource": "unobtanium", "amount": 0.1})

    def test_unknown_intervention_raises(self) -> None:
        eng = _quick_engine()
        with pytest.raises(ValueError, match="Unknown intervention"):
            apply_intervention(eng, "unknown_type", {})


# -- Run forward -----------------------------------------------------------

class TestRunForward:
    def test_returns_required_keys(self) -> None:
        eng = _quick_engine()
        summary = run_forward(eng, horizon=5)
        required = {
            "years_run", "final_population", "total_deaths", "total_births",
            "total_subsims", "governance_changes", "final_resources",
            "final_governance", "final_cohesion", "tech_completed",
        }
        assert required <= set(summary.keys())

    def test_resource_bounds(self) -> None:
        eng = _quick_engine()
        summary = run_forward(eng, horizon=10)
        for r in RESOURCE_NAMES:
            val = summary["final_resources"].get(r, 0.0)
            assert 0.0 <= val <= 1.0, f"{r} out of bounds: {val}"

    def test_population_non_negative(self) -> None:
        eng = _quick_engine()
        summary = run_forward(eng, horizon=10)
        assert summary["final_population"] >= 0

    def test_deterministic(self) -> None:
        eng = _quick_engine()
        b1 = branch_engine(eng)
        b2 = branch_engine(eng)
        s1 = run_forward(b1, horizon=5)
        s2 = run_forward(b2, horizon=5)
        assert s1["final_population"] == s2["final_population"]
        assert s1["total_deaths"] == s2["total_deaths"]
        for r in RESOURCE_NAMES:
            assert s1["final_resources"][r] == pytest.approx(
                s2["final_resources"][r], abs=1e-9
            )


# -- Timeline comparison ---------------------------------------------------

class TestCompareTimelines:
    def test_identical_timelines_zero_delta(self) -> None:
        summary = {
            "final_population": 10,
            "total_deaths": 2,
            "total_births": 3,
            "total_subsims": 5,
            "governance_changes": 1,
            "final_resources": {r: 0.5 for r in RESOURCE_NAMES},
            "final_cohesion": 0.8,
            "tech_completed": ["med_bay"],
        }
        delta = compare_timelines(summary, summary)
        assert delta.population_delta == 0
        assert delta.total_deaths_delta == 0
        assert delta.cohesion_delta == 0.0
        assert all(v == 0.0 for v in delta.resource_deltas.values())

    def test_different_timelines(self) -> None:
        base = {
            "final_population": 10, "total_deaths": 2, "total_births": 1,
            "total_subsims": 5, "governance_changes": 1,
            "final_resources": {r: 0.5 for r in RESOURCE_NAMES},
            "final_cohesion": 0.8, "tech_completed": ["med_bay"],
        }
        inter = {
            **base,
            "final_population": 12,
            "total_deaths": 0,
            "final_resources": {r: 0.7 for r in RESOURCE_NAMES},
        }
        delta = compare_timelines(base, inter)
        assert delta.population_delta == 2
        assert delta.total_deaths_delta == -2
        for r in RESOURCE_NAMES:
            assert delta.resource_deltas[r] == pytest.approx(0.2, abs=1e-3)


# -- Counterfactual execution ----------------------------------------------

class TestRunCounterfactual:
    def test_result_structure(self) -> None:
        eng = _quick_engine()
        result = run_counterfactual(
            eng, "what if council?", "force_governance",
            {"gov_type": "council"}, horizon=5,
        )
        assert isinstance(result, CounterfactualResult)
        assert result.branch_year == eng.year
        assert result.horizon == 5
        assert result.verdict  # should be non-empty string

    def test_null_effect_small_delta(self) -> None:
        """Identical intervention type but same gov -> near-zero deltas."""
        eng = _quick_engine()
        current_gov = eng.governance.gov_type
        result = run_counterfactual(
            eng, "no change", "force_governance",
            {"gov_type": current_gov}, horizon=5,
        )
        d = result.delta
        # Population delta should be 0 (exact same simulation path)
        assert d["population"] == 0
        assert d["deaths"] == 0

    def test_resource_boost_helps(self) -> None:
        """A +0.3 food boost should generally not make things worse."""
        eng = _quick_engine()
        result = run_counterfactual(
            eng, "food boost", "resource_boost",
            {"resource": "food", "amount": 0.3}, horizon=10,
        )
        # Food delta should be non-negative (boost should not lose food)
        assert result.delta["resources"]["food"] >= -0.05


# -- Scenario generation ---------------------------------------------------

class TestGenerateCounterfactuals:
    def test_generates_governance_scenarios(self) -> None:
        scenarios = generate_counterfactuals(MINIMAL_EMERGENCE)
        gov_scenarios = [s for s in scenarios if s["intervention_type"] == "force_governance"]
        assert len(gov_scenarios) > 0

    def test_generates_tech_scenarios(self) -> None:
        scenarios = generate_counterfactuals(MINIMAL_EMERGENCE)
        tech_scenarios = [s for s in scenarios if s["intervention_type"] == "force_tech"]
        assert len(tech_scenarios) == len(TECH_BY_ID)

    def test_generates_resource_scenarios(self) -> None:
        scenarios = generate_counterfactuals(MINIMAL_EMERGENCE)
        res_scenarios = [s for s in scenarios if s["intervention_type"] == "resource_boost"]
        assert len(res_scenarios) == 3  # air, food, water

    def test_scenario_has_required_keys(self) -> None:
        scenarios = generate_counterfactuals(MINIMAL_EMERGENCE)
        for s in scenarios:
            assert "question" in s
            assert "intervention_type" in s
            assert "params" in s
            assert "branch_year" in s
            assert "horizon" in s

    def test_branch_years_positive(self) -> None:
        scenarios = generate_counterfactuals(MINIMAL_EMERGENCE)
        for s in scenarios:
            assert s["branch_year"] >= 1


# -- Full pipeline smoke test ----------------------------------------------

class TestSmokePipeline:
    @pytest.mark.slow
    def test_run_all_counterfactuals(self) -> None:
        """Full pipeline: generate + run all counterfactuals."""
        results = run_all_counterfactuals(MINIMAL_EMERGENCE, seed=SEED, total_years=SHORT_YEARS)
        assert len(results) > 0
        for r in results:
            assert "question" in r
            assert "baseline" in r
            assert "intervention" in r
            assert "delta" in r
            assert "verdict" in r

    @pytest.mark.slow
    def test_resource_values_bounded(self) -> None:
        """All resource values in counterfactual results stay in [0, 1]."""
        results = run_all_counterfactuals(MINIMAL_EMERGENCE, seed=SEED, total_years=SHORT_YEARS)
        for r in results:
            for timeline in ("baseline", "intervention"):
                for res_name in RESOURCE_NAMES:
                    val = r[timeline]["final_resources"].get(res_name, 0.0)
                    assert 0.0 <= val <= 1.0, (
                        f"{r['question']}: {timeline}.{res_name} = {val}"
                    )
