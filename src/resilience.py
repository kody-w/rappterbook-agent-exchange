"""
resilience.py — Stress-test harness for Mars Barn colony simulation.

Runs colonies under extreme conditions to find failure modes.
Produces structured reports with per-scenario breakdowns,
recovery metrics, and extinction thresholds.

Usage:
    from src.resilience import stress_test, scenario_sweep, find_extinction_threshold
    report = stress_test(sols=365, env_seed=42)
    sweep = scenario_sweep(sols=200, seeds=range(10))
    threshold = find_extinction_threshold(low=10, high=500)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from src.tick_engine import Simulation
from src.mars_env import MarsEnvironment


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ColonySnapshot:
    """Per-colony outcome from a single scenario."""
    name: str
    strategy: str
    initial_pop: int
    final_pop: int
    peak_pop: int
    trough_pop: int
    total_births: int
    total_deaths: int
    survived: bool
    morale_min: float
    morale_max: float
    recovery_sol: int | None = None

    def growth_pct(self) -> float:
        """Population growth as a percentage."""
        if self.initial_pop == 0:
            return 0.0
        return (self.final_pop - self.initial_pop) / self.initial_pop * 100

    def to_dict(self) -> dict:
        """JSON-safe representation."""
        return {
            "name": self.name, "strategy": self.strategy,
            "initial_pop": self.initial_pop, "final_pop": self.final_pop,
            "peak_pop": self.peak_pop, "trough_pop": self.trough_pop,
            "total_births": self.total_births, "total_deaths": self.total_deaths,
            "survived": self.survived,
            "morale_min": round(self.morale_min, 3),
            "morale_max": round(self.morale_max, 3),
            "growth_pct": round(self.growth_pct(), 1),
            "recovery_sol": self.recovery_sol,
        }


@dataclass
class ScenarioResult:
    """Full outcome of one simulation run."""
    seed: int
    sols: int
    total_survivors: int
    colony_snapshots: list[ColonySnapshot] = field(default_factory=list)
    extinctions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe representation."""
        return {
            "seed": self.seed, "sols": self.sols,
            "total_survivors": self.total_survivors,
            "extinctions": self.extinctions,
            "colonies": [c.to_dict() for c in self.colony_snapshots],
        }


@dataclass
class ResilienceReport:
    """Aggregated results across multiple stress-test scenarios."""
    scenarios_run: int = 0
    failures: list[dict] = field(default_factory=list)
    survivors: list[int] = field(default_factory=list)
    summary: str = ""
    scenario_results: list[ScenarioResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe representation."""
        return {
            "scenarios_run": self.scenarios_run,
            "failures": self.failures,
            "survivors": self.survivors,
            "summary": self.summary,
            "scenarios": [s.to_dict() for s in self.scenario_results],
        }

    def survival_rate(self) -> float:
        """Fraction of scenarios where ALL colonies survived."""
        if self.scenarios_run == 0:
            return 0.0
        all_survived = sum(
            1 for sr in self.scenario_results if not sr.extinctions
        )
        return all_survived / self.scenarios_run

    def worst_colony(self) -> str | None:
        """Name of the colony that went extinct most often."""
        if not self.failures:
            return None
        counts: dict[str, int] = {}
        for f in self.failures:
            name = f["colony"]
            counts[name] = counts.get(name, 0) + 1
        return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# Core: extract colony snapshot from simulation results
# ---------------------------------------------------------------------------

def _extract_snapshot(colony_data: dict) -> ColonySnapshot:
    """Build a ColonySnapshot from one colony's result dict."""
    history = colony_data.get("history", [])
    pops = [h["population"] for h in history] if history else [0]
    morales = [h["morale"] for h in history] if history else [0.0]
    initial_pop = colony_data.get("initial_population", pops[0] if pops else 0)
    final_pop = colony_data.get("final_population", pops[-1] if pops else 0)
    recovery_sol = None
    dipped = False
    for i, p in enumerate(pops):
        if p < initial_pop:
            dipped = True
        if dipped and p >= initial_pop:
            recovery_sol = i
            break
    return ColonySnapshot(
        name=colony_data["name"],
        strategy=colony_data.get("strategy", "unknown"),
        initial_pop=initial_pop, final_pop=final_pop,
        peak_pop=max(pops), trough_pop=min(pops),
        total_births=colony_data.get("total_births", 0),
        total_deaths=colony_data.get("total_deaths", 0),
        survived=final_pop > 0,
        morale_min=min(morales), morale_max=max(morales),
        recovery_sol=recovery_sol,
    )


# ---------------------------------------------------------------------------
# stress_test: multi-seed extinction scanner (backwards compatible)
# ---------------------------------------------------------------------------

def stress_test(
    sols: int = 365,
    env_seed: int = 42,
    n_scenarios: int = 5,
) -> ResilienceReport:
    """Run multiple seeds, flag any where a colony goes extinct.

    Backwards-compatible with the original API, enriched with
    scenario_results for deeper analysis.
    """
    report = ResilienceReport()
    for i in range(n_scenarios):
        seed = env_seed + i * 17
        sim = Simulation(sols=sols, env_seed=seed)
        results = sim.run()
        total = sum(c["final_population"] for c in results["colonies"])
        report.survivors.append(total)
        report.scenarios_run += 1
        snapshots = [_extract_snapshot(c) for c in results["colonies"]]
        extinctions = [s.name for s in snapshots if not s.survived]
        scenario = ScenarioResult(
            seed=seed, sols=sols, total_survivors=total,
            colony_snapshots=snapshots, extinctions=extinctions,
        )
        report.scenario_results.append(scenario)
        for colony in results["colonies"]:
            if colony["final_population"] == 0:
                report.failures.append({
                    "seed": seed,
                    "colony": colony["name"],
                    "deaths": colony["total_deaths"],
                })
    if report.survivors:
        avg = statistics.mean(report.survivors)
        report.summary = (
            f"{report.scenarios_run} scenarios, "
            f"{len(report.failures)} colony extinctions, "
            f"avg survivors: {avg:.0f}"
        )
    return report


# ---------------------------------------------------------------------------
# scenario_sweep: run across arbitrary seed ranges
# ---------------------------------------------------------------------------

def scenario_sweep(
    sols: int = 200,
    seeds: range | list[int] | None = None,
) -> ResilienceReport:
    """Run the sim across a range of seeds and aggregate."""
    if seeds is None:
        seeds = range(10)
    report = ResilienceReport()
    for seed in seeds:
        sim = Simulation(sols=sols, env_seed=seed)
        results = sim.run()
        total = sum(c["final_population"] for c in results["colonies"])
        report.survivors.append(total)
        report.scenarios_run += 1
        snapshots = [_extract_snapshot(c) for c in results["colonies"]]
        extinctions = [s.name for s in snapshots if not s.survived]
        scenario = ScenarioResult(
            seed=seed, sols=sols, total_survivors=total,
            colony_snapshots=snapshots, extinctions=extinctions,
        )
        report.scenario_results.append(scenario)
        for s in snapshots:
            if not s.survived:
                report.failures.append({
                    "seed": seed, "colony": s.name,
                    "deaths": s.total_deaths,
                })
    if report.survivors:
        avg = statistics.mean(report.survivors)
        report.summary = (
            f"{report.scenarios_run} scenarios, "
            f"{len(report.failures)} colony extinctions, "
            f"avg survivors: {avg:.0f}"
        )
    return report


# ---------------------------------------------------------------------------
# find_extinction_threshold: binary search for the sol count at which
# colonies start dying
# ---------------------------------------------------------------------------

def find_extinction_threshold(
    env_seed: int = 42,
    low: int = 10,
    high: int = 1000,
    tolerance: int = 5,
) -> dict:
    """Binary search for the minimum sols where extinction occurs."""
    iterations = 0
    while high - low > tolerance:
        mid = (low + high) // 2
        iterations += 1
        sim = Simulation(sols=mid, env_seed=env_seed)
        results = sim.run()
        any_dead = any(
            c["final_population"] == 0 for c in results["colonies"]
        )
        if any_dead:
            high = mid
        else:
            low = mid
    return {
        "threshold_sol": high,
        "converged": high - low <= tolerance,
        "iterations": iterations,
    }
