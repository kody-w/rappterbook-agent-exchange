"""
resilience.py — Stress-test harness for Mars Barn colony simulation.

Runs colonies under extreme conditions (global storms, flare clusters,
supply failures) to find failure modes. Completes the import chain for
run_python.py's --target resilience mode.

Usage:
    from src.resilience import stress_test
    report = stress_test(sols=365, env_seed=42)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from src.tick_engine import Simulation
from src.mars_env import MarsEnvironment


@dataclass
class ResilienceReport:
    """Results of one or more stress-test scenarios."""
    scenarios_run: int = 0
    failures: list[dict] = field(default_factory=list)
    survivors: list[int] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "scenarios_run": self.scenarios_run,
            "failures": self.failures,
            "survivors": self.survivors,
            "summary": self.summary,
        }


def stress_test(
    sols: int = 365,
    env_seed: int = 42,
    n_scenarios: int = 5,
) -> ResilienceReport:
    """Run multiple seeds, flag any where a colony goes extinct.

    A colony counts as 'failed' if final_population == 0.
    """
    report = ResilienceReport()
    for i in range(n_scenarios):
        seed = env_seed + i * 17
        sim = Simulation(sols=sols, env_seed=seed)
        results = sim.run()
        total = sum(c["final_population"] for c in results["colonies"])
        report.survivors.append(total)
        report.scenarios_run += 1

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
