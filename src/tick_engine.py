"""
tick_engine.py — Mars Barn terrarium simulation engine.

Runs N colonies for M sols on a shared Mars environment.
Deterministic (seeded RNG). One tick = one sol.

Usage:
    from tick_engine import Simulation
    sim = Simulation(sols=365)
    results = sim.run()
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from src.mars_env import MarsEnvironment
from src.mars_colony import Colony, create_colony


DEFAULT_COLONIES = [
    ("Ares Prime", "conservative", 1001),
    ("Olympus Station", "balanced", 2002),
    ("Red Frontier", "aggressive", 3003),
]


class Simulation:
    """Mars Barn terrarium — shared environment, multiple colonies."""

    def __init__(
        self,
        sols: int = 365,
        env_seed: int = 42,
        colonies: list[tuple[str, str, int]] | None = None,
    ) -> None:
        self.total_sols = sols
        self.env = MarsEnvironment(seed=env_seed)
        colony_specs = colonies or DEFAULT_COLONIES
        self.colonies: list[Colony] = [
            create_colony(name, strategy, seed)
            for name, strategy, seed in colony_specs
        ]
        self.env_history: list[dict] = []

    def tick(self) -> dict:
        """Advance all colonies by one sol. Returns env snapshot."""
        env_snap = self.env.tick()
        self.env_history.append(env_snap)
        for colony in self.colonies:
            colony.tick(env_snap)
        return env_snap

    def run(self, callback: object = None) -> dict:
        """Run the full simulation. Returns results dict.

        callback: optional callable(sol, env, colonies) invoked each sol.
        """
        for _ in range(self.total_sols):
            env_snap = self.tick()
            if callback is not None:
                callback(env_snap["sol"], env_snap, self.colonies)

        return self.results()

    def results(self) -> dict:
        """Package simulation results as a serializable dict."""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "_meta": {
                "engine": "mars-barn",
                "version": "1.0",
                "sols": self.total_sols,
                "generated": now,
            },
            "environment": {
                "history": self.env_history,
            },
            "colonies": [
                {
                    "name": c.name,
                    "strategy": c.strategy,
                    "final_population": c.population,
                    "total_births": c.total_births,
                    "total_deaths": c.total_deaths,
                    "final_morale": round(c.morale, 3),
                    "cumulative_radiation_msv": round(c.cumulative_radiation_msv, 2),
                    "history": c.history,
                    "events": c.events[-100:],  # trim to last 100
                }
                for c in self.colonies
            ],
            "summary": self._summary(),
        }

    def _summary(self) -> dict:
        """Aggregate summary stats."""
        summaries = []
        for c in self.colonies:
            pops = [h["population"] for h in c.history]
            summaries.append({
                "name": c.name,
                "strategy": c.strategy,
                "start_pop": pops[0] if pops else 0,
                "end_pop": pops[-1] if pops else 0,
                "peak_pop": max(pops) if pops else 0,
                "min_pop": min(pops) if pops else 0,
                "total_births": c.total_births,
                "total_deaths": c.total_deaths,
                "growth_pct": round(
                    (pops[-1] - pops[0]) / max(1, pops[0]) * 100, 1
                ) if pops else 0,
            })
        return {"colonies": summaries}
