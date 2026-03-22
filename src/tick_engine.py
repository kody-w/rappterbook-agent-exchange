"""
tick_engine.py — Mars Barn terrarium simulation engine.

Runs N colonies for M sols on a shared Mars environment.
Deterministic (seeded RNG). One tick = one sol.
Inter-colony migration based on relative attractiveness.

Usage:
    from tick_engine import Simulation
    sim = Simulation(sols=365)
    results = sim.run()
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from datetime import datetime, timezone

from src.mars_env import MarsEnvironment
from src.mars_colony import Colony, create_colony


DEFAULT_COLONIES = [
    ("Ares Prime", "conservative", 1001),
    ("Olympus Station", "balanced", 2002),
    ("Red Frontier", "aggressive", 3003),
]

# Migration: fraction of population that considers moving each sol
MIGRATION_RATE = 0.015  # ~1.5% consider moving per sol
MIGRATION_MIN_POP = 10  # colonies below this don't lose migrants


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
        self.migration_rng = random.Random(env_seed + 7777)
        colony_specs = colonies or DEFAULT_COLONIES
        self.colonies: list[Colony] = [
            create_colony(name, strategy, seed)
            for name, strategy, seed in colony_specs
        ]
        self.env_history: list[dict] = []
        self.migration_log: list[dict] = []

    def _migrate(self) -> None:
        """Inter-colony migration based on relative attractiveness.

        Each sol, a fraction of colonists consider moving.
        They probabilistically move toward more attractive colonies.
        Zero-sum: total emigrants == total immigrants.
        """
        n = len(self.colonies)
        if n < 2:
            return

        scores = [c.attractiveness() for c in self.colonies]
        total_score = sum(scores)
        if total_score <= 0:
            return

        # Weighted probabilities for destination choice
        probs = [s / total_score for s in scores]

        # Each colony contributes migrants proportional to dissatisfaction
        net_flows = [0] * n
        sol = self.colonies[0].sol

        for i, colony in enumerate(self.colonies):
            if colony.population < MIGRATION_MIN_POP:
                continue
            # Colonists consider leaving if their colony is below average
            avg_score = total_score / n
            if scores[i] >= avg_score * 1.1:
                continue  # happy enough to stay

            candidates = int(colony.population * MIGRATION_RATE)
            for _ in range(candidates):
                if self.migration_rng.random() > 0.5:
                    continue  # decides to stay

                # Pick destination weighted by attractiveness
                r = self.migration_rng.random()
                cumulative = 0.0
                dest = i  # fallback
                for j in range(n):
                    if j == i:
                        continue
                    cumulative += probs[j] / (1 - probs[i]) if probs[i] < 1 else 0
                    if r <= cumulative:
                        dest = j
                        break
                if dest != i:
                    net_flows[i] -= 1
                    net_flows[dest] += 1

        # Apply migration
        for i, colony in enumerate(self.colonies):
            if net_flows[i] != 0:
                colony.apply_migration(net_flows[i])

        # Log significant migration events
        total_moved = sum(max(0, f) for f in net_flows)
        if total_moved > 0:
            self.migration_log.append({
                "sol": sol,
                "flows": {self.colonies[i].name: net_flows[i] for i in range(n) if net_flows[i] != 0},
                "total_moved": total_moved,
            })

    def tick(self) -> dict:
        """Advance all colonies by one sol. Returns env snapshot."""
        env_snap = self.env.tick()
        self.env_history.append(env_snap)
        for colony in self.colonies:
            colony.tick(env_snap)
        # Inter-colony migration happens after all colonies tick
        self._migrate()
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
                "version": "1.1",
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
                    "initial_population": c.initial_population,
                    "final_population": c.population,
                    "total_births": c.total_births,
                    "total_deaths": c.total_deaths,
                    "total_immigrants": c.total_immigrants,
                    "total_emigrants": c.total_emigrants,
                    "final_morale": round(c.morale, 3),
                    "cumulative_radiation_msv": round(c.cumulative_radiation_msv, 2),
                    "history": c.history,
                    "events": c.events[-100:],  # trim to last 100
                }
                for c in self.colonies
            ],
            "migration": {
                "total_events": len(self.migration_log),
                "log": self.migration_log[-50:],  # trim
            },
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
                "total_immigrants": c.total_immigrants,
                "total_emigrants": c.total_emigrants,
                "net_migration": c.total_immigrants - c.total_emigrants,
                "growth_pct": round(
                    (pops[-1] - pops[0]) / max(1, pops[0]) * 100, 1
                ) if pops else 0,
            })
        return {"colonies": summaries}
