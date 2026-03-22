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
from src.mars_colony import Colony, create_colony, FOOD_KG_SOL, HABITAT_M2_MIN


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

        # Each colony ticks independently first
        for colony in self.colonies:
            colony.tick(env_snap)

        # Inter-colony migration (quality-of-life driven)
        self._migrate(env_snap)

        # Pandemic spread between colonies
        self._pandemic_spread(env_snap)

        # Reconcile history — migration and pandemic happened after snapshot
        for colony in self.colonies:
            if colony.history:
                colony.history[-1]["population"] = colony.population

        return env_snap

    def _migrate(self, env: dict) -> None:
        """Colonists migrate toward colonies with better conditions.

        Migration rate is small (~0.5% of pop per sol) and only triggers
        when quality-of-life differential exceeds threshold.
        """
        if len(self.colonies) < 2:
            return

        # Compute attractiveness score for each colony
        scores = []
        for c in self.colonies:
            food_days = c.food_kg / max(1, c.population * FOOD_KG_SOL) if c.population > 0 else 0
            density = c.population / max(1, c.habitat_m2 / HABITAT_M2_MIN) if c.population > 0 else 0
            score = (
                c.morale * 0.4
                + min(1.0, food_days / 60) * 0.3
                + max(0, 1.0 - density) * 0.2
                + c.medical_level * 0.1
            )
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        for i, src in enumerate(self.colonies):
            if src.population < 10:
                continue  # too small to lose anyone

            for j, dst in enumerate(self.colonies):
                if i == j:
                    continue

                diff = scores[j] - scores[i]
                if diff < 0.05:
                    continue  # not attractive enough

                # Migration rate scales with differential
                rate = min(0.005, diff * 0.01)
                migrants = 0
                for _ in range(src.population):
                    if src.rng.random() < rate:
                        migrants += 1

                migrants = min(migrants, src.population - 5)  # keep minimum viable pop
                if migrants > 0:
                    src.population -= migrants
                    dst.population += migrants
                    src.total_migrations_out = getattr(src, "total_migrations_out", 0) + migrants
                    dst.total_migrations_in = getattr(dst, "total_migrations_in", 0) + migrants
                    if migrants >= 3:
                        src.events.append({
                            "sol": env["sol"], "type": "migration_out",
                            "count": migrants, "to": dst.name,
                        })
                        dst.events.append({
                            "sol": env["sol"], "type": "migration_in",
                            "count": migrants, "from": src.name,
                        })

    def _pandemic_spread(self, env: dict) -> None:
        """Rare pandemic events that spread between dense colonies.

        ~0.3% chance per sol, but only affects colonies above 80% capacity.
        Disease severity scales with density. Creates selection pressure
        favoring balanced growth over aggressive expansion.
        """
        # Check if pandemic starts (shared environment RNG)
        if self.env.rng.random() > 0.003:
            return

        severity = self.env.rng.uniform(0.005, 0.02)  # daily death rate boost

        for colony in self.colonies:
            if colony.population < 10:
                continue

            density = colony.population / max(1, colony.habitat_m2 / HABITAT_M2_MIN)
            if density < 0.6:
                continue  # sparse colonies avoid pandemic

            # Medical level reduces impact
            effective_medical = min(1.0, colony.medical_level + colony.medical_breakthroughs * 0.05)
            adjusted_severity = severity * density * (1.0 - 0.6 * effective_medical)

            pandemic_deaths = 0
            for _ in range(colony.population):
                if colony.rng.random() < adjusted_severity:
                    pandemic_deaths += 1

            pandemic_deaths = min(pandemic_deaths, colony.population - 1)
            if pandemic_deaths > 0:
                colony.population -= pandemic_deaths
                colony.total_deaths += pandemic_deaths
                colony.morale = max(0.0, colony.morale - 0.1)
                # Record pandemic deaths in the sol's history entry
                if colony.history:
                    colony.history[-1]["deaths"] += pandemic_deaths
                colony.events.append({
                    "sol": env["sol"], "type": "pandemic",
                    "deaths": pandemic_deaths, "severity": round(severity, 4),
                })

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
                    "total_migrations_in": getattr(c, "total_migrations_in", 0),
                    "total_migrations_out": getattr(c, "total_migrations_out", 0),
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
                "migrations_in": getattr(c, "total_migrations_in", 0),
                "migrations_out": getattr(c, "total_migrations_out", 0),
                "growth_pct": round(
                    (pops[-1] - pops[0]) / max(1, pops[0]) * 100, 1
                ) if pops else 0,
            })
        return {"colonies": summaries}
