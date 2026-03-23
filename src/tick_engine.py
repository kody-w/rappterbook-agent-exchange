"""
tick_engine.py — Mars Barn terrarium simulation engine.

Runs N colonies for M sols on a shared Mars environment.
Deterministic (seeded RNG). One tick = one sol.
Supports inter-colony migration, epidemic spread, and food trade.

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
from src.mars_colony import (
    Colony, create_colony, Epidemic,
    FOOD_KG_SOL, HABITAT_M2_MIN,
    EPIDEMIC_MIN_POP, EPIDEMIC_STRAINS,
)
from src.tech_tree import ResearchEngine


DEFAULT_COLONIES = [
    ("Ares Prime", "conservative", 1001),
    ("Olympus Station", "balanced", 2002),
    ("Red Frontier", "aggressive", 3003),
]

# Migration parameters
MIGRATION_MORALE_THRESHOLD = 0.12
MIGRATION_BASE_RATE = 0.003
EMERGENCY_FOOD_SOLS = 10
EMERGENCY_EVAC_FRACTION = 0.08


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
        # Initialize tech research engines per colony
        for i, colony in enumerate(self.colonies):
            colony.research_engine = ResearchEngine(
                strategy=colony.strategy,
                rng=random.Random(colony_specs[i][2] + 9999),
            )
        self.migration_rng = random.Random(env_seed + 7777)
        self.initial_populations = [c.population for c in self.colonies]
        self.total_migrations = 0
        self.env_history: list[dict] = []

    def _colony_attractiveness(self, colony: Colony) -> float:
        """Score how attractive a colony is for migration."""
        if colony.population == 0:
            return 0.0
        food_days = colony.food_kg / max(1, colony.population * FOOD_KG_SOL)
        food_score = min(1.0, food_days / 120)
        density = colony.population / max(1, colony.habitat_m2 / HABITAT_M2_MIN)
        headroom = max(0.0, 1.0 - density)
        return colony.morale * 0.5 + food_score * 0.3 + headroom * 0.2

    def _process_migration(self, env_snap: dict) -> None:
        """Inter-colony migration — morale/resource differential pulls colonists."""
        if len(self.colonies) < 2:
            return
        scores = [self._colony_attractiveness(c) for c in self.colonies]
        transfers: list[tuple[int, int, int]] = []
        for i, colony in enumerate(self.colonies):
            if colony.population < 5:
                continue
            food_sols = colony.food_kg / max(1, colony.population * FOOD_KG_SOL)
            if food_sols < EMERGENCY_FOOD_SOLS and colony.population > 10:
                best_j = max(
                    (j for j in range(len(self.colonies)) if j != i),
                    key=lambda j: scores[j],
                )
                if scores[best_j] > scores[i] + 0.05:
                    evac = max(1, int(colony.population * EMERGENCY_EVAC_FRACTION))
                    evac = min(evac, colony.population - 2)
                    if evac > 0:
                        transfers.append((i, best_j, evac))
                        colony.events.append({
                            "sol": colony.sol, "type": "evacuation",
                            "to": self.colonies[best_j].name, "count": evac,
                        })
                continue
            for j, other in enumerate(self.colonies):
                if j == i or other.population == 0:
                    continue
                morale_gap = scores[j] - scores[i]
                if morale_gap > MIGRATION_MORALE_THRESHOLD:
                    migrants = 0
                    for _ in range(int(colony.population * MIGRATION_BASE_RATE * 3)):
                        if self.migration_rng.random() < morale_gap * 0.5:
                            migrants += 1
                    migrants = min(migrants, colony.population - 2)
                    if migrants > 0:
                        transfers.append((i, j, migrants))
        for from_idx, to_idx, count in transfers:
            actual = min(count, self.colonies[from_idx].population - 1)
            if actual <= 0:
                continue
            self.colonies[from_idx].population -= actual
            self.colonies[to_idx].population += actual
            self.colonies[from_idx].total_emigrants += actual
            self.colonies[to_idx].total_immigrants += actual
            self.colonies[to_idx].receive_immigrants(actual)
            self.total_migrations += actual
            if self.colonies[from_idx].history:
                self.colonies[from_idx].history[-1]["net_migration"] -= actual
                self.colonies[from_idx].history[-1]["population"] = self.colonies[from_idx].population
            if self.colonies[to_idx].history:
                self.colonies[to_idx].history[-1]["net_migration"] += actual
                self.colonies[to_idx].history[-1]["population"] = self.colonies[to_idx].population

    def _spread_epidemics(self) -> None:
        """Cross-colony epidemic contagion via contact."""
        infected = [(i, c) for i, c in enumerate(self.colonies)
                     if c.epidemic is not None and not c.epidemic.quarantined]
        clean = [(i, c) for i, c in enumerate(self.colonies)
                  if c.epidemic is None and c.population >= EPIDEMIC_MIN_POP]
        if not infected or not clean:
            return
        for _si, src in infected:
            for _di, dst in clean:
                spread_chance = 0.005 * src.epidemic.infection_rate()
                spread_chance *= (1.0 - 0.5 * dst.medical_level)
                if self.migration_rng.random() < spread_chance:
                    strain = next(
                        (s for s in EPIDEMIC_STRAINS if s["name"] == src.epidemic.strain),
                        EPIDEMIC_STRAINS[0],
                    )
                    dur = self.migration_rng.randint(strain["duration"][0], strain["duration"][1])
                    dst.epidemic = Epidemic(strain, dur, dst.population)
                    dst.morale = max(0.0, dst.morale - 0.08)
                    dst.events.append({
                        "sol": dst.sol, "type": "epidemic_spread",
                        "strain": strain["name"], "from": src.name,
                    })

    def _process_food_trade(self) -> None:
        """Inter-colony food trade — surplus shares with starving (zero-sum)."""
        if len(self.colonies) < 2:
            return
        food_security = []
        for c in self.colonies:
            daily = max(1, c.population * FOOD_KG_SOL)
            food_security.append(c.food_kg / daily)
        surplus = [(i, c) for i, c in enumerate(self.colonies)
                   if food_security[i] > 90 and c.population > 0]
        deficit = [(i, c) for i, c in enumerate(self.colonies)
                   if food_security[i] < 30 and c.population > 0]
        if not surplus or not deficit:
            return
        for di, dc in deficit:
            need = max(1, dc.population * FOOD_KG_SOL)
            shortfall = max(0, 30 * need - dc.food_kg)
            for si, sc in surplus:
                sn = max(1, sc.population * FOOD_KG_SOL)
                tradeable = min(sc.food_kg - 90 * sn, sc.food_kg * 0.05)
                transfer = min(tradeable, shortfall)
                if transfer > 10:
                    sc.food_kg -= transfer
                    dc.food_kg += transfer
                    shortfall -= transfer
                if shortfall <= 0:
                    break

    def tick(self) -> dict:
        """Advance all colonies by one sol. Returns env snapshot."""
        env_snap = self.env.tick()
        self.env_history.append(env_snap)
        for colony in self.colonies:
            colony.tick(env_snap)

        # Terraforming feedback: colony industrial output modifies the
        # shared environment. The aggregated delta feeds into the next sol.
        total_terraform_delta = sum(
            colony.history[-1]["terraforming_contribution"]
            for colony in self.colonies
            if colony.history
        )
        if total_terraform_delta > 0:
            self.env.apply_terraforming(total_terraform_delta)

        self._process_migration(env_snap)
        self._spread_epidemics()
        self._process_food_trade()
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
                "version": "5.0",
                "sols": self.total_sols,
                "generated": now,
            },
            "environment": {
                "history": self.env_history,
                "final_terraforming_progress": round(self.env.terraforming_progress, 6),
                "terraform_phase": self.env.terraform_phase(),
            },
            "colonies": [
                {
                    "name": c.name,
                    "strategy": c.strategy,
                    "initial_population": self.initial_populations[i],
                    "final_population": c.population,
                    "total_births": c.total_births,
                    "total_deaths": c.total_deaths,
                    "total_immigrants": c.total_immigrants,
                    "total_emigrants": c.total_emigrants,
                    "final_morale": round(c.morale, 3),
                    "cumulative_radiation_msv": round(c.cumulative_radiation_msv, 2),
                    "death_causes": c.cumulative_death_causes,
                    "terraforming_output": round(c.terraforming_output, 6),
                    "tech": c.research_engine.snapshot() if c.research_engine else None,
                    "history": c.history,
                    "events": c.events[-150:],
                }
                for i, c in enumerate(self.colonies)
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
                "death_causes": c.cumulative_death_causes,
                "net_migration": sum(h.get("net_migration", 0) for h in c.history),
                "techs_unlocked": (
                    len(c.research_engine.unlocked) if c.research_engine else 0
                ),
                "growth_pct": round(
                    (pops[-1] - pops[0]) / max(1, pops[0]) * 100, 1
                ) if pops else 0,
                "terraforming_output": round(c.terraforming_output, 6),
            })
        return {
            "colonies": summaries,
            "total_migrations": self.total_migrations,
            "terraforming": {
                "progress": round(self.env.terraforming_progress, 6),
                "phase": self.env.terraform_phase(),
                "contributions": {
                    c.name: round(c.terraforming_output, 6)
                    for c in self.colonies
                },
            },
        }
