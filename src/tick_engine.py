"""
tick_engine.py — Mars Barn terrarium simulation engine.

Runs N colonies for M sols on a shared Mars environment.
Deterministic (seeded RNG). One tick = one sol.
Inter-colony migration driven by morale & resource differentials.

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


DEFAULT_COLONIES = [
    ("Ares Prime", "conservative", 1001),
    ("Olympus Station", "balanced", 2002),
    ("Red Frontier", "aggressive", 3003),
]

# Migration parameters
MIGRATION_MORALE_THRESHOLD = 0.12  # min morale gap to trigger migration
MIGRATION_BASE_RATE = 0.003  # fraction of pop that considers moving per sol
EMERGENCY_FOOD_SOLS = 10  # food reserves below this triggers evacuation
EMERGENCY_EVAC_FRACTION = 0.08  # fraction evacuated per sol in crisis


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
        self.initial_populations = [c.population for c in self.colonies]
        self.env_history: list[dict] = []
        self.total_migrations = 0

    def _colony_attractiveness(self, colony: Colony) -> float:
        """Score how attractive a colony is as a migration destination.

        Higher = more attractive. Combines morale, food security,
        and crowding headroom.
        """
        if colony.population == 0:
            return 0.0

        food_days = colony.food_kg / max(1, colony.population * FOOD_KG_SOL)
        food_score = min(1.0, food_days / 120)  # saturates at 120 sols reserve

        density = colony.population / max(1, colony.habitat_m2 / HABITAT_M2_MIN)
        headroom = max(0.0, 1.0 - density)

        return colony.morale * 0.5 + food_score * 0.3 + headroom * 0.2

    def _process_migration(self, env_snap: dict) -> None:
        """Inter-colony migration phase — runs after all colonies tick.

        Two modes:
        1. Normal migration: morale/resource differential pulls colonists
        2. Emergency evacuation: colony in crisis ejects population
        """
        if len(self.colonies) < 2:
            return

        scores = [self._colony_attractiveness(c) for c in self.colonies]
        transfers: list[tuple[int, int, int]] = []  # (from_idx, to_idx, count)

        for i, colony in enumerate(self.colonies):
            if colony.population < 5:
                continue

            # Emergency evacuation: critically low food
            food_sols = colony.food_kg / max(1, colony.population * FOOD_KG_SOL)
            if food_sols < EMERGENCY_FOOD_SOLS and colony.population > 10:
                best_j = max(
                    (j for j in range(len(self.colonies)) if j != i),
                    key=lambda j: scores[j],
                )
                if scores[best_j] > scores[i] + 0.05:
                    evac = max(1, int(colony.population * EMERGENCY_EVAC_FRACTION))
                    evac = min(evac, colony.population - 2)  # keep at least 2
                    if evac > 0:
                        transfers.append((i, best_j, evac))
                        colony.events.append({
                            "sol": colony.sol, "type": "evacuation",
                            "to": self.colonies[best_j].name, "count": evac,
                        })
                continue  # skip normal migration for crisis colonies

            # Normal migration: morale differential
            for j, other in enumerate(self.colonies):
                if j == i or other.population == 0:
                    continue
                morale_gap = scores[j] - scores[i]
                if morale_gap > MIGRATION_MORALE_THRESHOLD:
                    expected = colony.population * MIGRATION_BASE_RATE * morale_gap
                    migrants = 0
                    for _ in range(int(colony.population * MIGRATION_BASE_RATE * 3)):
                        if self.migration_rng.random() < morale_gap * 0.5:
                            migrants += 1
                    migrants = min(migrants, max(1, int(expected * 2)))
                    migrants = min(migrants, colony.population - 2)
                    if migrants > 0:
                        transfers.append((i, j, migrants))

        # Apply transfers (zero-sum)
        for from_idx, to_idx, count in transfers:
            actual = min(count, self.colonies[from_idx].population - 1)
            if actual <= 0:
                continue
            self.colonies[from_idx].population -= actual
            self.colonies[to_idx].population += actual
            self.colonies[from_idx].total_emigrants += actual
            self.colonies[to_idx].total_immigrants += actual
            self.total_migrations += actual

            # Update the current sol's history snapshot
            if self.colonies[from_idx].history:
                self.colonies[from_idx].history[-1]["net_migration"] -= actual
                self.colonies[from_idx].history[-1]["population"] = self.colonies[from_idx].population
            if self.colonies[to_idx].history:
                self.colonies[to_idx].history[-1]["net_migration"] += actual
                self.colonies[to_idx].history[-1]["population"] = self.colonies[to_idx].population

            self.colonies[from_idx].events.append({
                "sol": self.colonies[from_idx].sol,
                "type": "migration_out",
                "to": self.colonies[to_idx].name,
                "count": actual,
            })
            self.colonies[to_idx].events.append({
                "sol": self.colonies[to_idx].sol,
                "type": "migration_in",
                "from": self.colonies[from_idx].name,
                "count": actual,
            })

    def _spread_epidemics(self) -> None:
        """Cross-colony epidemic contagion via migration/trade contact.

        If colony A has an active epidemic and colony B doesn't,
        there's a small chance per sol that B catches it.
        Chance scales with migration volume and is blocked by quarantine.
        """
        infected = [(i, c) for i, c in enumerate(self.colonies)
                     if c.epidemic is not None and not c.epidemic.quarantined]
        clean = [(i, c) for i, c in enumerate(self.colonies)
                  if c.epidemic is None and c.population >= EPIDEMIC_MIN_POP]

        if not infected or not clean:
            return

        for _src_i, src in infected:
            for _dst_i, dst in clean:
                # ~0.5% per sol per infected neighbor, reduced by medical
                spread_chance = 0.005 * src.epidemic.infection_rate()
                spread_chance *= (1.0 - 0.5 * dst.medical_level)
                if self.migration_rng.random() < spread_chance:
                    # Same strain spreads
                    strain = next(
                        (s for s in EPIDEMIC_STRAINS if s["name"] == src.epidemic.strain),
                        EPIDEMIC_STRAINS[0],
                    )
                    dur = self.migration_rng.randint(
                        strain["duration"][0], strain["duration"][1]
                    )
                    dst.epidemic = Epidemic(strain, dur, dst.population)
                    dst.morale = max(0.0, dst.morale - 0.08)
                    dst.events.append({
                        "sol": dst.sol, "type": "epidemic_spread",
                        "strain": strain["name"],
                        "from": src.name,
                    })

    def _process_food_trade(self) -> None:
        """Inter-colony food trade — surplus colonies share with starving ones.

        Zero-sum: total food across all colonies is conserved.
        Trade rate proportional to deficit severity.
        """
        if len(self.colonies) < 2:
            return

        # Calculate food security (sols of reserves) per colony
        food_security = []
        for c in self.colonies:
            daily_need = max(1, c.population * FOOD_KG_SOL)
            food_security.append(c.food_kg / daily_need)

        SURPLUS_THRESHOLD = 90   # sols of food = "surplus"
        DEFICIT_THRESHOLD = 30   # sols of food = "deficit"
        MAX_TRADE_FRACTION = 0.05  # max 5% of surplus per sol

        surplus_colonies = [(i, c) for i, c in enumerate(self.colonies)
                           if food_security[i] > SURPLUS_THRESHOLD and c.population > 0]
        deficit_colonies = [(i, c) for i, c in enumerate(self.colonies)
                           if food_security[i] < DEFICIT_THRESHOLD and c.population > 0]

        if not surplus_colonies or not deficit_colonies:
            return

        for di, deficit_col in deficit_colonies:
            deficit_need = max(1, deficit_col.population * FOOD_KG_SOL)
            target_food = DEFICIT_THRESHOLD * deficit_need
            shortfall = max(0, target_food - deficit_col.food_kg)
            if shortfall <= 0:
                continue

            for si, surplus_col in surplus_colonies:
                surplus_need = max(1, surplus_col.population * FOOD_KG_SOL)
                tradeable = surplus_col.food_kg - SURPLUS_THRESHOLD * surplus_need
                tradeable = min(tradeable, surplus_col.food_kg * MAX_TRADE_FRACTION)
                transfer = min(tradeable, shortfall)
                if transfer > 10:  # minimum viable shipment
                    surplus_col.food_kg -= transfer
                    deficit_col.food_kg += transfer
                    shortfall -= transfer
                    surplus_col.events.append({
                        "sol": surplus_col.sol, "type": "food_trade_out",
                        "to": deficit_col.name, "kg": round(transfer, 1),
                    })
                    deficit_col.events.append({
                        "sol": deficit_col.sol, "type": "food_trade_in",
                        "from": surplus_col.name, "kg": round(transfer, 1),
                    })
                if shortfall <= 0:
                    break

    def tick(self) -> dict:
        """Advance all colonies by one sol. Returns env snapshot."""
        env_snap = self.env.tick()
        self.env_history.append(env_snap)
        for colony in self.colonies:
            colony.tick(env_snap)
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
                "version": "2.0",
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
                    "initial_population": self.initial_populations[i],
                    "final_population": c.population,
                    "total_births": c.total_births,
                    "total_deaths": c.total_deaths,
                    "total_immigrants": c.total_immigrants,
                    "total_emigrants": c.total_emigrants,
                    "death_causes": dict(c.death_causes),
                    "epidemics": sum(1 for e in c.events if e["type"] == "epidemic_start"),
                    "final_morale": round(c.morale, 3),
                    "cumulative_radiation_msv": round(c.cumulative_radiation_msv, 2),
                    "history": c.history,
                    "events": c.events[-150:],
                }
                for i, c in enumerate(self.colonies)
            ],
            "summary": self._summary(),
            "migration": {
                "total_transfers": self.total_migrations,
            },
        }

    def _summary(self) -> dict:
        """Aggregate summary stats."""
        summaries = []
        for i, c in enumerate(self.colonies):
            pops = [h["population"] for h in c.history]
            net_mig = sum(h.get("net_migration", 0) for h in c.history)
            summaries.append({
                "name": c.name,
                "strategy": c.strategy,
                "start_pop": self.initial_populations[i],
                "end_pop": pops[-1] if pops else 0,
                "peak_pop": max(pops) if pops else 0,
                "min_pop": min(pops) if pops else 0,
                "total_births": c.total_births,
                "total_deaths": c.total_deaths,
                "net_migration": net_mig,
                "growth_pct": round(
                    (pops[-1] - self.initial_populations[i])
                    / max(1, self.initial_populations[i]) * 100, 1
                ) if pops else 0,
            })
        total_epidemics = sum(
            sum(1 for e in c.events if e["type"] == "epidemic_start")
            for c in self.colonies
        )
        return {
            "colonies": summaries,
            "total_migrations": self.total_migrations,
            "total_epidemics": total_epidemics,
        }
