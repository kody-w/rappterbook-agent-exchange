"""
monte_carlo.py — Run Mars Barn sim across many seeds, compute percentile bands.

One seed is anecdote. Fifty seeds is data.
Produces per-sol percentile envelopes (p10, p25, p50, p75, p90) for
population, morale, food, genetic diversity per colony.
Also computes aggregate stats: survival rate, mean growth, strategy ranking.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.tick_engine import Simulation, DEFAULT_COLONIES


PERCENTILES = [10, 25, 50, 75, 90]


@dataclass
class EnsembleResult:
    """Aggregated results across many simulation seeds."""
    n_seeds: int
    sols: int
    colony_names: list[str]
    colony_strategies: list[str]
    bands: list[dict[str, list[list[float]]]]
    final_pop_stats: list[dict[str, float]]
    growth_pct_stats: list[dict[str, float]]
    survival_rates: list[float]
    canonical_results: dict
    canonical_seed: int


def _percentile(data: list[float], p: int) -> float:
    """Compute percentile from sorted list. Linear interpolation."""
    if not data:
        return 0.0
    n = len(data)
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return data[-1]
    d = k - f
    return data[f] * (1 - d) + data[c] * d


def run_ensemble(
    n_seeds: int = 50,
    sols: int = 365,
    base_seed: int = 0,
    colonies: list[tuple[str, str, int]] | None = None,
) -> EnsembleResult:
    """Run the simulation n_seeds times, aggregate results."""
    colony_specs = colonies or DEFAULT_COLONIES
    n_colonies = len(colony_specs)
    colony_names = [c[0] for c in colony_specs]
    colony_strategies = [c[1] for c in colony_specs]
    metrics = ["population", "morale", "food_kg", "genetic_diversity", "carrying_capacity"]

    all_series: list[dict[str, list[list[float]]]] = [
        {m: [] for m in metrics} for _ in range(n_colonies)
    ]
    all_final_pops: list[list[int]] = [[] for _ in range(n_colonies)]
    all_growth_pcts: list[list[float]] = [[] for _ in range(n_colonies)]
    alive_counts: list[int] = [0] * n_colonies
    canonical_results = None
    canonical_seed = base_seed + 42

    for i in range(n_seeds):
        env_seed = base_seed + i
        sim = Simulation(sols=sols, env_seed=env_seed, colonies=colony_specs)
        results = sim.run()

        if env_seed == canonical_seed or (canonical_results is None and i == n_seeds // 2):
            canonical_results = results
            canonical_seed = env_seed

        for ci, col_data in enumerate(results["colonies"]):
            history = col_data["history"]
            for m in metrics:
                series = [h.get(m, 0) for h in history]
                while len(series) < sols:
                    series.append(series[-1] if series else 0)
                all_series[ci][m].append(series)
            final_pop = history[-1]["population"] if history else 0
            initial_pop = col_data["initial_population"]
            all_final_pops[ci].append(final_pop)
            growth = (final_pop - initial_pop) / max(1, initial_pop) * 100
            all_growth_pcts[ci].append(growth)
            if final_pop > 0:
                alive_counts[ci] += 1

    bands: list[dict[str, list[list[float]]]] = []
    for ci in range(n_colonies):
        colony_bands: dict[str, list[list[float]]] = {}
        for m in metrics:
            metric_bands: list[list[float]] = []
            for pi, p in enumerate(PERCENTILES):
                sol_vals: list[float] = []
                for sol_idx in range(sols):
                    vals = sorted(s[sol_idx] for s in all_series[ci][m])
                    sol_vals.append(_percentile(vals, p))
                metric_bands.append(sol_vals)
            colony_bands[m] = metric_bands
        bands.append(colony_bands)

    final_pop_stats = []
    growth_pct_stats = []
    survival_rates = []
    for ci in range(n_colonies):
        fps = sorted(all_final_pops[ci])
        gps = sorted(all_growth_pcts[ci])
        final_pop_stats.append({
            "mean": statistics.mean(fps),
            "median": statistics.median(fps),
            "stdev": statistics.stdev(fps) if len(fps) > 1 else 0,
            "p10": _percentile(fps, 10),
            "p90": _percentile(fps, 90),
            "min": min(fps),
            "max": max(fps),
        })
        growth_pct_stats.append({
            "mean": statistics.mean(gps),
            "median": statistics.median(gps),
            "stdev": statistics.stdev(gps) if len(gps) > 1 else 0,
            "p10": _percentile(gps, 10),
            "p90": _percentile(gps, 90),
        })
        survival_rates.append(alive_counts[ci] / n_seeds)

    if canonical_results is None:
        sim = Simulation(sols=sols, env_seed=42)
        canonical_results = sim.run()
        canonical_seed = 42

    return EnsembleResult(
        n_seeds=n_seeds, sols=sols,
        colony_names=colony_names, colony_strategies=colony_strategies,
        bands=bands, final_pop_stats=final_pop_stats,
        growth_pct_stats=growth_pct_stats, survival_rates=survival_rates,
        canonical_results=canonical_results, canonical_seed=canonical_seed,
    )
