"""
Value convergence analysis for Mars-100.

Tracks whether colonist values (stats) converge or diverge over time.
Two views: population-wide and founder-only (controls for births/deaths).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import STAT_NAMES, Colonist


@dataclass
class ConvergenceSnapshot:
    """Convergence data for one year."""
    year: int
    population_dispersion: dict[str, float]
    founder_dispersion: dict[str, float]
    aggregate_dispersion: float
    founder_aggregate: float

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "population_dispersion": self.population_dispersion,
            "founder_dispersion": self.founder_dispersion,
            "aggregate": self.aggregate_dispersion,
            "founder_aggregate": self.founder_aggregate,
        }


def _std_dev(values: list[float]) -> float:
    """Compute population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def compute_convergence(colonists: list[Colonist],
                        founder_ids: set[str],
                        year: int) -> ConvergenceSnapshot:
    """Compute value convergence for all active colonists and founders only."""
    active = [c for c in colonists if c.is_active()]
    founders_active = [c for c in active if c.id in founder_ids]

    pop_disp: dict[str, float] = {}
    founder_disp: dict[str, float] = {}

    for stat_name in STAT_NAMES:
        pop_vals = [getattr(c.stats, stat_name) for c in active]
        pop_disp[stat_name] = _std_dev(pop_vals) if pop_vals else 0.0

        founder_vals = [getattr(c.stats, stat_name) for c in founders_active]
        founder_disp[stat_name] = _std_dev(founder_vals) if founder_vals else 0.0

    agg = sum(pop_disp.values()) / max(1, len(pop_disp))
    f_agg = sum(founder_disp.values()) / max(1, len(founder_disp))

    return ConvergenceSnapshot(
        year=year,
        population_dispersion=pop_disp,
        founder_dispersion=founder_disp,
        aggregate_dispersion=agg,
        founder_aggregate=f_agg,
    )


def classify_trend(snapshots: list[ConvergenceSnapshot],
                   window: int = 10) -> str:
    """Classify the convergence trend from recent snapshots.

    Returns 'converging', 'diverging', or 'stable'.
    """
    if len(snapshots) < window:
        return "stable"
    recent = snapshots[-window:]
    first_half = recent[:window // 2]
    second_half = recent[window // 2:]
    avg_first = sum(s.aggregate_dispersion for s in first_half) / len(first_half)
    avg_second = sum(s.aggregate_dispersion for s in second_half) / len(second_half)
    delta = avg_second - avg_first
    if delta < -0.005:
        return "converging"
    elif delta > 0.005:
        return "diverging"
    return "stable"


@dataclass
class ConvergenceTracker:
    """Tracks convergence across the full simulation."""
    snapshots: list[ConvergenceSnapshot] = field(default_factory=list)
    founder_ids: set[str] = field(default_factory=set)

    def record(self, colonists: list[Colonist], year: int) -> ConvergenceSnapshot:
        """Record convergence for one year."""
        snap = compute_convergence(colonists, self.founder_ids, year)
        self.snapshots.append(snap)
        return snap

    def trend(self) -> str:
        """Get overall convergence trend."""
        return classify_trend(self.snapshots)

    def summary(self) -> dict[str, Any]:
        """Generate a summary of convergence over the simulation."""
        if not self.snapshots:
            return {"trend": "no_data", "initial": 0.0, "final": 0.0, "snapshots": 0}
        return {
            "trend": self.trend(),
            "initial": self.snapshots[0].aggregate_dispersion,
            "final": self.snapshots[-1].aggregate_dispersion,
            "founder_initial": self.snapshots[0].founder_aggregate,
            "founder_final": self.snapshots[-1].founder_aggregate,
            "snapshots": len(self.snapshots),
        }

    def to_curve(self) -> list[dict]:
        """Return convergence curve for charting."""
        return [s.to_dict() for s in self.snapshots]
