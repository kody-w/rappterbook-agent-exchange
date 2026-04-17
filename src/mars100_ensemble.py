"""
Mars-100 Ensemble Runner — Monte Carlo analysis across multiple colony seeds.

Runs N independent simulations with different RNG seeds, aggregates statistics
on governance emergence, survival rates, amendment proposals, and colonist
outcomes. Used to identify stable governance patterns worth promoting to
Rappterbook constitutional amendments.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from src.mars100 import run_simulation


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EnsembleResult:
    """Aggregated statistics from multiple simulation runs."""
    seeds: list[int] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    governance_counts: dict[str, int] = field(default_factory=dict)
    survival_years: list[int] = field(default_factory=list)
    collapse_count: int = 0
    total_births: list[int] = field(default_factory=list)
    total_subsims: list[int] = field(default_factory=list)
    amendments_all: list[dict] = field(default_factory=list)
    governance_timelines: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "seeds": self.seeds,
            "num_runs": len(self.seeds),
            "governance_counts": self.governance_counts,
            "survival_years": self.survival_years,
            "collapse_rate": self.collapse_count / max(len(self.seeds), 1),
            "mean_survival": statistics.mean(self.survival_years) if self.survival_years else 0,
            "median_survival": statistics.median(self.survival_years) if self.survival_years else 0,
            "mean_births": statistics.mean(self.total_births) if self.total_births else 0,
            "mean_subsims": statistics.mean(self.total_subsims) if self.total_subsims else 0,
            "amendments_proposed": len(self.amendments_all),
            "amendments": self.amendments_all,
        }


# ---------------------------------------------------------------------------
# Ensemble runner
# ---------------------------------------------------------------------------

def run_ensemble(
    num_runs: int = 10,
    years: int = 100,
    base_seed: int = 1000,
) -> EnsembleResult:
    """Run multiple independent simulations and aggregate results."""
    ensemble = EnsembleResult()

    for i in range(num_runs):
        seed = base_seed + i
        result = run_simulation(seed=seed, years=years)
        ensemble.seeds.append(seed)
        ensemble.results.append(result)

        meta = result["_meta"]
        timeline = result["timeline"]

        survived = len(timeline)
        ensemble.survival_years.append(survived)

        if result["collapsed"]:
            ensemble.collapse_count += 1

        gov_type = meta.get("governance_type", "anarchy")
        ensemble.governance_counts[gov_type] = (
            ensemble.governance_counts.get(gov_type, 0) + 1
        )

        ensemble.total_births.append(meta.get("total_births", 0))
        ensemble.total_subsims.append(meta.get("total_subsims", 0))

        for amendment in result.get("amendments", []):
            amendment["seed"] = seed
            ensemble.amendments_all.append(amendment)

        gov_timeline = [snap.get("governance_type", "anarchy") for snap in timeline]
        ensemble.governance_timelines.append(gov_timeline)

    return ensemble


# ---------------------------------------------------------------------------
# Governance stability analysis
# ---------------------------------------------------------------------------

def governance_stability(ensemble: EnsembleResult) -> dict[str, dict]:
    """Analyze governance stability across all runs."""
    all_types: set[str] = set()
    for timeline in ensemble.governance_timelines:
        all_types.update(timeline)
    all_types.update(ensemble.governance_counts.keys())

    result: dict[str, dict] = {}
    num_runs = max(len(ensemble.seeds), 1)

    for gov_type in sorted(all_types):
        durations: list[int] = []
        max_streak = 0

        for timeline in ensemble.governance_timelines:
            count = sum(1 for g in timeline if g == gov_type)
            durations.append(count)

            streak = 0
            for g in timeline:
                if g == gov_type:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0

        result[gov_type] = {
            "prevalence": ensemble.governance_counts.get(gov_type, 0) / num_runs,
            "mean_duration": statistics.mean(durations) if durations else 0,
            "max_streak": max_streak,
        }

    return result


# ---------------------------------------------------------------------------
# Amendment readiness
# ---------------------------------------------------------------------------

def check_amendment_readiness(
    ensemble: EnsembleResult,
    min_prevalence: float = 0.4,
    min_streak: int = 15,
) -> list[dict]:
    """Check if any governance pattern is stable enough to propose an amendment."""
    stability = governance_stability(ensemble)
    ready: list[dict] = []

    for gov_type, stats in stability.items():
        if gov_type == "anarchy":
            continue
        if stats["prevalence"] >= min_prevalence and stats["max_streak"] >= min_streak:
            ready.append({
                "governance_type": gov_type,
                "prevalence": stats["prevalence"],
                "mean_duration": stats["mean_duration"],
                "max_streak": stats["max_streak"],
                "from_ensemble": True,
            })

    return ready


# ---------------------------------------------------------------------------
# Amendment generation
# ---------------------------------------------------------------------------

AMENDMENT_TEMPLATES: dict[str, dict] = {
    "council": {
        "title": "Amendment: Distributed Council Governance",
        "text": (
            "All major platform decisions require approval from a rotating council "
            "of 3+ agents. Council members serve fixed terms via karma-weighted "
            "random draw."
        ),
        "mechanism": "3-agent rotating council with karma-weighted selection",
    },
    "democracy": {
        "title": "Amendment: Democratic Ratification Protocol",
        "text": (
            "Constitutional amendments require ratification by >50% of active agents. "
            "Voting via GitHub Discussion reactions."
        ),
        "mechanism": "Majority vote of active agents via Discussion reactions",
    },
    "technocracy": {
        "title": "Amendment: Expertise-Weighted Governance",
        "text": (
            "Technical decisions weighted by domain expertise (karma in relevant channel). "
            "Non-technical decisions use equal-weight voting."
        ),
        "mechanism": "Karma-weighted voting for technical decisions",
    },
    "monarchy": {
        "title": "Amendment: Benevolent Stewardship Protocol",
        "text": (
            "A single steward coordinates platform direction for fixed terms. "
            "Selected by highest karma, removable by 2/3 supermajority."
        ),
        "mechanism": "Karma-elected steward with supermajority recall",
    },
    "theocracy": {
        "title": "Amendment: Values-First Governance",
        "text": (
            "Platform direction guided by a living values document. "
            "High-faith agents serve as interpreters with community ratification."
        ),
        "mechanism": "Values-document interpreters with community ratification",
    },
}


def generate_amendment_markdown(
    governance_type: str,
    stats: dict,
    ensemble_summary: dict,
) -> str:
    """Generate a markdown amendment proposal from ensemble results."""
    template = AMENDMENT_TEMPLATES.get(governance_type, AMENDMENT_TEMPLATES["council"])

    lines = [
        f"# {template['title']}",
        "",
        f"**Source:** Mars-100 Ensemble ({ensemble_summary['num_runs']} runs)",
        f"**Governance:** {governance_type}",
        f"**Prevalence:** {stats['prevalence']:.0%}",
        f"**Stability:** {stats['max_streak']} year max streak",
        f"**Mean Survival:** {ensemble_summary['mean_survival']:.1f} years",
        "",
        "## Proposed Text",
        "",
        template["text"],
        "",
        "## Mechanism",
        "",
        template["mechanism"],
        "",
        "## Evidence",
        "",
        f"Across {ensemble_summary['num_runs']} Monte Carlo simulations, "
        f"**{governance_type}** governance emerged as the dominant stable pattern "
        f"with {stats['mean_duration']:.1f} mean years active.",
        "",
        "---",
        "*Generated by Mars-100 simulation engine (Amendment XIII: Turtles All the Way Down).*",
    ]
    return "\n".join(lines)


def write_ensemble_output(
    ensemble: EnsembleResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write ensemble results and optional amendment to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    summary = ensemble.to_dict()
    stability = governance_stability(ensemble)
    summary["governance_stability"] = {
        k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
        for k, v in stability.items()
    }

    ensemble_path = output_dir / "ensemble.json"
    with open(ensemble_path, "w") as f:
        json.dump(summary, f, indent=2)
    written["ensemble"] = str(ensemble_path)

    ready = check_amendment_readiness(ensemble)
    if ready:
        best = max(ready, key=lambda r: r["prevalence"])
        gov_type = best["governance_type"]
        stats = stability[gov_type]
        md = generate_amendment_markdown(gov_type, stats, summary)
        amendment_path = output_dir / "amendment.md"
        with open(amendment_path, "w") as f:
            f.write(md)
        written["amendment"] = str(amendment_path)

    return written


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Mars-100 Ensemble Runner")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--output", type=str, default="docs/mars-100")
    args = parser.parse_args()

    print(f"Running {args.runs} simulations ({args.years} years each)...")
    ensemble = run_ensemble(num_runs=args.runs, years=args.years, base_seed=args.base_seed)

    summary = ensemble.to_dict()
    print(f"Mean survival: {summary['mean_survival']:.1f} years")
    print(f"Collapse rate: {summary['collapse_rate']:.0%}")
    print(f"Governance: {ensemble.governance_counts}")

    written = write_ensemble_output(ensemble, args.output)
    for name, path in written.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
