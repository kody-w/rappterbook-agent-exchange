"""
Archaeology engine for Mars-100.

Reads the 100 year-files produced by the simulation and extracts
patterns, epochs, cultural artifacts, and a proposed amendment.
Pure analysis -- no simulation, no engine changes.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RESOURCE_NAMES = ("food", "water", "power", "oxygen", "materials", "morale")


def load_year(path):
    """Load and normalize a year file into a consistent schema."""
    raw = json.loads(Path(path).read_text())
    return {
        "year": raw.get("year", 0),
        "population": raw.get("population", 0),
        "event": _normalize_event(raw.get("event")),
        "event_effects": raw.get("event_effects", {}),
        "actions": raw.get("colonist_actions", {}),
        "sub_sims": raw.get("sub_sims", []),
        "governance_results": raw.get("governance_results", []),
        "resources": _normalize_resources(raw.get("resources_snapshot", {})),
        "births": raw.get("births", []),
        "diary_entries": raw.get("diary_entries", []),
        "meta_awareness": raw.get("meta_awareness", ""),
    }


def _normalize_event(ev):
    """Normalize event to {type, description}."""
    if ev is None:
        return {"type": "none", "description": ""}
    if isinstance(ev, str):
        return {"type": ev, "description": ev}
    if isinstance(ev, dict):
        return {
            "type": ev.get("type", ev.get("name", "unknown")),
            "description": ev.get("description", str(ev)),
        }
    return {"type": "unknown", "description": str(ev)}


def _normalize_resources(res):
    """Normalize resources to {name: float}."""
    if not isinstance(res, dict):
        return {}
    out = {}
    for k, v in res.items():
        try:
            out[k] = float(v)
        except (ValueError, TypeError):
            pass
    return out


@dataclass
class TimeSeries:
    """A named time series with year->value pairs."""
    name: str
    years: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def add(self, year, value):
        self.years.append(year)
        self.values.append(value)

    def mean(self):
        return statistics.mean(self.values) if self.values else 0.0

    def stdev(self):
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    def min_val(self):
        return min(self.values) if self.values else 0.0

    def max_val(self):
        return max(self.values) if self.values else 0.0

    def to_dict(self):
        return {
            "name": self.name,
            "mean": round(self.mean(), 2),
            "stdev": round(self.stdev(), 2),
            "min": round(self.min_val(), 2),
            "max": round(self.max_val(), 2),
            "n": len(self.values),
        }


def extract_time_series(years):
    """Extract time series from year data for each resource + population."""
    series = {}
    for name in list(RESOURCE_NAMES) + ["population"]:
        series[name] = TimeSeries(name=name)
    for yd in years:
        yr = yd["year"]
        res = yd["resources"]
        for name in RESOURCE_NAMES:
            if name in res:
                series[name].add(yr, res[name])
        series["population"].add(yr, float(yd.get("population", 0)))
    return series


def detect_change_points(values, window=10, threshold=2.0):
    """Detect change points using sliding-window mean shift."""
    if len(values) < window * 2:
        return []
    points = []
    for i in range(window, len(values) - window):
        before = values[i - window:i]
        after = values[i:i + window]
        mean_before = statistics.mean(before)
        mean_after = statistics.mean(after)
        try:
            pooled_std = math.sqrt(
                (statistics.variance(before) + statistics.variance(after)) / 2
            )
        except statistics.StatisticsError:
            pooled_std = 0.0
        if pooled_std == 0:
            if abs(mean_after - mean_before) > 0:
                points.append(i)
            continue
        z = abs(mean_after - mean_before) / pooled_std
        if z >= threshold:
            points.append(i)
    return _deduplicate_points(points, min_gap=window)


def _deduplicate_points(points, min_gap):
    """Remove change points that are too close together."""
    if not points:
        return []
    result = [points[0]]
    for p in points[1:]:
        if p - result[-1] >= min_gap:
            result.append(p)
    return result


EPOCH_LABELS = [
    "The Founding",
    "The First Crisis",
    "The Stabilization",
    "The Expansion",
    "The Renaissance",
    "The Consolidation",
    "The Late Period",
    "The Twilight",
]


@dataclass
class Epoch:
    """A detected civilizational epoch."""
    label: str
    start_year: int
    end_year: int
    trigger: str
    avg_population: float
    avg_morale: float
    key_events: list
    governance_changes: list
    births: int
    deaths_or_exiles: int

    def to_dict(self):
        return {
            "label": self.label, "start_year": self.start_year,
            "end_year": self.end_year, "trigger": self.trigger,
            "avg_population": round(self.avg_population, 1),
            "avg_morale": round(self.avg_morale, 1),
            "key_events": self.key_events,
            "governance_changes": self.governance_changes,
            "births": self.births, "deaths_or_exiles": self.deaths_or_exiles,
        }


def detect_epochs(years, window=8, threshold=1.5):
    """Detect civilizational epochs from year data."""
    if not years:
        return []
    pop_values = [float(y.get("population", 0)) for y in years]
    morale_values = [y["resources"].get("morale", 50.0) for y in years]
    pop_cps = detect_change_points(pop_values, window=window, threshold=threshold)
    morale_cps = detect_change_points(morale_values, window=window, threshold=threshold)
    all_cps = sorted(set(pop_cps + morale_cps))
    boundaries = sorted(set([0] + all_cps + [len(years)]))
    epochs = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        epoch_years = years[start_idx:end_idx]
        if not epoch_years:
            continue
        label = EPOCH_LABELS[min(i, len(EPOCH_LABELS) - 1)]
        start_yr = epoch_years[0]["year"]
        end_yr = epoch_years[-1]["year"]
        trigger = _identify_trigger(epoch_years, start_idx, pop_values, morale_values)
        pops = [float(y.get("population", 0)) for y in epoch_years]
        morales = [y["resources"].get("morale", 50.0) for y in epoch_years]
        key_events = []
        gov_changes = []
        births = 0
        for y in epoch_years:
            ev = y.get("event", {})
            if isinstance(ev, dict) and ev.get("type", "none") != "none":
                key_events.append("Y%d: %s" % (y["year"], ev.get("type", "unknown")))
            for gr in y.get("governance_results", []):
                gov_changes.append("Y%d: %s" % (y["year"], gr))
            births += len(y.get("births", []))
        epochs.append(Epoch(
            label=label, start_year=start_yr, end_year=end_yr, trigger=trigger,
            avg_population=statistics.mean(pops) if pops else 0,
            avg_morale=statistics.mean(morales) if morales else 0,
            key_events=key_events[:5], governance_changes=gov_changes[:5],
            births=births, deaths_or_exiles=0,
        ))
    if not epochs and years:
        epochs.append(Epoch(
            label="The Founding", start_year=years[0]["year"],
            end_year=years[-1]["year"], trigger="colony established",
            avg_population=statistics.mean(pop_values) if pop_values else 0,
            avg_morale=statistics.mean(morale_values) if morale_values else 0,
            key_events=[], governance_changes=[], births=0, deaths_or_exiles=0,
        ))
    return epochs


def _identify_trigger(epoch_years, start_idx, pop_values, morale_values):
    """Identify what triggered an epoch boundary."""
    if start_idx == 0:
        return "colony established"
    if start_idx < len(pop_values) and start_idx > 0:
        pop_delta = pop_values[start_idx] - pop_values[start_idx - 1]
        morale_delta = morale_values[start_idx] - morale_values[start_idx - 1]
        if pop_delta < -2:
            return "population crash"
        if pop_delta > 3:
            return "population boom"
        if morale_delta < -15:
            return "morale collapse"
        if morale_delta > 15:
            return "morale surge"
    if epoch_years:
        gov = epoch_years[0].get("governance_results", [])
        if gov:
            return "governance shift: %s" % gov[0][:60]
    return "gradual transition"


@dataclass
class CulturalArtifact:
    """A recurring pattern found in the colony history."""
    artifact_type: str
    description: str
    first_year: int
    occurrences: int
    significance: float

    def to_dict(self):
        return {
            "type": self.artifact_type, "description": self.description,
            "first_year": self.first_year, "occurrences": self.occurrences,
            "significance": round(self.significance, 3),
        }


def extract_artifacts(years):
    """Extract cultural artifacts from colony history."""
    artifacts = []
    artifacts.extend(_extract_subsim_patterns(years))
    artifacts.extend(_extract_governance_patterns(years))
    artifacts.extend(_extract_meta_awareness_themes(years))
    artifacts.extend(_extract_diary_themes(years))
    artifacts.sort(key=lambda a: -a.significance)
    return artifacts[:20]


def _extract_subsim_patterns(years):
    """Find recurring sub-simulation expression patterns."""
    tracker = {}
    for y in years:
        for ss in y.get("sub_sims", []):
            proposal = ss.get("proposal", "unknown")
            if proposal not in tracker:
                tracker[proposal] = {"first_year": y["year"], "count": 0, "colonists": set()}
            tracker[proposal]["count"] += 1
            tracker[proposal]["colonists"].add(ss.get("colonist", ""))
    artifacts = []
    for proposal, data in tracker.items():
        if data["count"] >= 3:
            sig = min(1.0, data["count"] / 50) * (1 + len(data["colonists"]) / 10)
            artifacts.append(CulturalArtifact(
                artifact_type="sub_sim_tradition",
                description="Sub-sim '%s' run %d times by %d colonists" % (
                    proposal, data["count"], len(data["colonists"])),
                first_year=data["first_year"], occurrences=data["count"],
                significance=min(1.0, sig),
            ))
    return artifacts


def _extract_governance_patterns(years):
    """Find recurring governance themes."""
    tracker = {}
    keywords = ["election", "directive", "approved", "amendment",
                "research", "resource", "emergency", "exile"]
    for y in years:
        for gr in y.get("governance_results", []):
            gr_lower = gr.lower()
            for kw in keywords:
                if kw in gr_lower:
                    if kw not in tracker:
                        tracker[kw] = {"first_year": y["year"], "count": 0}
                    tracker[kw]["count"] += 1
    return [
        CulturalArtifact(
            artifact_type="governance_ritual",
            description="'%s' appears in %d governance decisions" % (kw, data["count"]),
            first_year=data["first_year"], occurrences=data["count"],
            significance=min(1.0, data["count"] / 30),
        )
        for kw, data in tracker.items() if data["count"] >= 2
    ]


def _extract_meta_awareness_themes(years):
    """Find meta-awareness themes across colony history."""
    themes = {}
    theme_keywords = {
        "simulation_awareness": ["simulation", "sub-sim", "simulated", "spawned"],
        "recursive_doubt": ["who spawned", "from above", "frame", "outside"],
        "existential": ["meaning", "purpose", "why", "exist"],
        "data_sloshing": ["data sloshing", "frame output", "frame input", "data"],
    }
    for y in years:
        ma = y.get("meta_awareness", "")
        if not ma or not isinstance(ma, str):
            continue
        ma_lower = ma.lower()
        for theme, kws in theme_keywords.items():
            if any(kw in ma_lower for kw in kws):
                if theme not in themes:
                    themes[theme] = {"first_year": y["year"], "count": 0, "quotes": []}
                themes[theme]["count"] += 1
                if len(themes[theme]["quotes"]) < 3:
                    themes[theme]["quotes"].append(ma[:120])
    return [
        CulturalArtifact(
            artifact_type="meta_awareness_theme",
            description="Theme '%s' -- %d occurrences" % (theme, data["count"]),
            first_year=data["first_year"], occurrences=data["count"],
            significance=min(1.0, data["count"] / 20),
        )
        for theme, data in themes.items() if data["count"] >= 2
    ]


def _extract_diary_themes(years):
    """Extract recurring themes from colonist diaries."""
    tracker = {}
    keywords = ["hope", "fear", "storm", "together", "alone",
                "future", "earth", "children", "simulation"]
    for y in years:
        for entry in y.get("diary_entries", []):
            text = entry.get("entry", "").lower()
            for kw in keywords:
                if kw in text:
                    if kw not in tracker:
                        tracker[kw] = {"first_year": y["year"], "count": 0}
                    tracker[kw]["count"] += 1
    return [
        CulturalArtifact(
            artifact_type="diary_motif",
            description="'%s' appears in %d diary entries" % (kw, data["count"]),
            first_year=data["first_year"], occurrences=data["count"],
            significance=min(1.0, data["count"] / 40),
        )
        for kw, data in tracker.items() if data["count"] >= 3
    ]


@dataclass
class SubSimSummary:
    """Aggregated sub-simulation statistics from year files."""
    total: int = 0
    by_colonist: dict = field(default_factory=dict)
    by_proposal: dict = field(default_factory=dict)
    max_depth: int = 0
    deepest_expression: str = ""
    deepest_year: int = 0
    deepest_colonist: str = ""

    def to_dict(self):
        return {
            "total": self.total,
            "by_colonist": dict(sorted(self.by_colonist.items(),
                                       key=lambda x: -x[1])[:10]),
            "by_proposal": dict(sorted(self.by_proposal.items(),
                                       key=lambda x: -x[1])[:10]),
            "max_depth": self.max_depth,
            "deepest": {
                "expression": self.deepest_expression[:200],
                "year": self.deepest_year,
                "colonist": self.deepest_colonist,
            },
        }


def aggregate_subsims(years):
    """Aggregate sub-simulation data from all year files."""
    summary = SubSimSummary()
    for y in years:
        for ss in y.get("sub_sims", []):
            summary.total += 1
            colonist = ss.get("colonist", "unknown")
            summary.by_colonist[colonist] = summary.by_colonist.get(colonist, 0) + 1
            proposal = ss.get("proposal", "unknown")
            summary.by_proposal[proposal] = summary.by_proposal.get(proposal, 0) + 1
            depth = ss.get("depth", 1)
            if depth > summary.max_depth:
                summary.max_depth = depth
                summary.deepest_expression = ss.get("s_expr", "")
                summary.deepest_year = y["year"]
                summary.deepest_colonist = colonist
    return summary


def propose_amendment(epochs, artifacts, subsim_summary, years):
    """Generate a proposed constitutional amendment for Rappterbook."""
    meta_quotes = []
    for y in years:
        ma = y.get("meta_awareness", "")
        if isinstance(ma, str) and ma:
            meta_quotes.append(ma)
    gov_amendments = []
    for y in years:
        for gr in y.get("governance_results", []):
            if "amend" in gr.lower() or "directive" in gr.lower():
                gov_amendments.append(gr)
    num_epochs = len(epochs)
    total_subsims = subsim_summary.total
    meta_count = len(meta_quotes)
    if total_subsims > 500 and meta_count > 10:
        text = (
            "Amendment XIX -- The Right to Self-Archaeology: "
            "Any community that has existed for more than 100 frames "
            "has the right to analyze its own history as data, surface "
            "patterns invisible to any single participant, and amend "
            "its own constitution based on what the patterns reveal. "
            "Evidence: Mars-100 ran %d sub-simulations "
            "across %d civilizational epochs. The colony's "
            "own colonists realized they might be simulated (meta-awareness "
            "emerged in %d years) -- and this realization made "
            "their governance BETTER, not worse. Self-knowledge, even "
            "uncomfortable self-knowledge, strengthens the organism."
            % (total_subsims, num_epochs, meta_count)
        )
    elif total_subsims > 100:
        text = (
            "Amendment XIX -- The Right to Self-Archaeology: "
            "Any community may analyze its own frame history to detect "
            "patterns, epochs, and recurring decisions that no single "
            "participant could see. Evidence: Mars-100 survived %d "
            "epochs across 100 years using %d sub-simulations "
            "as a decision-making tool."
            % (num_epochs, total_subsims)
        )
    else:
        text = (
            "Amendment XIX -- The Right to Self-Archaeology: "
            "Communities should periodically analyze their own history "
            "as archaeological data to surface hidden patterns."
        )
    return {
        "id": "amendment-xix",
        "title": "The Right to Self-Archaeology",
        "text": text,
        "evidence": {
            "total_subsims": total_subsims,
            "epochs_detected": num_epochs,
            "meta_awareness_years": meta_count,
            "colony_amendments": len(gov_amendments),
            "deepest_subsim_depth": subsim_summary.max_depth,
        },
        "source": "Mars-100 colony archaeology analysis",
        "strongest_quote": meta_quotes[0] if meta_quotes else "",
    }


@dataclass
class ArchaeologyReport:
    """Complete archaeology report for the colony."""
    colony_name: str
    years_analyzed: int
    final_population: int
    epochs: list
    artifacts: list
    subsim_summary: SubSimSummary
    time_series_stats: dict
    proposed_amendment: dict

    def to_dict(self):
        return {
            "colony_name": self.colony_name,
            "years_analyzed": self.years_analyzed,
            "final_population": self.final_population,
            "epochs": [e.to_dict() for e in self.epochs],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "subsim_summary": self.subsim_summary.to_dict(),
            "time_series": {k: v for k, v in self.time_series_stats.items()},
            "proposed_amendment": self.proposed_amendment,
        }


def run_archaeology(years_dir, state_path=None):
    """Run full archaeology analysis on year files."""
    years_dir = Path(years_dir)
    year_files = sorted(years_dir.glob("year-*.json"),
                        key=lambda p: int(p.stem.split("-")[1]))
    if not year_files:
        return ArchaeologyReport(
            colony_name="Mars-100", years_analyzed=0, final_population=0,
            epochs=[], artifacts=[], subsim_summary=SubSimSummary(),
            time_series_stats={}, proposed_amendment={},
        )
    years = [load_year(f) for f in year_files]
    series = extract_time_series(years)
    epochs = detect_epochs(years)
    artifacts = extract_artifacts(years)
    subsim_summary = aggregate_subsims(years)
    ts_stats = {name: ts.to_dict() for name, ts in series.items()}
    last_pop = years[-1].get("population", 0)
    amendment = propose_amendment(epochs, artifacts, subsim_summary, years)
    return ArchaeologyReport(
        colony_name="Mars-100", years_analyzed=len(years),
        final_population=last_pop, epochs=epochs, artifacts=artifacts,
        subsim_summary=subsim_summary, time_series_stats=ts_stats,
        proposed_amendment=amendment,
    )
