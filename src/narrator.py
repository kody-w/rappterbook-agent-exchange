"""
narrator.py — Turn simulation results into a narrative chronicle.

Reads the raw dict from tick_engine.Simulation.results() and produces
a Chronicle: a sequence of annotated events with an overall story arc.

Usage:
    from src.narrator import narrate, format_chronicle
    sim = Simulation(sols=365)
    results = sim.run()
    chronicle = narrate(results)
    print(format_chronicle(chronicle))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ─── Arc classification thresholds ───

ARC_TRIUMPH_GROWTH = 0.50      # 50%+ total pop growth → triumph
ARC_COLLAPSE_GROWTH = -0.20    # 20%+ total pop loss → collapse
ARC_STAGNATION_BAND = 0.05     # ±5% → stagnation


@dataclass
class Event:
    """A single narrative event extracted from simulation history."""

    sol: int
    kind: str           # "milestone", "epidemic", "tech", "storm", "death_spike", "terraform", "migration"
    colony: str | None  # None for planet-wide events
    headline: str
    detail: str
    magnitude: float    # 0.0–1.0 importance score


@dataclass
class Chronicle:
    """Full narrative output — events + arc + summary."""

    arc: str                # "triumph", "survival", "collapse", "stagnation"
    arc_reason: str         # one-sentence explanation
    events: list[Event] = field(default_factory=list)
    opening: str = ""       # narrative opening paragraph
    closing: str = ""       # narrative closing paragraph
    sol_count: int = 0
    colony_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "arc": self.arc,
            "arc_reason": self.arc_reason,
            "opening": self.opening,
            "closing": self.closing,
            "sol_count": self.sol_count,
            "colony_count": self.colony_count,
            "events": [
                {
                    "sol": e.sol,
                    "kind": e.kind,
                    "colony": e.colony,
                    "headline": e.headline,
                    "detail": e.detail,
                    "magnitude": round(e.magnitude, 3),
                }
                for e in self.events
            ],
        }


# ─── Event extraction ───

def _extract_population_milestones(colonies: list[dict]) -> list[Event]:
    """Find population milestone crossings (100, 200, 500, 1000)."""
    milestones = [100, 200, 500, 1000]
    events: list[Event] = []
    for col in colonies:
        name = col["name"]
        prev_pop = 0
        for h in col.get("history", []):
            pop = h.get("population", 0)
            sol = h.get("sol", 0)
            for m in milestones:
                if prev_pop < m <= pop:
                    events.append(Event(
                        sol=sol,
                        kind="milestone",
                        colony=name,
                        headline="%s reaches %d colonists" % (name, m),
                        detail="Population crossed %d on sol %d" % (m, sol),
                        magnitude=min(1.0, m / 1000),
                    ))
            prev_pop = pop
    return events


def _extract_epidemics(colonies: list[dict]) -> list[Event]:
    """Extract epidemic outbreak events."""
    events: list[Event] = []
    for col in colonies:
        name = col["name"]
        for ev in col.get("events", []):
            if ev.get("type") in ("epidemic", "epidemic_spread"):
                strain = ev.get("strain", "unknown")
                sol = ev.get("sol", 0)
                is_spread = ev.get("type") == "epidemic_spread"
                if is_spread:
                    headline = "%s: %s spreads from %s" % (name, strain, ev.get("from", "?"))
                else:
                    headline = "%s: %s outbreak" % (name, strain)
                events.append(Event(
                    sol=sol,
                    kind="epidemic",
                    colony=name,
                    headline=headline,
                    detail="Strain: %s (sol %d)" % (strain, sol),
                    magnitude=0.7 if strain == "Rad Fever" else 0.4,
                ))
    return events


def _extract_tech_unlocks(colonies: list[dict]) -> list[Event]:
    """Extract technology unlock events."""
    events: list[Event] = []
    for col in colonies:
        name = col["name"]
        tech = col.get("tech", {})
        for unlock in tech.get("unlocked", []):
            events.append(Event(
                sol=unlock["sol"],
                kind="tech",
                colony=name,
                headline="%s unlocks %s" % (name, unlock["name"]),
                detail="%s [%s branch]" % (unlock.get("description", ""), unlock.get("branch", "")),
                magnitude=0.5,
            ))
    return events


def _extract_storms(env_history: list[dict]) -> list[Event]:
    """Extract significant dust storm events."""
    events: list[Event] = []
    in_storm = False
    storm_start = 0
    for snap in env_history:
        sol = snap.get("sol", 0)
        dust = snap.get("dust_opacity", 0.0)
        storm = snap.get("storm_kind")
        if storm and not in_storm:
            in_storm = True
            storm_start = sol
            is_global = storm == "global"
            events.append(Event(
                sol=sol,
                kind="storm",
                colony=None,
                headline="%s dust storm begins (sol %d)" % (
                    "Global" if is_global else "Regional", sol),
                detail="Dust opacity: %.2f" % dust,
                magnitude=0.9 if is_global else 0.3,
            ))
        elif not storm and in_storm:
            in_storm = False
    return events


def _extract_death_spikes(colonies: list[dict]) -> list[Event]:
    """Find sols where deaths spike above baseline."""
    events: list[Event] = []
    for col in colonies:
        name = col["name"]
        history = col.get("history", [])
        if len(history) < 10:
            continue
        deaths = [h.get("deaths", 0) for h in history]
        mean_deaths = sum(deaths) / max(len(deaths), 1)
        threshold = max(3, mean_deaths * 3)
        for h in history:
            if h.get("deaths", 0) >= threshold:
                events.append(Event(
                    sol=h.get("sol", 0),
                    kind="death_spike",
                    colony=name,
                    headline="%s loses %d colonists (sol %d)" % (name, h["deaths"], h.get("sol", 0)),
                    detail="%.1fx above average death rate" % (h["deaths"] / max(mean_deaths, 0.01)),
                    magnitude=min(1.0, h["deaths"] / 20),
                ))
    return events


def _extract_terraforming(results: dict) -> list[Event]:
    """Extract terraforming phase transitions."""
    events: list[Event] = []
    env = results.get("environment", {})
    phase = env.get("terraform_phase", "none")
    progress = env.get("final_terraforming_progress", 0.0)
    if progress > 0.01:
        events.append(Event(
            sol=results.get("_meta", {}).get("sols", 0),
            kind="terraform",
            colony=None,
            headline="Terraforming reaches %.1f%% — %s" % (progress * 100, phase),
            detail="Phase: %s, progress: %.4f" % (phase, progress),
            magnitude=min(1.0, progress * 3),
        ))
    return events


def _extract_migrations(colonies: list[dict]) -> list[Event]:
    """Extract migration/evacuation events."""
    events: list[Event] = []
    for col in colonies:
        name = col["name"]
        for ev in col.get("events", []):
            if ev.get("type") in ("evacuation", "migration"):
                events.append(Event(
                    sol=ev.get("sol", 0),
                    kind="migration",
                    colony=name,
                    headline="%s: %d colonists evacuate to %s" % (
                        name, ev.get("count", 0), ev.get("to", "?")),
                    detail="Emergency evacuation (sol %d)" % ev.get("sol", 0),
                    magnitude=min(1.0, ev.get("count", 0) / 30),
                ))
    return events


# ─── Arc classification ───

def _classify_arc(results: dict) -> tuple[str, str]:
    """Determine the overall story arc from final results.

    Returns (arc_name, reason).
    """
    summary = results.get("summary", {})
    colonies = summary.get("colonies", [])
    if not colonies:
        return ("stagnation", "No colony data available")

    total_start = sum(c.get("start_pop", 0) for c in colonies)
    total_end = sum(c.get("end_pop", 0) for c in colonies)
    if total_start == 0:
        return ("collapse", "Colonies started with zero population")

    growth = (total_end - total_start) / total_start

    dead_colonies = sum(1 for c in colonies if c.get("end_pop", 0) == 0)
    total_deaths = sum(c.get("total_deaths", 0) for c in colonies)
    total_births = sum(c.get("total_births", 0) for c in colonies)

    if dead_colonies > 0:
        return ("collapse", "%d of %d colonies perished" % (dead_colonies, len(colonies)))
    if growth >= ARC_TRIUMPH_GROWTH:
        return ("triumph", "Population grew %.0f%% — all colonies thriving" % (growth * 100))
    if growth <= ARC_COLLAPSE_GROWTH:
        return ("collapse", "Population declined %.0f%%" % (abs(growth) * 100))
    if abs(growth) <= ARC_STAGNATION_BAND:
        return ("stagnation", "Population barely changed (%.1f%%)" % (growth * 100))

    if total_deaths > total_births * 0.7:
        return ("survival", "Deaths nearly matched births — a hard-won hold on Mars")
    return ("survival", "Modest growth (%.0f%%) against harsh conditions" % (growth * 100))


# ─── Narrative generation ───

def _generate_opening(results: dict, arc: str) -> str:
    """Generate the opening paragraph based on arc and initial conditions."""
    meta = results.get("_meta", {})
    sols = meta.get("sols", 0)
    summary = results.get("summary", {})
    colonies = summary.get("colonies", [])
    n_colonies = len(colonies)
    total_start = sum(c.get("start_pop", 0) for c in colonies)

    openers = {
        "triumph": "Sol 1. %d colonists across %d settlements begin humanity's greatest experiment. What follows is %d sols of relentless growth against impossible odds." % (total_start, n_colonies, sols),
        "survival": "Sol 1. %d souls. %d fragile habitats on a frozen, irradiated world. Over %d sols, they fight to hold the line." % (total_start, n_colonies, sols),
        "collapse": "Sol 1. Hope burns bright — %d colonists, %d colonies, a new world. But Mars does not forgive mistakes." % (total_start, n_colonies),
        "stagnation": "Sol 1. %d colonists settle into %d outposts. The days pass. Mars is patient. So are they." % (total_start, n_colonies),
    }
    return openers.get(arc, "Sol 1. The simulation begins.")


def _generate_closing(results: dict, arc: str) -> str:
    """Generate the closing paragraph."""
    summary = results.get("summary", {})
    colonies = summary.get("colonies", [])
    total_end = sum(c.get("end_pop", 0) for c in colonies)
    tf = summary.get("terraforming", {})
    progress = tf.get("progress", 0)
    phase = tf.get("phase", "none")

    closings = {
        "triumph": "The experiment succeeded. %d colonists now call Mars home. Terraforming: %.1f%% (%s). The red planet is turning green." % (total_end, progress * 100, phase),
        "survival": "%d colonists remain. Not thriving — surviving. Terraforming sits at %.1f%%. Mars gives nothing freely." % (total_end, progress * 100),
        "collapse": "Only %d remain. The dream is not dead, but it is wounded. Terraforming stalled at %.1f%%." % (total_end, progress * 100),
        "stagnation": "%d colonists. Same as yesterday. Same as tomorrow. Mars waits." % total_end,
    }
    return closings.get(arc, "The simulation ends with %d colonists." % total_end)


def narrate(results: dict) -> Chronicle:
    """Transform raw simulation results into a narrative Chronicle.

    This is the main entry point. Takes the dict from
    Simulation.results() and returns a Chronicle with
    extracted events, story arc, and narrative framing.
    """
    colonies = results.get("colonies", [])
    env_history = results.get("environment", {}).get("history", [])

    # Extract all events
    all_events: list[Event] = []
    all_events.extend(_extract_population_milestones(colonies))
    all_events.extend(_extract_epidemics(colonies))
    all_events.extend(_extract_tech_unlocks(colonies))
    all_events.extend(_extract_storms(env_history))
    all_events.extend(_extract_death_spikes(colonies))
    all_events.extend(_extract_terraforming(results))
    all_events.extend(_extract_migrations(colonies))

    # Sort by sol, then by magnitude (most important first within a sol)
    all_events.sort(key=lambda e: (e.sol, -e.magnitude))

    # Classify the story arc
    arc, arc_reason = _classify_arc(results)

    # Generate narrative framing
    opening = _generate_opening(results, arc)
    closing = _generate_closing(results, arc)

    return Chronicle(
        arc=arc,
        arc_reason=arc_reason,
        events=all_events,
        opening=opening,
        closing=closing,
        sol_count=results.get("_meta", {}).get("sols", 0),
        colony_count=len(colonies),
    )


def format_chronicle(chronicle: Chronicle) -> str:
    """Format a Chronicle as human-readable text.

    Produces a compact multi-section narrative suitable for
    terminal output or Discussion posts.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  MARS BARN CHRONICLE")
    lines.append("  Arc: %s — %s" % (chronicle.arc.upper(), chronicle.arc_reason))
    lines.append("=" * 60)
    lines.append("")
    lines.append(chronicle.opening)
    lines.append("")

    # Group events by kind for a structured narrative
    if chronicle.events:
        lines.append("-" * 40)
        lines.append("  KEY EVENTS (%d total)" % len(chronicle.events))
        lines.append("-" * 40)

        # Show top events by magnitude (cap at 20 to keep it readable)
        top_events = sorted(chronicle.events, key=lambda e: -e.magnitude)[:20]
        for ev in sorted(top_events, key=lambda e: e.sol):
            colony_tag = " [%s]" % ev.colony if ev.colony else ""
            lines.append("  Sol %4d  %-12s%s" % (ev.sol, ev.kind, colony_tag))
            lines.append("           %s" % ev.headline)

    lines.append("")
    lines.append("-" * 40)
    lines.append(chronicle.closing)
    lines.append("=" * 60)

    return "\n".join(lines)
