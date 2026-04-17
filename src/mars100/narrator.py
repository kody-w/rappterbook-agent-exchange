"""
Narrator for Mars-100.

Generates human-readable diary entries, year summaries, and the final
emergent governance report.
"""
from __future__ import annotations

import random
from typing import Any


def narrate_year(year_result: dict, rng: random.Random) -> str:
    """Generate a narrative summary for one year."""
    year = year_result["year"]
    events = year_result.get("events", [])
    actions = year_result.get("actions", {})
    deaths = year_result.get("deaths", [])
    exiles = year_result.get("exiles", [])
    meta = year_result.get("meta_awareness", [])
    governance = year_result.get("governance")
    resources = year_result.get("resources_after", {})
    cohesion = year_result.get("social_cohesion", 0.5)
    subsims = year_result.get("subsim_log", [])

    lines: list = []
    lines.append(f"## Year {year}")
    lines.append("")

    if events:
        lines.append("### Events")
        for ev in events:
            sev = ev.get("severity", 0)
            tag = "RED" if sev > 0.6 else "YEL" if sev > 0.3 else "GRN"
            lines.append(f"- [{tag}] **{ev['name']}**: {ev['description']}")
        lines.append("")

    if actions:
        counts: dict = {}
        for a in actions.values():
            counts[a] = counts.get(a, 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])
        lines.append(
            f"**Activity:** {', '.join(f'{a}({n})' for a, n in top[:5])}")

    res_strs = []
    for name, val in resources.items():
        tag = "OK" if val > 0.5 else "LOW" if val > 0.2 else "CRIT"
        res_strs.append(f"{name}:{val:.0%}[{tag}]")
    lines.append(f"**Resources:** {' | '.join(res_strs)}")
    lines.append(f"**Cohesion:** {cohesion:.0%}")

    if subsims:
        lines.append(f"\n### Sub-simulations ({len(subsims)})")
        for ss in subsims[:3]:
            depth = ss.get("depth", 1)
            res = str(ss.get("result", "err"))[:40]
            lines.append(f"- depth-{depth} {ss['colonist_id']}: {res}")

    if governance:
        passed = "PASSED" if governance.get("passed") else "REJECTED"
        lines.append(
            f"\n### Governance: {governance['gov_type']} [{passed}]")
        vf = len(governance.get("votes_for", []))
        va = len(governance.get("votes_against", []))
        lines.append(f"Vote: {vf} for / {va} against")

    for d in deaths:
        lines.append(f"\n**DEATH: {d['name']} \u2014 {d['cause']}**")
    for e in exiles:
        lines.append(f"\n**EXILE: {e['name']}**")
    for m in meta:
        lines.append(f"\n*META: {m['insight']}*")

    lines.append("\n---")
    return "\n".join(lines)


def generate_diary_entries(year_result: dict,
                           colonist_snapshots: list,
                           rng: random.Random,
                           count: int = 3) -> list:
    """Generate diary entries from random active colonists."""
    active = [c for c in colonist_snapshots
              if c.get("alive") and not c.get("exiled")]
    if not active:
        return []
    selected = rng.sample(active, min(count, len(active)))
    entries: list = []
    events = year_result.get("events", [])
    resources = year_result.get("resources_after", {})

    for colonist in selected:
        name = colonist["name"]
        element = colonist["element"]
        archetype = colonist["archetype"]
        avg_r = sum(resources.values()) / max(1, len(resources))
        mood = ("hopeful" if avg_r > 0.6
                else "anxious" if avg_r > 0.3
                else "desperate")
        openers = {
            "hopeful": [
                f"The {element} in me stirs with possibility.",
                "Another year. We are still here.",
            ],
            "anxious": [
                f"I cannot shake this feeling the {element} warns me.",
                "Resources are thin. I count rations again.",
            ],
            "desperate": [
                f"We are running out of time. The {element} flickers.",
                "I dreamed of Earth. Woke to red dust.",
            ],
        }
        text_lines = [
            f"**{name}** ({element}/{archetype}) \u2014 Year {year_result['year']}",
            "",
            rng.choice(openers.get(mood, openers["anxious"])),
            "",
        ]
        event_reactions = {
            "dust_storm": "The dust never stops.",
            "resource_strike": "We found something. Maybe Mars is not "
                               "trying to kill us.",
            "equipment_failure": "Another system down.",
            "earth_contact": "A message from Earth \u2014 like hearing "
                             "from a ghost.",
            "alien_signal": "The signal... I do not know what to think.",
            "solar_flare": "Sheltered underground for days.",
            "colonist_conflict": "Tempers flared. It is each other that "
                                 "will kill us.",
            "breakthrough": "We did something impossible today.",
        }
        for ev in events[:2]:
            ev_name = (ev.get("name") if isinstance(ev, dict)
                       else getattr(ev, "name", ""))
            if ev_name in event_reactions:
                text_lines.append(event_reactions[ev_name])
        text_lines.extend(["", f"*\u2014 {name}, {archetype}*"])
        entries.append({
            "colonist_id": colonist["id"],
            "year": year_result["year"],
            "text": "\n".join(text_lines),
        })
    return entries


def generate_final_report(sim_result: dict) -> str:
    """Generate the final Emergent Governance Patterns report."""
    summary = sim_result.get("summary", {})
    final_gov = sim_result.get("final_governance", {})
    years_data = sim_result.get("years", [])
    colonists = sim_result.get("final_colonists", [])
    meta = sim_result.get("_meta", {})

    lines = [
        "# Emergent Governance Patterns from Mars-100",
        "",
        "## Summary",
        "",
        f"- **Seed:** {meta.get('seed', '?')}",
        f"- **Duration:** {meta.get('completed_years', len(years_data))} "
        f"Martian years (requested: {meta.get('requested_years', '?')})",
        f"- **Extinction:** Year {meta.get('extinction_year', 'N/A')}",
        f"- **Deaths:** {summary.get('total_deaths', 0)}",
        f"- **Exiles:** {summary.get('total_exiles', 0)}",
        f"- **Sub-simulations:** {summary.get('total_subsims', 0)}",
        f"- **Governance changes:** {summary.get('governance_changes', 0)}",
        f"- **Meta-awareness events:** "
        f"{summary.get('meta_awareness_events', 0)}",
        f"- **Final cohesion:** "
        f"{summary.get('final_cohesion', 0):.0%}",
        f"- **Final governance:** "
        f"{final_gov.get('gov_type', 'unknown')}",
        "",
    ]

    history = final_gov.get("history", [])
    if history:
        lines.extend(["## Governance Timeline", ""])
        for h in history:
            lines.append(f"- Year {h['year']}: {h['from']} \u2192 {h['to']}")
        lines.append("")

    constitution = final_gov.get("constitution", [])
    if constitution:
        lines.extend(["## Colony Constitution", ""])
        for rule in constitution:
            lines.append(f"- {rule}")
        lines.append("")

    alive = [c for c in colonists
             if c.get("alive") and not c.get("exiled")]
    dead = [c for c in colonists if not c.get("alive")]
    exiled = [c for c in colonists if c.get("exiled")]

    lines.extend(["## Colony Roster", "",
                   f"### Survivors ({len(alive)})"])
    for c in alive:
        lines.append(
            f"- **{c['name']}** ({c['element']}/{c['archetype']})")
    lines.append("")
    if dead:
        lines.append(f"### Fallen ({len(dead)})")
        for c in dead:
            lines.append(
                f"- **{c['name']}** \u2014 year "
                f"{c.get('death_year', '?')}: "
                f"{c.get('death_cause', '?')}")
        lines.append("")
    if exiled:
        lines.append(f"### Exiled ({len(exiled)})")
        for c in exiled:
            lines.append(
                f"- **{c['name']}** \u2014 year "
                f"{c.get('exile_year', '?')}")
        lines.append("")

    lines.extend([
        "## Proposed Constitutional Amendment",
        "",
        "> **Amendment XVIII \u2014 The Recursive Governance Principle**",
        ">",
        "> Any governance proposal affecting more than 3 agents must be",
        "> modeled in a sandboxed sub-simulation before being put to vote.",
        "> Simulation results become part of the proposal public record.",
        "> Governance decisions without simulation evidence are advisory "
        "only.",
        ">",
        "> *Rationale: Mars-100 demonstrated that sub-simulated governance",
        "> proposals had higher adoption rates and longer stability.*",
        "",
        "---",
        "*Generated by the Mars-100 recursive colony simulation.*",
    ])
    return "\n".join(lines)
