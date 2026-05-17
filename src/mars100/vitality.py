"""
Vitality organ for Mars-100 (engine v12.0).

Monitors dormancy across subsystems — factions, action categories, traditions,
infrastructure projects, ideologies. When a subsystem flatlines (0 activity for
N+ years), the organ emits a RevivalPrompt: a structured suggestion the engine
(or a narrator) can turn into a concrete revival event, immigrant archetype,
or governance proposal.

This is the immune system's other half: psychology handles personal stress,
diplomacy handles social fracture, but neither notices when a whole *category*
of life quietly stops happening. Vitality watches the silences.

Seed analogue (Rappterbook channel_health.py):
  - "dead channels (0 posts in 10+ frames)" -> dormant subjects (0 activity in
    DORMANCY_THRESHOLD+ years)
  - "revival prompts for the autonomy loop" -> RevivalPrompt list, consumable
    by the engine's event generator or governance layer
  - "state/channel_health.json" -> VitalityState.to_dict(), per-subject vitals

Pure computation. No I/O. RNG offset: seed + 13691.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DORMANCY_THRESHOLD = 10
CRITICAL_THRESHOLD = 25
MAX_PROMPTS_PER_TICK = 5
PROMPT_COOLDOWN_YEARS = 5

SUBJECT_TYPES = ("faction", "action", "tradition", "project", "ideology")

TRACKED_ACTIONS = (
    "terraform", "farm", "mediate", "code", "pray",
    "sabotage", "cooperate", "hoard", "explore", "rest", "research",
)

TRACKED_IDEOLOGIES = (
    "cooperative", "survivalist", "spiritual", "technocratic", "isolationist",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VitalitySubject:
    subject_type: str
    subject_id: str
    label: str = ""
    last_active_year: int = 0
    years_dormant: int = 0
    total_activity: int = 0
    last_prompt_year: int = -PROMPT_COOLDOWN_YEARS
    revival_count: int = 0
    first_seen_year: int = 0

    def to_dict(self) -> dict:
        return {
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "label": self.label,
            "last_active_year": self.last_active_year,
            "years_dormant": self.years_dormant,
            "total_activity": self.total_activity,
            "last_prompt_year": self.last_prompt_year,
            "revival_count": self.revival_count,
            "first_seen_year": self.first_seen_year,
            "status": self.status(),
            "vitality_score": self.vitality_score(),
        }

    def status(self) -> str:
        if self.years_dormant >= CRITICAL_THRESHOLD:
            return "critical"
        if self.years_dormant >= DORMANCY_THRESHOLD:
            return "dormant"
        if self.years_dormant >= DORMANCY_THRESHOLD // 2:
            return "fading"
        return "alive"

    def vitality_score(self) -> float:
        if self.years_dormant <= 0:
            return 1.0
        if self.years_dormant >= CRITICAL_THRESHOLD:
            return 0.0
        return max(0.0, 1.0 - (self.years_dormant / CRITICAL_THRESHOLD))


@dataclass
class RevivalPrompt:
    subject_type: str
    subject_id: str
    severity: str
    years_dormant: int
    suggestion: str
    suggested_event: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "severity": self.severity,
            "years_dormant": self.years_dormant,
            "suggestion": self.suggestion,
            "suggested_event": dict(self.suggested_event),
        }


@dataclass
class VitalityState:
    subjects: dict[str, VitalitySubject] = field(default_factory=dict)
    revival_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        by_type: dict[str, list[dict]] = {t: [] for t in SUBJECT_TYPES}
        for subj in self.subjects.values():
            by_type.setdefault(subj.subject_type, []).append(subj.to_dict())
        for t in by_type:
            by_type[t].sort(key=lambda s: s["years_dormant"], reverse=True)

        counts = _status_counts(self.subjects.values())
        return {
            "subjects": [s.to_dict() for s in self.subjects.values()],
            "by_type": by_type,
            "status_counts": counts,
            "total_subjects": len(self.subjects),
            "revival_history": list(self.revival_history),
            "dormancy_threshold": DORMANCY_THRESHOLD,
            "critical_threshold": CRITICAL_THRESHOLD,
        }

    def get_or_create(self, subject_type: str, subject_id: str, year: int,
                      label: str = "") -> VitalitySubject:
        key = _subject_key(subject_type, subject_id)
        if key not in self.subjects:
            self.subjects[key] = VitalitySubject(
                subject_type=subject_type,
                subject_id=subject_id,
                label=label or subject_id,
                first_seen_year=year,
                last_active_year=year,
            )
        return self.subjects[key]


@dataclass
class VitalityTickResult:
    year: int
    prompts: list[RevivalPrompt] = field(default_factory=list)
    newly_dormant: list[str] = field(default_factory=list)
    revived: list[str] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "prompts": [p.to_dict() for p in self.prompts],
            "newly_dormant": list(self.newly_dormant),
            "revived": list(self.revived),
            "summary": dict(self.summary),
        }


# ---------------------------------------------------------------------------
# Activity gathering
# ---------------------------------------------------------------------------

def _subject_key(subject_type: str, subject_id: str) -> str:
    return f"{subject_type}:{subject_id}"


def gather_activity(
    year: int,
    actions: dict[str, str] | None = None,
    factions: Iterable[Any] | None = None,
    cultural_traditions: Iterable[Any] | None = None,
    active_projects: Iterable[Any] | None = None,
    ideology_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    """Collapse a year's heterogeneous activity into {subject_key: count}."""
    activity: dict[str, int] = {}

    if actions:
        for act in actions.values():
            if act in TRACKED_ACTIONS:
                key = _subject_key("action", act)
                activity[key] = activity.get(key, 0) + 1

    if factions:
        for fac in factions:
            fid = _faction_id(fac)
            members = _faction_members(fac)
            if fid and members:
                key = _subject_key("faction", fid)
                activity[key] = activity.get(key, 0) + len(members)

    if cultural_traditions:
        for trad in cultural_traditions:
            tid = _tradition_id(trad)
            if tid:
                key = _subject_key("tradition", tid)
                activity[key] = activity.get(key, 0) + 1

    if active_projects:
        for proj in active_projects:
            pid = _project_id(proj)
            if pid:
                key = _subject_key("project", pid)
                activity[key] = activity.get(key, 0) + 1

    if ideology_counts:
        for ideo, count in ideology_counts.items():
            if ideo in TRACKED_IDEOLOGIES and count > 0:
                key = _subject_key("ideology", ideo)
                activity[key] = activity.get(key, 0) + count

    return activity


def _faction_id(fac: Any) -> str:
    for attr in ("id", "faction_id", "name"):
        v = getattr(fac, attr, None) or (fac.get(attr) if isinstance(fac, dict) else None)
        if v:
            return str(v)
    return ""


def _faction_members(fac: Any) -> list[Any]:
    for attr in ("members", "member_ids", "colonist_ids"):
        v = getattr(fac, attr, None)
        if v is None and isinstance(fac, dict):
            v = fac.get(attr)
        if v:
            return list(v)
    return []


def _tradition_id(trad: Any) -> str:
    for attr in ("id", "name"):
        v = getattr(trad, attr, None) or (trad.get(attr) if isinstance(trad, dict) else None)
        if v:
            return str(v)
    return str(trad) if isinstance(trad, str) else ""


def _project_id(proj: Any) -> str:
    for attr in ("tech_id", "id", "name"):
        v = getattr(proj, attr, None) or (proj.get(attr) if isinstance(proj, dict) else None)
        if v:
            return str(v)
    return ""


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

def tick_vitality(
    state: VitalityState,
    year: int,
    activity: dict[str, int],
    known_subjects: dict[str, list[tuple[str, str]]] | None = None,
) -> VitalityTickResult:
    """Advance the vitality ledger one year."""
    result = VitalityTickResult(year=year)

    if known_subjects:
        for subject_type, entries in known_subjects.items():
            for subject_id, label in entries:
                state.get_or_create(subject_type, subject_id, year, label)

    for key, count in activity.items():
        if count <= 0:
            continue
        subject_type, _, subject_id = key.partition(":")
        state.get_or_create(subject_type, subject_id, year)

    for key, subj in state.subjects.items():
        count = activity.get(key, 0)
        if count > 0:
            was_dormant = subj.years_dormant >= DORMANCY_THRESHOLD
            subj.last_active_year = year
            subj.total_activity += count
            if was_dormant:
                subj.revival_count += 1
                result.revived.append(key)
                state.revival_history.append({
                    "year": year,
                    "subject_type": subj.subject_type,
                    "subject_id": subj.subject_id,
                    "after_dormant_years": subj.years_dormant,
                    "kind": "natural",
                })
            subj.years_dormant = 0
        else:
            # Don't count the birth tick against a brand-new subject.
            if subj.first_seen_year == year:
                continue
            prev = subj.years_dormant
            subj.years_dormant += 1
            if prev < DORMANCY_THRESHOLD <= subj.years_dormant:
                result.newly_dormant.append(key)

    candidates = [
        s for s in state.subjects.values()
        if s.years_dormant >= DORMANCY_THRESHOLD
        and (year - s.last_prompt_year) >= PROMPT_COOLDOWN_YEARS
    ]
    candidates.sort(key=lambda s: s.years_dormant, reverse=True)

    for subj in candidates[:MAX_PROMPTS_PER_TICK]:
        prompt = _make_prompt(subj, year)
        result.prompts.append(prompt)
        subj.last_prompt_year = year

    result.summary = _status_counts(state.subjects.values())
    result.summary["prompts_emitted"] = len(result.prompts)
    result.summary["newly_dormant"] = len(result.newly_dormant)
    result.summary["revived"] = len(result.revived)
    return result


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def _make_prompt(subj: VitalitySubject, year: int) -> RevivalPrompt:
    severity = "critical" if subj.years_dormant >= CRITICAL_THRESHOLD else "dormant"
    suggestion, event = _suggestion_for(subj, year, severity)
    return RevivalPrompt(
        subject_type=subj.subject_type,
        subject_id=subj.subject_id,
        severity=severity,
        years_dormant=subj.years_dormant,
        suggestion=suggestion,
        suggested_event=event,
    )


def _suggestion_for(subj: VitalitySubject, year: int,
                    severity: str) -> tuple[str, dict[str, Any]]:
    t = subj.subject_type
    sid = subj.subject_id
    yd = subj.years_dormant

    if t == "action":
        return (
            f"No colonist has chosen '{sid}' in {yd} years. "
            f"Consider an event that rewards or requires it, or an immigrant "
            f"archetype with a strong preference for {sid}.",
            {
                "kind": "event",
                "template": "action_revival",
                "action": sid,
                "morale_bonus_if_chosen": 0.05,
                "severity": severity,
            },
        )

    if t == "faction":
        return (
            f"Faction '{sid}' has been inactive for {yd} years. "
            f"Either dissolve it formally (cultural memory entry) or seed a "
            f"recruitment event that draws aligned colonists.",
            {
                "kind": "diplomacy",
                "template": "faction_revival",
                "faction_id": sid,
                "action": "recruit" if severity == "dormant" else "dissolve",
                "severity": severity,
            },
        )

    if t == "tradition":
        return (
            f"Tradition '{sid}' has not been referenced in {yd} years. "
            f"Mark it as fading folklore — let it influence a single "
            f"colonist's diary entry, or promote it back via a festival event.",
            {
                "kind": "culture",
                "template": "tradition_revival",
                "tradition_id": sid,
                "action": "festival" if severity == "dormant" else "archive",
                "severity": severity,
            },
        )

    if t == "project":
        return (
            f"Infrastructure project '{sid}' has stalled for {yd} years. "
            f"Reallocate researchers or cancel and refund a fraction of cost.",
            {
                "kind": "infrastructure",
                "template": "project_revival",
                "project_id": sid,
                "action": "reallocate" if severity == "dormant" else "cancel",
                "severity": severity,
            },
        )

    if t == "ideology":
        return (
            f"Ideology '{sid}' has no adherents in {yd} years. "
            f"Consider an immigrant cohort or a generational backlash event "
            f"that reintroduces this worldview.",
            {
                "kind": "immigration",
                "template": "ideology_revival",
                "ideology": sid,
                "severity": severity,
            },
        )

    return (
        f"Subject '{sid}' (type={t}) dormant {yd} years.",
        {"kind": "generic", "subject_id": sid, "severity": severity},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_counts(subjects: Iterable[VitalitySubject]) -> dict[str, int]:
    counts: dict[str, int] = {"alive": 0, "fading": 0, "dormant": 0, "critical": 0}
    for s in subjects:
        counts[s.status()] = counts.get(s.status(), 0) + 1
    return counts


def compute_colony_vitality_index(state: VitalityState) -> float:
    """Aggregate vitality of the whole colony: mean per-subject score, 0..1."""
    if not state.subjects:
        return 1.0
    total = sum(s.vitality_score() for s in state.subjects.values())
    return total / len(state.subjects)
