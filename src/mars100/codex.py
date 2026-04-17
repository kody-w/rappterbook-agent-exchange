"""Memory Codex — colony-level cultural memory.

The codex stores events, ancestor wisdom, and laws that persist across
generations.  Each entry type decays at a different rate per simulation
year so traumatic dust-storms fade faster than constitutional laws.

LisPy bindings expose five colony-wide numeric values that colonists
can read when evaluating their decision expressions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ── constants ──────────────────────────────────────────────────────
DECAY_RATES: dict[str, float] = {
    "event":    0.15,   # fades ~22 years
    "ancestor": 0.05,   # lingers ~60 years
    "law":      0.02,   # near-permanent (0.98^100 ≈ 0.13)
}
ACTIVE_THRESHOLD = 0.05
MAX_ENTRIES = 200


# ── data ───────────────────────────────────────────────────────────
@dataclass
class CodexEntry:
    """A single memory in the colony codex."""
    event_name: str
    entry_type: str          # "event" | "ancestor" | "law"
    strength: float = 1.0
    impact: float = 0.5      # peak impact score
    year_added: int = 0
    detail: str = ""

    # ── serialisation ──
    def to_dict(self) -> dict:
        return {
            "event_name": self.event_name,
            "entry_type": self.entry_type,
            "strength": round(self.strength, 6),
            "impact": round(self.impact, 6),
            "year_added": self.year_added,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CodexEntry:
        return cls(
            event_name=d["event_name"],
            entry_type=d["entry_type"],
            strength=d.get("strength", 1.0),
            impact=d.get("impact", 0.5),
            year_added=d.get("year_added", 0),
            detail=d.get("detail", ""),
        )


# ── codex ──────────────────────────────────────────────────────────
class Codex:
    """Colony-level cultural memory."""

    def __init__(self) -> None:
        self.entries: list[CodexEntry] = []
        self._event_index: dict[str, int] = {}

    # ── writers ──
    def add_event(self, name: str, *, impact: float = 0.5,
                  year: int = 0, detail: str = "") -> None:
        """Record a colony event.  Duplicate names reinforce."""
        if name in self._event_index:
            idx = self._event_index[name]
            e = self.entries[idx]
            e.strength = min(e.strength + 0.15, 1.0)
            return
        entry = CodexEntry(event_name=name, entry_type="event",
                           impact=impact, year_added=year, detail=detail)
        self._event_index[name] = len(self.entries)
        self.entries.append(entry)

    def add_ancestor_wisdom(self, ancestor_id: str,
                            memories: list[dict]) -> None:
        """Distil a dead colonist's memories into the codex."""
        if not memories:
            return
        avg_valence = sum(m.get("valence", 0) for m in memories) / len(memories)
        detail = "; ".join(m.get("event", "?") for m in memories[:5])
        name = f"ancestor:{ancestor_id}"
        entry = CodexEntry(
            event_name=name, entry_type="ancestor",
            impact=abs(avg_valence), detail=detail,
        )
        self._event_index[name] = len(self.entries)
        self.entries.append(entry)

    def add_law(self, description: str, *, year: int = 0) -> None:
        """Record a governance law."""
        name = f"law:{description[:60]}"
        entry = CodexEntry(
            event_name=name, entry_type="law",
            impact=0.8, year_added=year, detail=description,
        )
        self._event_index[name] = len(self.entries)
        self.entries.append(entry)

    # ── tick ──
    def tick_decay(self) -> None:
        """Apply one year of decay to every entry."""
        for e in self.entries:
            rate = DECAY_RATES.get(e.entry_type, 0.10)
            e.strength *= (1.0 - rate)

    def reinforce(self, event_name: str, amount: float = 0.1) -> None:
        """Strengthen an existing entry (capped at its impact)."""
        if event_name in self._event_index:
            e = self.entries[self._event_index[event_name]]
            e.strength = min(e.strength + amount, 1.0)

    # ── readers ──
    def get_active(self) -> list[CodexEntry]:
        """Return entries above the fade threshold."""
        return [e for e in self.entries if e.strength >= ACTIVE_THRESHOLD]

    def get_bindings(self) -> dict[str, float]:
        """Numeric bindings for LisPy decision expressions."""
        active = self.get_active()
        if not active:
            return {
                "codex-wisdom": 0.0,
                "codex-trauma": 0.0,
                "codex-law-count": 0.0,
                "codex-strength": 0.0,
                "codex-memory": 0.0,
            }
        wisdom = sum((e.strength for e in active if e.entry_type == "ancestor"), 0.0)
        trauma = sum((e.strength for e in active if e.entry_type == "event"), 0.0)
        laws = sum(1 for e in active if e.entry_type == "law")
        total = sum(e.strength for e in active)
        return {
            "codex-wisdom": round(wisdom, 4),
            "codex-trauma": round(trauma, 4),
            "codex-law-count": float(laws),
            "codex-strength": round(total, 4),
            "codex-memory": float(len(active)),
        }

    def snapshot(self) -> dict:
        """Compact dict for year-result serialisation."""
        active = self.get_active()
        return {
            "total_entries": len(self.entries),
            "active_entries": len(active),
            "bindings": self.get_bindings(),
        }

    # ── serialisation ──
    def to_dict(self) -> dict:
        return {
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Codex:
        c = cls()
        for ed in d.get("entries", []):
            entry = CodexEntry.from_dict(ed)
            c._event_index[entry.event_name] = len(c.entries)
            c.entries.append(entry)
        return c

    # ── housekeeping ──
    def _prune(self) -> None:
        """Drop weakest inactive entries when over MAX_ENTRIES."""
        if len(self.entries) <= MAX_ENTRIES:
            return
        # keep all active + strongest inactive up to limit
        active = [(i, e) for i, e in enumerate(self.entries)
                  if e.strength >= ACTIVE_THRESHOLD]
        inactive = [(i, e) for i, e in enumerate(self.entries)
                    if e.strength < ACTIVE_THRESHOLD]
        inactive.sort(key=lambda t: t[1].strength, reverse=True)
        keep_inactive = inactive[:MAX_ENTRIES - len(active)]
        keep_indices = {i for i, _ in active} | {i for i, _ in keep_inactive}
        self.entries = [e for i, e in enumerate(self.entries) if i in keep_indices]
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._event_index = {e.event_name: i for i, e in enumerate(self.entries)}


# ── child imprinting ──────────────────────────────────────────────
def imprint_child(codex: Codex, child_stats: dict[str, Any]) -> dict[str, Any]:
    """Nudge a newborn's stats based on colony memory (±0.02 max).

    Returns a *new* dict — does not mutate the input.
    """
    out = dict(child_stats)
    bindings = codex.get_bindings()
    trauma = bindings.get("codex-trauma", 0.0)
    wisdom = bindings.get("codex-wisdom", 0.0)

    # High trauma → slightly more paranoia, less faith
    if trauma > 1.0:
        nudge = min(0.02, trauma * 0.005)
        out["paranoia"] = min(1.0, out.get("paranoia", 0.5) + nudge)
        out["faith"] = max(0.0, out.get("faith", 0.5) - nudge * 0.5)

    # High wisdom → slightly more empathy
    if wisdom > 0.5:
        nudge = min(0.02, wisdom * 0.008)
        out["empathy"] = min(1.0, out.get("empathy", 0.5) + nudge)

    return out
