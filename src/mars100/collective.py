"""
Collective memory organ for Mars-100.

A shared knowledge substrate that colonists read from and write to.
Knowledge entries carry provenance (source, conditions, confidence).
Reads use snapshot semantics: colonists see last-tick state only.
Traditions emerge from repeated convergent behavior and can decay.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


TOPICS = ("resources", "governance", "survival", "social", "exploration", "technology")
KNOWLEDGE_HALF_LIFE_YEARS = 20
MAX_ARCHIVE_SIZE = 200
TRADITION_THRESHOLD = 5  # years of convergent behavior to form
TRADITION_BONUS = 0.03   # stat bonus from active tradition
ARCHIVE_INFLUENCE_CAP = 0.15  # max weight shift from archive
EXPLORATION_FLOOR = 0.20  # minimum weight for non-archive-biased actions
MIN_CONFIDENCE = 0.3  # below this, entries are ignored in queries


@dataclass
class KnowledgeEntry:
    """A single piece of colony knowledge with full provenance."""
    id: str
    topic: str
    source_colonist: str
    year_created: int
    content: str
    confidence: float
    conditions: dict[str, Any]
    outcome: float | None = None
    last_validated_year: int | None = None

    def age(self, current_year: int) -> int:
        """Years since creation."""
        return current_year - self.year_created

    def effective_confidence(self, current_year: int) -> float:
        """Confidence decayed by age using half-life model."""
        age = self.age(current_year)
        decay = math.pow(0.5, age / KNOWLEDGE_HALF_LIFE_YEARS)
        return self.confidence * decay

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id, "topic": self.topic,
            "source_colonist": self.source_colonist,
            "year_created": self.year_created,
            "content": self.content, "confidence": self.confidence,
            "conditions": self.conditions,
        }
        if self.outcome is not None:
            d["outcome"] = self.outcome
        if self.last_validated_year is not None:
            d["last_validated_year"] = self.last_validated_year
        return d

    @classmethod
    def from_dict(cls, d: dict) -> KnowledgeEntry:
        return cls(
            id=d["id"], topic=d["topic"],
            source_colonist=d["source_colonist"],
            year_created=d["year_created"],
            content=d["content"], confidence=d["confidence"],
            conditions=d.get("conditions", {}),
            outcome=d.get("outcome"),
            last_validated_year=d.get("last_validated_year"),
        )


@dataclass
class Tradition:
    """An emergent cultural tradition from convergent behavior."""
    id: str
    name: str
    action: str
    year_formed: int
    strength: float = 1.0
    streak_years: int = 0
    participants: int = 0

    def bonus(self) -> float:
        """Stat bonus provided by this tradition."""
        return TRADITION_BONUS * min(self.strength, 2.0)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "action": self.action,
            "year_formed": self.year_formed, "strength": self.strength,
            "streak_years": self.streak_years, "participants": self.participants,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Tradition:
        return cls(
            id=d["id"], name=d["name"], action=d["action"],
            year_formed=d["year_formed"], strength=d.get("strength", 1.0),
            streak_years=d.get("streak_years", 0),
            participants=d.get("participants", 0),
        )


TRADITION_NAMES = {
    "terraform": "The Greening",
    "farm": "Harvest Festival",
    "mediate": "Peace Circle",
    "code": "The Builders",
    "pray": "Temple Hour",
    "sabotage": "Shadow Game",
    "cooperate": "Unity Day",
    "hoard": "The Stockpile",
    "explore": "Frontier March",
    "rest": "Sabbath Rest",
}


@dataclass
class CollectiveMemory:
    """Colony-wide shared knowledge and cultural traditions.

    Uses snapshot semantics: ``snapshot()`` freezes current state at tick
    start.  All reads during the tick come from the snapshot.  Writes go
    to the live archive and become visible on the next ``snapshot()`` call.
    """
    archive: list[KnowledgeEntry] = field(default_factory=list)
    traditions: list[Tradition] = field(default_factory=list)
    action_history: dict[str, list[int]] = field(default_factory=dict)
    _snapshot_archive: list[KnowledgeEntry] = field(default_factory=list)
    _next_id: int = 0
    events: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Snapshot lifecycle
    # ------------------------------------------------------------------

    def snapshot(self) -> None:
        """Freeze current archive for reads during this tick."""
        self._snapshot_archive = list(self.archive)

    # ------------------------------------------------------------------
    # Write path (goes to live archive, visible next tick)
    # ------------------------------------------------------------------

    def store_knowledge(self, topic: str, source_colonist: str,
                        year: int, content: str, confidence: float,
                        conditions: dict[str, Any] | None = None,
                        outcome: float | None = None) -> KnowledgeEntry:
        """Store a new knowledge entry from a sub-sim or observation."""
        entry = KnowledgeEntry(
            id=f"k-{self._next_id}",
            topic=topic if topic in TOPICS else "survival",
            source_colonist=source_colonist,
            year_created=year,
            content=content[:200],
            confidence=max(0.0, min(1.0, confidence)),
            conditions=conditions or {},
            outcome=outcome,
        )
        self._next_id += 1
        self.archive.append(entry)
        self._prune_archive(year)
        self.events.append({
            "type": "knowledge_stored", "year": year,
            "topic": topic, "source": source_colonist,
            "confidence": entry.confidence,
        })
        return entry

    def _prune_archive(self, current_year: int) -> None:
        """Remove low-confidence expired entries to stay within budget."""
        if len(self.archive) <= MAX_ARCHIVE_SIZE:
            return
        scored = [(e, e.effective_confidence(current_year)) for e in self.archive]
        scored.sort(key=lambda x: x[1])
        while len(self.archive) > MAX_ARCHIVE_SIZE:
            weakest = scored.pop(0)
            self.archive.remove(weakest[0])

    # ------------------------------------------------------------------
    # Read path (from snapshot only)
    # ------------------------------------------------------------------

    def query(self, topic: str, current_year: int,
              max_results: int = 5) -> list[KnowledgeEntry]:
        """Query the snapshot archive for entries matching a topic.

        Returns entries sorted by effective confidence (highest first).
        Only returns entries above MIN_CONFIDENCE.
        """
        candidates = [
            e for e in self._snapshot_archive
            if e.topic == topic
            and e.effective_confidence(current_year) >= MIN_CONFIDENCE
        ]
        candidates.sort(
            key=lambda e: e.effective_confidence(current_year), reverse=True)
        return candidates[:max_results]

    def archive_stats(self, current_year: int) -> dict[str, Any]:
        """Summary statistics of the knowledge archive."""
        by_topic: dict[str, int] = {}
        total_confidence = 0.0
        for e in self.archive:
            by_topic[e.topic] = by_topic.get(e.topic, 0) + 1
            total_confidence += e.effective_confidence(current_year)
        return {
            "total_entries": len(self.archive),
            "by_topic": by_topic,
            "avg_confidence": total_confidence / max(1, len(self.archive)),
            "traditions_active": len([t for t in self.traditions if t.strength > 0.5]),
        }

    # ------------------------------------------------------------------
    # Action bias from archive (capped influence)
    # ------------------------------------------------------------------

    def action_bias(self, topic: str, current_year: int) -> dict[str, float]:
        """Compute action weight biases from archived knowledge.

        Returns a dict of action -> weight delta, capped at ARCHIVE_INFLUENCE_CAP.
        The bias is derived from outcomes of past knowledge entries.
        """
        entries = self.query(topic, current_year, max_results=10)
        if not entries:
            return {}
        bias: dict[str, float] = {}
        for entry in entries:
            ec = entry.effective_confidence(current_year)
            outcome = entry.outcome if entry.outcome is not None else 0.0
            action_hint = entry.conditions.get("action")
            if action_hint and isinstance(action_hint, str):
                current = bias.get(action_hint, 0.0)
                delta = outcome * ec * 0.1
                bias[action_hint] = max(-ARCHIVE_INFLUENCE_CAP,
                                        min(ARCHIVE_INFLUENCE_CAP, current + delta))
        return bias

    # ------------------------------------------------------------------
    # Tradition lifecycle
    # ------------------------------------------------------------------

    def update_traditions(self, year: int, action_counts: dict[str, int],
                          active_count: int) -> list[dict]:
        """Update traditions based on this year's action distribution.

        A tradition forms when the same action is dominant for
        TRADITION_THRESHOLD consecutive years.  Traditions weaken when
        the action drops below 20% of colonists.
        """
        tradition_events: list[dict] = []
        if active_count < 3:
            return tradition_events

        dominant_action = max(action_counts, key=lambda a: action_counts.get(a, 0))
        dominant_fraction = action_counts.get(dominant_action, 0) / active_count

        # Track streak
        history = self.action_history
        if dominant_action not in history:
            history[dominant_action] = []
        history[dominant_action].append(year)

        streak = self._compute_streak(dominant_action, year)

        # Form new tradition
        if (streak >= TRADITION_THRESHOLD
                and dominant_fraction >= 0.3
                and not self._has_tradition(dominant_action)):
            name = TRADITION_NAMES.get(dominant_action, f"The {dominant_action.title()}")
            tradition = Tradition(
                id=f"trad-{dominant_action}-y{year}",
                name=name, action=dominant_action,
                year_formed=year, strength=1.0,
                streak_years=streak,
                participants=action_counts.get(dominant_action, 0),
            )
            self.traditions.append(tradition)
            tradition_events.append({
                "type": "tradition_formed", "year": year,
                "name": name, "action": dominant_action,
            })

        # Strengthen or weaken existing traditions
        for tradition in self.traditions:
            count = action_counts.get(tradition.action, 0)
            fraction = count / active_count
            if fraction >= 0.3:
                tradition.strength = min(3.0, tradition.strength + 0.1)
                tradition.streak_years += 1
                tradition.participants = count
            elif fraction < 0.15:
                tradition.strength -= 0.2
                tradition.streak_years = 0
                if tradition.strength <= 0:
                    tradition_events.append({
                        "type": "tradition_died", "year": year,
                        "name": tradition.name, "action": tradition.action,
                    })

        # Remove dead traditions
        self.traditions = [t for t in self.traditions if t.strength > 0]
        self.events.extend(tradition_events)
        return tradition_events

    def _compute_streak(self, action: str, current_year: int) -> int:
        """Count consecutive recent years where this action appeared."""
        years = self.action_history.get(action, [])
        if not years:
            return 0
        streak = 0
        for y in range(current_year, 0, -1):
            if y in years:
                streak += 1
            else:
                break
        return streak

    def _has_tradition(self, action: str) -> bool:
        """Check if a tradition for this action already exists."""
        return any(t.action == action for t in self.traditions)

    def tradition_bonuses(self) -> dict[str, float]:
        """Compute stat bonuses from active traditions."""
        bonuses: dict[str, float] = {}
        skill_map = {
            "terraform": "terraforming", "farm": "hydroponics",
            "mediate": "mediation", "code": "coding",
            "pray": "prayer", "cooperate": "empathy_boost",
            "explore": "improvisation_boost",
        }
        for tradition in self.traditions:
            if tradition.strength > 0.5:
                skill = skill_map.get(tradition.action)
                if skill:
                    bonuses[skill] = bonuses.get(skill, 0.0) + tradition.bonus()
        return bonuses

    # ------------------------------------------------------------------
    # Knowledge extraction from sub-sims
    # ------------------------------------------------------------------

    def extract_from_subsim(self, subsim_result: dict, year: int,
                            resource_state: dict[str, float]) -> KnowledgeEntry | None:
        """Extract knowledge from a sub-simulation result.

        Only stores entries with meaningful results (non-None, numeric).
        Confidence is derived from sub-sim depth and result magnitude.
        """
        result = subsim_result.get("result")
        if result is None or subsim_result.get("error"):
            return None
        if not isinstance(result, (int, float)):
            return None

        depth = subsim_result.get("depth", 1)
        confidence = min(1.0, 0.4 + depth * 0.15 + min(abs(result), 1.0) * 0.2)
        topic = self._classify_topic(subsim_result.get("expression", ""))

        conditions: dict[str, Any] = {
            "depth": depth,
            "resources": {k: round(v, 2) for k, v in resource_state.items()},
        }

        return self.store_knowledge(
            topic=topic, source_colonist=subsim_result.get("colonist_id", "unknown"),
            year=year, content=f"subsim d{depth}: {subsim_result.get('expression', '')[:100]}",
            confidence=confidence, conditions=conditions,
            outcome=float(result),
        )

    def _classify_topic(self, expression: str) -> str:
        """Classify a LisPy expression into a topic category."""
        expr_lower = expression.lower()
        if any(w in expr_lower for w in ("food", "water", "power", "air", "surplus")):
            return "resources"
        if any(w in expr_lower for w in ("gov", "trust", "vote", "council")):
            return "governance"
        if any(w in expr_lower for w in ("empathy", "resolve", "faith", "paranoia")):
            return "social"
        if any(w in expr_lower for w in ("explore", "terrain", "cave")):
            return "exploration"
        if any(w in expr_lower for w in ("code", "tech", "build")):
            return "technology"
        return "survival"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "archive": [e.to_dict() for e in self.archive],
            "traditions": [t.to_dict() for t in self.traditions],
            "action_history": self.action_history,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CollectiveMemory:
        cm = cls()
        cm.archive = [KnowledgeEntry.from_dict(e) for e in d.get("archive", [])]
        cm.traditions = [Tradition.from_dict(t) for t in d.get("traditions", [])]
        cm.action_history = d.get("action_history", {})
        cm._next_id = d.get("next_id", len(cm.archive))
        return cm
