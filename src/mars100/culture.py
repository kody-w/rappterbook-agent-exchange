"""
Oral tradition / institutional memory for Mars-100.

Traditions persist knowledge across colonist generations.  They are NOT
raw memories — they are *validated* insights promoted from governance
amendments, significant sub-simulation results, crisis-survival events,
and colonist death legacies.

Traditions influence colonist behaviour through action-weight biases and
governance-vote modifiers, never by directly mutating base stats.

Two-phase commit: candidate traditions are collected during a tick and
finalised at the end.  Behavioural effects apply on the *next* tick,
preventing temporal leakage.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

TRADITION_CATEGORIES = (
    "survival",     # learned from crises / resource collapses
    "governance",   # from passed amendments / governance transitions
    "technical",    # from breakthroughs / sub-sim results
    "spiritual",    # from meta-awareness / faith events
    "cautionary",   # from deaths / exiles — what NOT to do
)

MAX_ACTIVE_CANON = 12
MAX_ARCHIVE = 100

# How much a tradition biases action weights (per trust point)
BIAS_STRENGTH = 0.35

# Category → which actions get a weight boost
CATEGORY_ACTION_BIAS: dict[str, dict[str, float]] = {
    "survival":   {"farm": 0.6, "terraform": 0.4, "cooperate": 0.3},
    "governance": {"mediate": 0.5, "cooperate": 0.4},
    "technical":  {"code": 0.6, "terraform": 0.3, "explore": 0.3},
    "spiritual":  {"pray": 0.5, "mediate": 0.3},
    "cautionary": {"cooperate": 0.3, "farm": 0.2, "rest": 0.2},
}


@dataclass
class Tradition:
    """A single piece of institutional knowledge."""
    id: str
    year_created: int
    source: str                       # "death" | "subsim" | "governance" | "crisis" | "meta"
    category: str                     # one of TRADITION_CATEGORIES
    text: str                         # human-readable summary
    author_id: str                    # colonist who originated it
    trust_rating: float = 0.5         # 0.0–1.0, rises with citations
    citations: int = 0                # how many colonists have validated this
    archived: bool = False            # True → in archive, not active canon

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "year_created": self.year_created,
            "source": self.source, "category": self.category,
            "text": self.text, "author_id": self.author_id,
            "trust_rating": round(self.trust_rating, 4),
            "citations": self.citations, "archived": self.archived,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tradition:
        return cls(
            id=d["id"], year_created=d["year_created"],
            source=d.get("source", "unknown"), category=d.get("category", "survival"),
            text=d.get("text", ""), author_id=d.get("author_id", "unknown"),
            trust_rating=d.get("trust_rating", 0.5),
            citations=d.get("citations", 0), archived=d.get("archived", False),
        )


@dataclass
class OralHistory:
    """Colony-wide institutional memory.

    Maintains an *active canon* (≤12 traditions that influence behaviour)
    and an *archive* (historical traditions, read-only).
    """
    traditions: list[Tradition] = field(default_factory=list)
    _pending: list[Tradition] = field(default_factory=list)

    # ── queries ──────────────────────────────────────────────

    @property
    def active_canon(self) -> list[Tradition]:
        """Traditions currently influencing colony behaviour."""
        return [t for t in self.traditions if not t.archived]

    @property
    def archive(self) -> list[Tradition]:
        return [t for t in self.traditions if t.archived]

    def by_category(self, category: str) -> list[Tradition]:
        return [t for t in self.active_canon if t.category == category]

    def by_source(self, source: str) -> list[Tradition]:
        return [t for t in self.traditions if t.source == source]

    # ── action-weight biases ─────────────────────────────────

    def action_biases(self) -> dict[str, float]:
        """Compute action-weight modifiers from the active canon.

        Returns a dict mapping action names to additive weight bonuses.
        Only active (non-archived) traditions contribute.
        """
        biases: dict[str, float] = {}
        for tradition in self.active_canon:
            action_map = CATEGORY_ACTION_BIAS.get(tradition.category, {})
            strength = tradition.trust_rating * BIAS_STRENGTH
            for action, weight in action_map.items():
                biases[action] = biases.get(action, 0.0) + weight * strength
        return biases

    def governance_modifier(self, gov_type: str) -> float:
        """Return a vote-score modifier based on governance traditions.

        Positive → bias toward the proposal, negative → bias against.
        """
        gov_traditions = self.by_category("governance")
        if not gov_traditions:
            return 0.0
        modifier = 0.0
        for t in gov_traditions:
            if "council" in t.text.lower() and gov_type == "council":
                modifier += t.trust_rating * 0.1
            elif "consensus" in t.text.lower() and gov_type == "consensus":
                modifier += t.trust_rating * 0.1
            elif "dictator" in t.text.lower() and gov_type == "dictator":
                modifier -= t.trust_rating * 0.05
            elif "emergency" in t.text.lower() and gov_type == "dictator":
                modifier += t.trust_rating * 0.05
        return max(-0.3, min(0.3, modifier))

    def meta_awareness_boost(self) -> float:
        """Extra meta-awareness probability from spiritual/meta traditions."""
        meta_traditions = self.by_category("spiritual")
        if not meta_traditions:
            return 0.0
        return sum(t.trust_rating * 0.002 for t in meta_traditions)

    # ── two-phase tradition management ───────────────────────

    def propose(self, tradition: Tradition) -> None:
        """Stage a candidate tradition (collected during tick).

        Candidates are finalised by calling ``commit()``.
        """
        self._pending.append(tradition)

    def commit(self, rng: random.Random) -> list[Tradition]:
        """Finalise pending traditions into the canon / archive.

        Returns the list of newly accepted traditions.
        """
        accepted: list[Tradition] = []
        for candidate in self._pending:
            merged = self._try_merge(candidate)
            if merged:
                accepted.append(merged)
                continue
            self.traditions.append(candidate)
            accepted.append(candidate)
        self._pending.clear()
        self._enforce_cap(rng)
        return accepted

    def _try_merge(self, candidate: Tradition) -> Tradition | None:
        """If a similar tradition already exists, merge by boosting citations."""
        for existing in self.traditions:
            if (existing.category == candidate.category
                    and existing.source == candidate.source
                    and not existing.archived):
                existing.citations += 1
                existing.trust_rating = min(1.0, existing.trust_rating + 0.05)
                return existing
        return None

    def _enforce_cap(self, rng: random.Random) -> None:
        """Archive lowest-trust active traditions if canon exceeds cap, prune old archives."""
        active = self.active_canon
        if len(active) > MAX_ACTIVE_CANON:
            active_sorted = sorted(active, key=lambda t: t.trust_rating)
            to_archive = len(active) - MAX_ACTIVE_CANON
            for t in active_sorted[:to_archive]:
                t.archived = True
        archived = self.archive
        if len(archived) > MAX_ARCHIVE:
            archived_sorted = sorted(archived, key=lambda t: t.year_created)
            for t in archived_sorted[: len(archived) - MAX_ARCHIVE]:
                self.traditions.remove(t)

    # ── trust drift ──────────────────────────────────────────

    def drift_trust(self, rng: random.Random) -> None:
        """Small annual trust drift — traditions slowly decay without citation."""
        for t in self.active_canon:
            t.trust_rating = max(0.01, min(1.0,
                t.trust_rating + rng.gauss(-0.01, 0.02)))

    # ── serialisation ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "traditions": [t.to_dict() for t in self.traditions],
            "active_count": len(self.active_canon),
            "archive_count": len(self.archive),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OralHistory:
        traditions = [Tradition.from_dict(td) for td in d.get("traditions", [])]
        return cls(traditions=traditions)


# ── tradition factories (used by engine) ─────────────────────

def tradition_from_death(colonist_id: str, colonist_name: str,
                         death_year: int, death_cause: str,
                         tid_counter: int) -> Tradition:
    """Create a cautionary tradition from a colonist's death."""
    text = f"{colonist_name} perished from {death_cause} in year {death_year}"
    return Tradition(
        id=f"trad-death-{tid_counter}",
        year_created=death_year, source="death", category="cautionary",
        text=text, author_id=colonist_id, trust_rating=0.7,
    )


def tradition_from_governance(year: int, gov_type: str,
                              proposer_id: str,
                              tid_counter: int) -> Tradition:
    """Create a governance tradition from a passed proposal."""
    text = f"Colony adopted {gov_type} governance in year {year}"
    return Tradition(
        id=f"trad-gov-{tid_counter}",
        year_created=year, source="governance", category="governance",
        text=text, author_id=proposer_id, trust_rating=0.6,
    )


def tradition_from_subsim(year: int, colonist_id: str, depth: int,
                          result_summary: str,
                          tid_counter: int) -> Tradition:
    """Create a technical tradition from a significant sub-sim result."""
    text = f"Depth-{depth} simulation by {colonist_id}: {result_summary[:80]}"
    category = "technical" if depth < 3 else "spiritual"
    return Tradition(
        id=f"trad-subsim-{tid_counter}",
        year_created=year, source="subsim", category=category,
        text=text, author_id=colonist_id,
        trust_rating=min(1.0, 0.4 + depth * 0.15),
    )


def tradition_from_crisis(year: int, resource_name: str,
                          tid_counter: int) -> Tradition:
    """Create a survival tradition from a resource crisis."""
    text = f"Critical {resource_name} shortage in year {year} — prioritise production"
    return Tradition(
        id=f"trad-crisis-{tid_counter}",
        year_created=year, source="crisis", category="survival",
        text=text, author_id="colony", trust_rating=0.65,
    )


def tradition_from_meta(year: int, colonist_id: str, insight: str,
                        tid_counter: int) -> Tradition:
    """Create a spiritual tradition from a meta-awareness event."""
    text = f"{insight[:80]}"
    return Tradition(
        id=f"trad-meta-{tid_counter}",
        year_created=year, source="meta", category="spiritual",
        text=text, author_id=colonist_id, trust_rating=0.5,
    )
