"""
Oral tradition and cultural memory for the Mars-100 colony.

Stories emerge from significant colony events, spread between colonists,
decay over time, and may promote to myths — persistent cultural memory
that influences governance votes and action selection.

Split into two phases per tick:
  1. pre_tick_signals() — read-only, provides biases for actions/votes
  2. post_tick_update() — mutates state: generate, spread, decay, promote
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

THEMES = ("loss", "crisis", "governance", "discovery", "transcendence",
          "hope", "progress", "exile", "cooperation")

# Myth influence decays: a myth from year 1 retains ~37% influence by year 100
MYTH_INFLUENCE_HALFLIFE = 60  # years until influence halves

MAX_ACTIVE_STORIES = 50
STORY_DECAY_RATE = 0.95
MYTH_PROMOTION_THRESHOLD = 0.65  # strength
MYTH_SPREAD_THRESHOLD = 0.5  # fraction of colony that knows the story
MYTH_TELLING_THRESHOLD = 8
STORY_EVICTION_THRESHOLD = 0.08


@dataclass
class Story:
    """A story circulating in the colony's oral tradition."""
    id: str
    year_created: int
    source_event: str
    theme: str
    rememberers: list[str] = field(default_factory=list)
    tellings: int = 0
    strength: float = 1.0
    is_myth: bool = False
    governance_stance: str | None = None

    def to_dict(self) -> dict:
        """Serialize for YearResult output."""
        return {
            "id": self.id,
            "year_created": self.year_created,
            "source_event": self.source_event,
            "theme": self.theme,
            "rememberers": list(self.rememberers),
            "tellings": self.tellings,
            "strength": round(self.strength, 4),
            "is_myth": self.is_myth,
            "governance_stance": self.governance_stance,
        }


@dataclass
class CultureSnapshot:
    """Summary of cultural state at end of a year."""
    active_stories: int
    myths: int
    dominant_theme: str
    cultural_cohesion: float
    governance_signals: dict[str, float]
    stories_created: int
    stories_promoted: int
    stories_evicted: int

    def to_dict(self) -> dict:
        return {
            "active_stories": self.active_stories,
            "myths": self.myths,
            "dominant_theme": self.dominant_theme,
            "cultural_cohesion": round(self.cultural_cohesion, 4),
            "governance_signals": {k: round(v, 4) for k, v in self.governance_signals.items()},
            "stories_created": self.stories_created,
            "stories_promoted": self.stories_promoted,
            "stories_evicted": self.stories_evicted,
        }


def _myth_influence(myth_year: int, current_year: int) -> float:
    """Compute decaying influence of a myth based on age.

    Uses exponential decay with MYTH_INFLUENCE_HALFLIFE.
    A myth from year 1 at year 61 has ~50% influence.
    """
    age = max(0, current_year - myth_year)
    import math
    return math.exp(-0.693 * age / max(1, MYTH_INFLUENCE_HALFLIFE))


# Theme → governance stance mapping
_THEME_GOV_MAP: dict[str, str | None] = {
    "loss": "council",  # loss drives desire for collective safety
    "crisis": "dictator",  # crises make strong leaders appealing
    "governance": None,  # inherits stance from the proposal itself
    "discovery": "ai_governor",  # discoveries favor technocracy
    "transcendence": "consensus",  # spiritual experiences favor consensus
    "hope": "council",  # hope reinforces community
    "progress": "ai_governor",  # tech progress favors tech governance
    "exile": "anarchy",  # exile stories breed distrust of authority
    "cooperation": "consensus",  # cooperation stories favor consensus
}

# Theme → action bias mapping (small effects: max ±0.3)
_THEME_ACTION_MAP: dict[str, dict[str, float]] = {
    "loss": {"pray": 0.15, "cooperate": 0.1, "hoard": 0.1},
    "crisis": {"farm": 0.2, "terraform": 0.15},
    "governance": {},  # no direct action bias
    "discovery": {"code": 0.15, "explore": 0.15},
    "transcendence": {"pray": 0.2, "mediate": 0.1},
    "hope": {"cooperate": 0.15, "farm": 0.1},
    "progress": {"research": 0.2, "code": 0.1},
    "exile": {"hoard": 0.15, "pray": 0.1},
    "cooperation": {"cooperate": 0.2, "mediate": 0.15},
}


class OralTradition:
    """Colony-wide oral tradition system.

    Manages story lifecycle: creation → spreading → decay → myth promotion.
    """

    def __init__(self) -> None:
        self.stories: list[Story] = []
        self._next_id = 0
        self._first_birth_seen = False
        self._recent_triggers: dict[str, int] = {}  # theme → last year created

    def _make_id(self) -> str:
        sid = f"story-{self._next_id}"
        self._next_id += 1
        return sid

    def _can_trigger(self, theme: str, year: int, cooldown: int = 3) -> bool:
        """Prevent duplicate stories from repeated events."""
        last = self._recent_triggers.get(theme, -999)
        return (year - last) >= cooldown

    def _mark_trigger(self, theme: str, year: int) -> None:
        self._recent_triggers[theme] = year

    # ── Pre-tick: read-only signals for action/vote biasing ──

    def governance_signals(self, year: int) -> dict[str, float]:
        """Return governance type preferences from cultural memory.

        Only myths contribute. Influence decays with age.
        Returns dict mapping gov_type → preference score (can be negative).
        """
        signals: dict[str, float] = {}
        for story in self.stories:
            if not story.is_myth or not story.governance_stance:
                continue
            influence = _myth_influence(story.year_created, year)
            stance = story.governance_stance
            signals[stance] = signals.get(stance, 0.0) + influence * story.strength
        return signals

    def action_bias(self, year: int) -> dict[str, float]:
        """Return action weight biases from cultural memory.

        Only myths contribute. Influence decays with age.
        """
        biases: dict[str, float] = {}
        for story in self.stories:
            if not story.is_myth:
                continue
            influence = _myth_influence(story.year_created, year)
            theme_biases = _THEME_ACTION_MAP.get(story.theme, {})
            for action, bias in theme_biases.items():
                biases[action] = biases.get(action, 0.0) + bias * influence * story.strength
        # Cap individual biases at ±0.3
        return {k: max(-0.3, min(0.3, v)) for k, v in biases.items()}

    # ── Post-tick: mutate cultural state ──

    def post_tick_update(
        self,
        year: int,
        active_colonists: list,  # list[Colonist]
        events: list[dict],
        deaths: list[dict],
        exiles: list[dict],
        governance: dict | None,
        meta_events: list[dict],
        subsim_log: list[dict],
        births: list[dict],
        infra_event: dict | None,
        rng: random.Random,
    ) -> CultureSnapshot:
        """Generate stories, spread, decay, promote. Called after all yearly outcomes."""
        created = self._generate_stories(
            year, events, deaths, exiles, governance,
            meta_events, subsim_log, births, infra_event, rng)

        self._spread_stories(year, active_colonists, rng)
        self._decay_stories()
        promoted = self._promote_stories(year, active_colonists)
        evicted = self._evict_weak_stories()

        return self._snapshot(year, active_colonists, created, promoted, evicted)

    def _generate_stories(
        self, year: int, events: list[dict], deaths: list[dict],
        exiles: list[dict], governance: dict | None,
        meta_events: list[dict], subsim_log: list[dict],
        births: list[dict], infra_event: dict | None,
        rng: random.Random,
    ) -> int:
        """Generate new stories from this year's events. Returns count created."""
        created = 0

        # Deaths always generate loss stories
        for death in deaths:
            if self._can_trigger("loss", year, cooldown=1):
                name = death.get("name", death.get("id", "unknown"))
                self.stories.append(Story(
                    id=self._make_id(), year_created=year,
                    source_event=f"death of {name}",
                    theme="loss", rememberers=[], tellings=0,
                    governance_stance="council",
                ))
                self._mark_trigger("loss", year)
                created += 1

        # Exiles generate exile stories
        for exile in exiles:
            if self._can_trigger("exile", year, cooldown=2):
                name = exile.get("name", exile.get("id", "unknown"))
                self.stories.append(Story(
                    id=self._make_id(), year_created=year,
                    source_event=f"exile of {name}",
                    theme="exile", rememberers=[], tellings=0,
                    governance_stance="anarchy",
                ))
                self._mark_trigger("exile", year)
                created += 1

        # Resource crises
        for ev in events:
            severity = ev.get("severity", 0.0)
            if severity > 0.6 and self._can_trigger("crisis", year):
                if rng.random() < 0.7:
                    self.stories.append(Story(
                        id=self._make_id(), year_created=year,
                        source_event=ev.get("description", ev.get("name", "crisis")),
                        theme="crisis", rememberers=[], tellings=0,
                        governance_stance="dictator",
                    ))
                    self._mark_trigger("crisis", year)
                    created += 1
                    break  # one crisis story per year max

        # Governance changes
        if governance and governance.get("passed"):
            if self._can_trigger("governance", year, cooldown=2):
                gov_type = governance.get("gov_type", "unknown")
                self.stories.append(Story(
                    id=self._make_id(), year_created=year,
                    source_event=f"governance changed to {gov_type}",
                    theme="governance", rememberers=[], tellings=0,
                    governance_stance=gov_type,
                ))
                self._mark_trigger("governance", year)
                created += 1

        # Deep subsims → discovery stories
        for entry in subsim_log:
            depth = entry.get("depth", 1)
            if depth >= 2 and self._can_trigger("discovery", year):
                if rng.random() < 0.8:
                    self.stories.append(Story(
                        id=self._make_id(), year_created=year,
                        source_event=f"depth-{depth} subsim by {entry.get('colonist_id', '?')}",
                        theme="discovery", rememberers=[], tellings=0,
                        governance_stance="ai_governor",
                    ))
                    self._mark_trigger("discovery", year)
                    created += 1
                    break

        # Meta-awareness → transcendence
        for meta in meta_events:
            if self._can_trigger("transcendence", year):
                self.stories.append(Story(
                    id=self._make_id(), year_created=year,
                    source_event=meta.get("insight", "meta-awareness moment"),
                    theme="transcendence", rememberers=[], tellings=0,
                    governance_stance="consensus",
                ))
                self._mark_trigger("transcendence", year)
                created += 1
                break

        # First birth → hope (one-time)
        if births and not self._first_birth_seen:
            self._first_birth_seen = True
            child_name = births[0].get("name", "a child")
            self.stories.append(Story(
                id=self._make_id(), year_created=year,
                source_event=f"first child born: {child_name}",
                theme="hope", rememberers=[], tellings=0,
                governance_stance="council",
            ))
            created += 1

        # Infrastructure milestone → progress
        if infra_event and infra_event.get("completed"):
            tech_name = infra_event.get("tech_id", "unknown tech")
            if self._can_trigger("progress", year, cooldown=5):
                self.stories.append(Story(
                    id=self._make_id(), year_created=year,
                    source_event=f"completed {tech_name}",
                    theme="progress", rememberers=[], tellings=0,
                    governance_stance="ai_governor",
                ))
                self._mark_trigger("progress", year)
                created += 1

        # Cooperation events (from actions logged outside — detect via
        # events with name containing "cooperation" or positive social)
        for ev in events:
            if "cooperation" in ev.get("name", "").lower():
                if self._can_trigger("cooperation", year, cooldown=3):
                    self.stories.append(Story(
                        id=self._make_id(), year_created=year,
                        source_event=ev.get("description", "cooperation event"),
                        theme="cooperation", rememberers=[], tellings=0,
                        governance_stance="consensus",
                    ))
                    self._mark_trigger("cooperation", year)
                    created += 1
                    break

        # Enforce cap: evict oldest non-myth stories if over limit
        non_myths = [s for s in self.stories if not s.is_myth]
        if len(non_myths) > MAX_ACTIVE_STORIES:
            non_myths.sort(key=lambda s: s.strength)
            for s in non_myths[:len(non_myths) - MAX_ACTIVE_STORIES]:
                self.stories.remove(s)

        return created

    def _spread_stories(self, year: int, active_colonists: list,
                        rng: random.Random) -> None:
        """Spread stories between colonists based on empathy and faith.

        Uses sampled propagation: each colonist tries to share one story.
        """
        if not active_colonists or not self.stories:
            return

        active_ids = [c.id for c in active_colonists]
        active_non_myth = [s for s in self.stories if not s.is_myth and s.strength > 0.1]
        if not active_non_myth:
            return

        for colonist in active_colonists:
            # Each colonist picks one story they know and tells it
            known = [s for s in active_non_myth if colonist.id in s.rememberers]
            if not known:
                # If colonist knows no stories, they might hear one
                unknown = [s for s in active_non_myth if colonist.id not in s.rememberers]
                if unknown and rng.random() < 0.3:
                    story = rng.choice(unknown)
                    story.rememberers.append(colonist.id)
                continue

            story = rng.choice(known)
            # Try to tell it to one other colonist
            others = [cid for cid in active_ids if cid != colonist.id
                       and cid not in story.rememberers]
            if not others:
                continue

            listener_id = rng.choice(others)
            listener = next((c for c in active_colonists if c.id == listener_id), None)
            if listener is None:
                continue

            # Spread probability based on teller's empathy and listener's faith
            spread_prob = (getattr(colonist.stats, "empathy", 0.5) * 0.4 +
                          getattr(listener.stats, "faith", 0.5) * 0.3 +
                          story.strength * 0.3)
            if rng.random() < spread_prob:
                story.rememberers.append(listener_id)
                story.tellings += 1
                # Retelling reinforces strength slightly
                story.strength = min(1.0, story.strength + 0.02)

    def _decay_stories(self) -> None:
        """Apply yearly decay to all non-myth stories."""
        for story in self.stories:
            if not story.is_myth:
                story.strength *= STORY_DECAY_RATE

    def _promote_stories(self, year: int, active_colonists: list) -> int:
        """Promote qualifying stories to myths. Returns count promoted."""
        if not active_colonists:
            return 0
        colony_size = len(active_colonists)
        promoted = 0
        for story in self.stories:
            if story.is_myth:
                continue
            spread_fraction = len(story.rememberers) / max(1, colony_size)
            if (story.strength >= MYTH_PROMOTION_THRESHOLD and
                    spread_fraction >= MYTH_SPREAD_THRESHOLD and
                    story.tellings >= MYTH_TELLING_THRESHOLD):
                story.is_myth = True
                story.strength = 1.0  # myths start at full strength
                promoted += 1
        return promoted

    def _evict_weak_stories(self) -> int:
        """Remove non-myth stories that have decayed below threshold."""
        before = len(self.stories)
        self.stories = [s for s in self.stories
                        if s.is_myth or s.strength >= STORY_EVICTION_THRESHOLD]
        return before - len(self.stories)

    def teach_child(self, child_id: str, parent_ids: list[str],
                    rng: random.Random) -> None:
        """Teach a newborn child the colony's myths and some parent stories.

        Children inherit:
        - All myths (permanent cultural knowledge)
        - Strong parent stories (50% chance per story if strength > 0.3)
        """
        for story in self.stories:
            if story.is_myth and child_id not in story.rememberers:
                story.rememberers.append(child_id)
                continue
            # Inherit strong parent stories asymmetrically
            if not story.is_myth and story.strength > 0.3:
                parent_knows = any(pid in story.rememberers for pid in parent_ids)
                if parent_knows and rng.random() < 0.5:
                    if child_id not in story.rememberers:
                        story.rememberers.append(child_id)

    def _snapshot(self, year: int, active_colonists: list,
                  created: int, promoted: int, evicted: int) -> CultureSnapshot:
        """Build a summary snapshot of cultural state."""
        colony_size = max(1, len(active_colonists))
        myths = [s for s in self.stories if s.is_myth]
        active = [s for s in self.stories if not s.is_myth]

        # Dominant theme: most common theme among all stories
        theme_counts: dict[str, int] = {}
        for s in self.stories:
            theme_counts[s.theme] = theme_counts.get(s.theme, 0) + 1
        dominant = max(theme_counts, key=theme_counts.get) if theme_counts else "none"

        # Cultural cohesion: fraction of stories known by >50% of colony
        shared = sum(1 for s in self.stories
                     if len(s.rememberers) > colony_size * 0.5)
        total = max(1, len(self.stories))
        cohesion = shared / total

        return CultureSnapshot(
            active_stories=len(active),
            myths=len(myths),
            dominant_theme=dominant,
            cultural_cohesion=cohesion,
            governance_signals=self.governance_signals(year),
            stories_created=created,
            stories_promoted=promoted,
            stories_evicted=evicted,
        )

    def to_dict(self) -> dict:
        """Full serialization of oral tradition state."""
        return {
            "stories": [s.to_dict() for s in self.stories],
            "total_stories_ever": self._next_id,
            "first_birth_seen": self._first_birth_seen,
            "myths": [s.to_dict() for s in self.stories if s.is_myth],
        }
