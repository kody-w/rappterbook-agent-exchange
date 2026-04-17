"""
Memetics engine for Mars-100.

Models ideas (memes) as living entities that propagate through the
colony's social graph, get inherited by children, and bias colonist
behavior.  Memes are the colony's *cultural DNA* — they persist across
generations even when their originator dies.

V1 scope: genesis (governance + crisis), propagation, inheritance,
carrier cleanup, displacement.  No mutation or fitness tracking yet.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

MEME_TYPES = ("governance_norm", "crisis_response", "subsim_insight")
MAX_MEMES_PER_COLONIST = 5
PROPAGATION_BASE = 0.25
CRISIS_SEVERITY_THRESHOLD = 0.6


@dataclass
class Meme:
    """A single cultural unit that propagates through the colony."""
    id: str
    name: str
    meme_type: str
    origin_year: int
    origin_colonist: str
    content: dict[str, float]
    virality: float
    carriers: list[str] = field(default_factory=list)
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "meme_type": self.meme_type,
            "origin_year": self.origin_year,
            "origin_colonist": self.origin_colonist,
            "content": dict(self.content),
            "virality": self.virality,
            "carriers": sorted(self.carriers),
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Meme:
        return cls(
            id=d["id"], name=d["name"], meme_type=d["meme_type"],
            origin_year=d["origin_year"],
            origin_colonist=d["origin_colonist"],
            content=dict(d.get("content", {})),
            virality=d.get("virality", 0.3),
            carriers=list(d.get("carriers", [])),
            parent_id=d.get("parent_id"),
        )

    @property
    def carrier_count(self) -> int:
        return len(self.carriers)

    def salience(self, current_year: int) -> float:
        """Score for displacement: higher = harder to forget."""
        age = max(1, current_year - self.origin_year)
        recency = 1.0 / (1.0 + age * 0.1)
        spread = min(1.0, self.carrier_count / 5.0)
        return self.virality * 0.4 + recency * 0.3 + spread * 0.3


@dataclass
class MemePool:
    """Colony-wide meme pool with per-year lifecycle."""
    memes: dict[str, Meme] = field(default_factory=dict)
    _next_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_memes": len(self.memes),
            "active_memes": sum(1 for m in self.memes.values() if m.carriers),
            "memes": {mid: m.to_dict() for mid, m in sorted(self.memes.items())},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemePool:
        pool = cls()
        for mid, md in d.get("memes", {}).items():
            pool.memes[mid] = Meme.from_dict(md)
        if pool.memes:
            max_num = max(
                (int(mid.split("-")[1]) for mid in pool.memes if "-" in mid),
                default=0,
            )
            pool._next_id = max_num + 1
        return pool

    def _new_id(self) -> str:
        mid = f"meme-{self._next_id}"
        self._next_id += 1
        return mid

    def carrier_memes(self, colonist_id: str) -> list[Meme]:
        """Get all memes carried by a colonist, sorted by id for determinism."""
        return sorted(
            [m for m in self.memes.values() if colonist_id in m.carriers],
            key=lambda m: m.id,
        )

    def carrier_count(self, colonist_id: str) -> int:
        return sum(1 for m in self.memes.values() if colonist_id in m.carriers)

    # ------------------------------------------------------------------
    # Genesis: create new memes from colony events
    # ------------------------------------------------------------------

    def create_governance_meme(
        self, year: int, proposer_id: str, gov_type: str, rng: random.Random,
    ) -> Meme:
        """Create a meme from a passed governance proposal."""
        content = _governance_content(gov_type, rng)
        meme = Meme(
            id=self._new_id(),
            name=f"{gov_type} norm (y{year})",
            meme_type="governance_norm",
            origin_year=year,
            origin_colonist=proposer_id,
            content=content,
            virality=rng.uniform(0.3, 0.6),
            carriers=[proposer_id],
        )
        self.memes[meme.id] = meme
        return meme

    def create_crisis_meme(
        self, year: int, survivors: list[str], event_name: str,
        rng: random.Random,
    ) -> Meme:
        """Create a meme from surviving a severe crisis."""
        if not survivors:
            return Meme(id="none", name="", meme_type="crisis_response",
                        origin_year=year, origin_colonist="",
                        content={}, virality=0.0)
        content = _crisis_content(event_name, rng)
        originator = rng.choice(sorted(survivors))
        meme = Meme(
            id=self._new_id(),
            name=f"{event_name} lesson (y{year})",
            meme_type="crisis_response",
            origin_year=year,
            origin_colonist=originator,
            content=content,
            virality=rng.uniform(0.4, 0.7),
            carriers=sorted(survivors),
        )
        self.memes[meme.id] = meme
        return meme

    def create_subsim_meme(
        self, year: int, colonist_id: str, insight_text: str,
        rng: random.Random,
    ) -> Meme:
        """Create a meme from a promoted sub-sim insight."""
        content: dict[str, float] = {
            "cooperate": rng.uniform(0.02, 0.08),
            "research": rng.uniform(0.02, 0.06),
        }
        meme = Meme(
            id=self._new_id(),
            name=f"insight (y{year})",
            meme_type="subsim_insight",
            origin_year=year,
            origin_colonist=colonist_id,
            content=content,
            virality=rng.uniform(0.2, 0.5),
            carriers=[colonist_id],
        )
        self.memes[meme.id] = meme
        return meme

    # ------------------------------------------------------------------
    # Propagation: memes spread along social-graph edges
    # ------------------------------------------------------------------

    def propagate(
        self,
        year: int,
        active_ids: list[str],
        social_get: Any,
        stats_get: Any,
        rng: random.Random,
    ) -> list[dict[str, str]]:
        """Spread memes along social edges.  Returns propagation log.

        *social_get(a, b)* returns a Relationship with .trust.
        *stats_get(cid)* returns a ColonistStats with .empathy, .paranoia.
        """
        log: list[dict[str, str]] = []
        sorted_ids = sorted(active_ids)
        for carrier_id in sorted_ids:
            carried = self.carrier_memes(carrier_id)
            for meme in carried:
                for target_id in sorted_ids:
                    if target_id == carrier_id:
                        continue
                    if target_id in meme.carriers:
                        continue
                    rel = social_get(carrier_id, target_id)
                    carrier_stats = stats_get(carrier_id)
                    target_stats = stats_get(target_id)
                    if carrier_stats is None or target_stats is None:
                        continue
                    prob = (
                        meme.virality
                        * PROPAGATION_BASE
                        * rel.trust
                        * carrier_stats.empathy
                        * (1.0 - target_stats.paranoia * 0.3)
                    )
                    if rng.random() < prob:
                        if self._try_adopt(target_id, meme, year):
                            log.append({
                                "meme": meme.id,
                                "from": carrier_id,
                                "to": target_id,
                            })
        return log

    def _try_adopt(self, colonist_id: str, meme: Meme, year: int) -> bool:
        """Attempt to add meme to colonist. Displace weakest if at cap."""
        if colonist_id in meme.carriers:
            return False
        current = self.carrier_memes(colonist_id)
        if len(current) < MAX_MEMES_PER_COLONIST:
            meme.carriers.append(colonist_id)
            meme.carriers.sort()
            return True
        weakest = min(current, key=lambda m: m.salience(year))
        if weakest.salience(year) < meme.salience(year):
            weakest.carriers = [c for c in weakest.carriers if c != colonist_id]
            meme.carriers.append(colonist_id)
            meme.carriers.sort()
            return True
        return False

    # ------------------------------------------------------------------
    # Inheritance: newborns get parents' memes
    # ------------------------------------------------------------------

    def inherit_memes(
        self,
        child_id: str,
        parent_a_id: str,
        parent_b_id: str,
        year: int,
        rng: random.Random,
    ) -> list[str]:
        """Child inherits a sample of parent memes (cultural DNA).

        Returns list of inherited meme ids.
        """
        parent_memes = set()
        for m in self.carrier_memes(parent_a_id):
            parent_memes.add(m.id)
        for m in self.carrier_memes(parent_b_id):
            parent_memes.add(m.id)
        candidates = sorted(parent_memes)
        inherited: list[str] = []
        for mid in candidates:
            if len(inherited) >= MAX_MEMES_PER_COLONIST:
                break
            meme = self.memes.get(mid)
            if meme and rng.random() < 0.7:
                meme.carriers.append(child_id)
                meme.carriers.sort()
                inherited.append(mid)
        return inherited

    # ------------------------------------------------------------------
    # Carrier cleanup: remove dead/exiled colonists
    # ------------------------------------------------------------------

    def deactivate_carrier(self, colonist_id: str) -> None:
        """Remove a colonist from all meme carrier lists (death/exile)."""
        for meme in self.memes.values():
            meme.carriers = [c for c in meme.carriers if c != colonist_id]

    # ------------------------------------------------------------------
    # Behavioral effect: compute action-weight deltas from carried memes
    # ------------------------------------------------------------------

    def action_weight_deltas(self, colonist_id: str) -> dict[str, float]:
        """Sum action-weight deltas from all memes carried by a colonist."""
        deltas: dict[str, float] = {}
        for meme in self.carrier_memes(colonist_id):
            for action, value in meme.content.items():
                deltas[action] = deltas.get(action, 0.0) + value
        return deltas

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self, year: int) -> dict[str, Any]:
        """Compute summary stats for the current meme pool."""
        active = [m for m in self.memes.values() if m.carriers]
        extinct = [m for m in self.memes.values() if not m.carriers]
        by_type: dict[str, int] = {}
        for m in active:
            by_type[m.meme_type] = by_type.get(m.meme_type, 0) + 1
        oldest = min((m.origin_year for m in active), default=year)
        return {
            "total": len(self.memes),
            "active": len(active),
            "extinct": len(extinct),
            "by_type": by_type,
            "oldest_active_origin": oldest,
            "avg_carriers": (
                sum(m.carrier_count for m in active) / max(1, len(active))
            ),
        }


# ======================================================================
# Content generators: what behavioral effects does each meme type carry?
# ======================================================================

def _governance_content(gov_type: str, rng: random.Random) -> dict[str, float]:
    """Generate action-weight content for a governance meme."""
    base: dict[str, float] = {}
    if gov_type == "council":
        base["mediate"] = rng.uniform(0.05, 0.12)
        base["cooperate"] = rng.uniform(0.03, 0.08)
    elif gov_type == "dictator":
        base["hoard"] = rng.uniform(0.03, 0.08)
        base["cooperate"] = rng.uniform(-0.05, 0.0)
    elif gov_type == "lottery":
        base["pray"] = rng.uniform(0.03, 0.08)
        base["explore"] = rng.uniform(0.02, 0.06)
    elif gov_type == "consensus":
        base["mediate"] = rng.uniform(0.06, 0.15)
        base["cooperate"] = rng.uniform(0.05, 0.1)
    elif gov_type == "ai_governor":
        base["code"] = rng.uniform(0.05, 0.1)
        base["research"] = rng.uniform(0.04, 0.08)
    elif gov_type == "anarchy":
        base["explore"] = rng.uniform(0.05, 0.1)
        base["sabotage"] = rng.uniform(0.01, 0.04)
    return base


def _crisis_content(event_name: str, rng: random.Random) -> dict[str, float]:
    """Generate action-weight content for a crisis-response meme."""
    base: dict[str, float] = {}
    if event_name in ("dust_storm", "solar_flare"):
        base["cooperate"] = rng.uniform(0.04, 0.1)
        base["hoard"] = rng.uniform(0.02, 0.06)
    elif event_name == "equipment_failure":
        base["code"] = rng.uniform(0.05, 0.1)
        base["research"] = rng.uniform(0.03, 0.07)
    elif event_name == "epidemic":
        base["cooperate"] = rng.uniform(0.06, 0.12)
        base["pray"] = rng.uniform(0.02, 0.06)
    elif event_name == "colonist_conflict":
        base["mediate"] = rng.uniform(0.08, 0.15)
    else:
        base["cooperate"] = rng.uniform(0.02, 0.06)
        base["explore"] = rng.uniform(0.02, 0.06)
    return base
