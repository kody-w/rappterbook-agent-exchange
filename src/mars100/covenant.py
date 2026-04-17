"""
Covenant engine for Mars-100.

Persistent, executable LisPy programs that govern the colony autonomously.
Covenants are the colony's "living laws" — s-expressions evaluated each year
against colony state, returning resource deltas that are applied with clamping.

Covenants are:
- Drafted by colonists based on personality and crisis conditions
- Voted on through governance-aware approval
- Executed each year as pure LisPy evaluations (no mutation, no I/O)
- Logged for archaeology — every execution is recorded
- Revokable through governance vote
- Auto-suspended after repeated runtime errors

The homoiconic property of LisPy means the colony's laws ARE executable
programs — data and code are the same structure.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.lispy_vm import LispyError, run as lispy_run

RESOURCE_NAMES = ("food", "water", "power", "air", "medicine")
MAX_ACTIVE_COVENANTS = 5
MAX_DELTA_PER_COVENANT = 0.05
GLOBAL_DELTA_CAP = 0.10
COVENANT_MAX_STEPS = 1000
COVENANT_MAX_DEPTH = 50
MAX_CONSECUTIVE_ERRORS = 3
MAX_LOG_ENTRIES = 20
DRAFT_COOLDOWN_YEARS = 3


@dataclass
class Covenant:
    """A persistent LisPy program that executes each year."""
    id: str
    author_id: str
    year_enacted: int
    expression: str
    description: str
    votes_for: int = 0
    votes_against: int = 0
    active: bool = True
    consecutive_errors: int = 0
    execution_log: list[dict] = field(default_factory=list)
    revoked_year: int | None = None
    sunset_year: int | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "id": self.id, "author_id": self.author_id,
            "year_enacted": self.year_enacted,
            "expression": self.expression,
            "description": self.description,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "active": self.active,
            "consecutive_errors": self.consecutive_errors,
            "execution_log": self.execution_log[-MAX_LOG_ENTRIES:],
            "revoked_year": self.revoked_year,
            "sunset_year": self.sunset_year,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Covenant:
        """Deserialize from dict."""
        return cls(
            id=d["id"], author_id=d["author_id"],
            year_enacted=d["year_enacted"],
            expression=d["expression"],
            description=d["description"],
            votes_for=d.get("votes_for", 0),
            votes_against=d.get("votes_against", 0),
            active=d.get("active", True),
            consecutive_errors=d.get("consecutive_errors", 0),
            execution_log=d.get("execution_log", []),
            revoked_year=d.get("revoked_year"),
            sunset_year=d.get("sunset_year"),
        )


@dataclass
class CovenantRegistry:
    """Registry of all covenants — active, revoked, and expired."""
    covenants: list[Covenant] = field(default_factory=list)
    last_draft_year: int = 0

    def active_covenants(self) -> list[Covenant]:
        """Return currently active covenants."""
        return [c for c in self.covenants if c.active]

    def active_count(self) -> int:
        """Count active covenants."""
        return len(self.active_covenants())

    def can_enact(self) -> bool:
        """Check if the colony can enact another covenant."""
        return self.active_count() < MAX_ACTIVE_COVENANTS

    def enact(self, covenant: Covenant) -> bool:
        """Add a covenant to the registry if there's room."""
        if not self.can_enact():
            return False
        self.covenants.append(covenant)
        return True

    def revoke(self, covenant_id: str, year: int) -> bool:
        """Revoke an active covenant."""
        for c in self.covenants:
            if c.id == covenant_id and c.active:
                c.active = False
                c.revoked_year = year
                return True
        return False

    def suspend(self, covenant_id: str) -> bool:
        """Auto-suspend a covenant due to repeated errors."""
        for c in self.covenants:
            if c.id == covenant_id and c.active:
                c.active = False
                return True
        return False

    def check_sunsets(self, year: int) -> list[str]:
        """Expire covenants past their sunset year."""
        expired: list[str] = []
        for c in self.covenants:
            if c.active and c.sunset_year is not None and year > c.sunset_year:
                c.active = False
                c.revoked_year = year
                expired.append(c.id)
        return expired

    def can_draft(self, year: int) -> bool:
        """Check if a new covenant can be drafted (cooldown)."""
        return year - self.last_draft_year >= DRAFT_COOLDOWN_YEARS

    def to_dict(self) -> dict:
        """Serialize the entire registry."""
        return {
            "covenants": [c.to_dict() for c in self.covenants],
            "last_draft_year": self.last_draft_year,
            "active_count": self.active_count(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CovenantRegistry:
        """Deserialize from dict."""
        reg = cls(last_draft_year=d.get("last_draft_year", 0))
        for cd in d.get("covenants", []):
            reg.covenants.append(Covenant.from_dict(cd))
        return reg


def _parse_deltas(result: Any) -> dict[str, float]:
    """Parse a LisPy result into resource deltas.

    Accepted return formats:
    - nil / None → no effect
    - number → tagged as _scalar for caller context
    - list of [symbol, delta] pairs → named resource deltas
    """
    if result is None:
        return {}
    if isinstance(result, (int, float)):
        return {"_scalar": float(result)}
    if isinstance(result, list):
        deltas: dict[str, float] = {}
        for item in result:
            if (isinstance(item, list) and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], (int, float))):
                name = item[0]
                if name in RESOURCE_NAMES:
                    deltas[name] = float(item[1])
        return deltas
    return {}


def _clamp_deltas(deltas: dict[str, float]) -> dict[str, float]:
    """Clamp each delta to ±MAX_DELTA_PER_COVENANT."""
    return {
        k: max(-MAX_DELTA_PER_COVENANT, min(MAX_DELTA_PER_COVENANT, v))
        for k, v in deltas.items()
    }


def _safe_serialize(value: Any) -> Any:
    """Make a value JSON-safe."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe_serialize(v) for v in value[:20]]
    return str(value)[:200]


def execute_covenant(
    covenant: Covenant,
    resources: dict[str, float],
    population: int,
    year: int,
    gov_type: str,
) -> dict:
    """Execute a single covenant and return its effect.

    Returns a dict with:
        deltas: dict[str, float] — clamped resource deltas
        error: str | None — error message if execution failed
        raw_result: Any — the raw LisPy return value
    """
    bindings: dict[str, Any] = {}
    for name in RESOURCE_NAMES:
        bindings[name] = resources.get(name, 0.5)
    bindings["population"] = population
    bindings["year"] = year

    try:
        raw = lispy_run(
            covenant.expression,
            extra_bindings=bindings,
            max_steps=COVENANT_MAX_STEPS,
            max_depth=COVENANT_MAX_DEPTH,
        )
    except LispyError as exc:
        return {"deltas": {}, "error": str(exc), "raw_result": None}
    except Exception as exc:
        return {"deltas": {}, "error": str(exc), "raw_result": None}

    deltas = _parse_deltas(raw)
    deltas.pop("_scalar", None)
    clamped = _clamp_deltas(deltas)
    return {"deltas": clamped, "error": None, "raw_result": _safe_serialize(raw)}


def tick_covenants(
    registry: CovenantRegistry,
    resources: dict[str, float],
    population: int,
    year: int,
    gov_type: str,
) -> dict:
    """Execute all active covenants for this year.

    All covenants evaluate against the SAME pre-covenant baseline.
    Aggregate deltas are globally clamped to ±GLOBAL_DELTA_CAP.
    """
    expired = registry.check_sunsets(year)
    executions: list[dict] = []
    aggregate: dict[str, float] = {name: 0.0 for name in RESOURCE_NAMES}

    for covenant in registry.active_covenants():
        result = execute_covenant(covenant, resources, population, year, gov_type)
        log_entry = {
            "year": year,
            "result": result["raw_result"],
            "error": result["error"],
            "deltas": result["deltas"],
        }
        covenant.execution_log.append(log_entry)
        if len(covenant.execution_log) > MAX_LOG_ENTRIES:
            covenant.execution_log = covenant.execution_log[-MAX_LOG_ENTRIES:]

        if result["error"]:
            covenant.consecutive_errors += 1
            if covenant.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                registry.suspend(covenant.id)
        else:
            covenant.consecutive_errors = 0
            for name, delta in result["deltas"].items():
                if name in RESOURCE_NAMES:
                    aggregate[name] += delta

        executions.append({
            "covenant_id": covenant.id,
            "description": covenant.description,
            **result,
        })

    clamped_aggregate = {
        name: max(-GLOBAL_DELTA_CAP, min(GLOBAL_DELTA_CAP, val))
        for name, val in aggregate.items()
    }

    return {
        "aggregate_deltas": clamped_aggregate,
        "executions": executions,
        "expired": expired,
    }


# ── Covenant drafting ─────────────────────────────────────────────

# Templates keyed by dominant stat.  LisPy expressions use quoted symbols
# ('food, 'water etc.) because the VM has no string literals — bare
# symbols would evaluate to their numeric binding value.
COVENANT_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "empathy": [
        (
            "(if (< food 0.3) (list (list 'food 0.03)) nil)",
            "Communal rationing: boost food when reserves are critical",
        ),
        (
            "(if (< medicine 0.3) (list (list 'medicine 0.02)) nil)",
            "Healing mandate: prioritize medicine when health is low",
        ),
    ],
    "paranoia": [
        (
            "(if (< air 0.2) (list (list 'air 0.04) (list 'power -0.01)) nil)",
            "Emergency O2 protocol: divert power to air generation",
        ),
        (
            "(if (< water 0.25) (list (list 'water 0.03) (list 'food -0.01)) nil)",
            "Water emergency protocol: sacrifice food for water reclamation",
        ),
    ],
    "faith": [
        (
            "(if (< medicine 0.35) (list (list 'medicine 0.02)) nil)",
            "Spiritual healing rites: community care boosts medicine",
        ),
        (
            "(if (> year 50) (list (list 'air 0.01) (list 'water 0.01)) nil)",
            "Elder wisdom: long-lived colonies develop better resource habits",
        ),
    ],
    "resolve": [
        (
            "(if (> population 15) (list (list 'food 0.02) (list 'power 0.01)) nil)",
            "Mandatory work shifts: large colonies produce more efficiently",
        ),
        (
            "(list (list 'food 0.01) (list 'water 0.01))",
            "Discipline protocol: steady gains from organized labor",
        ),
    ],
    "improvisation": [
        (
            "(if (> year 30) (list (list 'power 0.02) (list 'water 0.01)) nil)",
            "Innovation dividend: experienced colonies develop better tech",
        ),
        (
            "(if (< power 0.3) (list (list 'power 0.03)) nil)",
            "Emergency engineering: improvise power solutions in crisis",
        ),
    ],
    "hoarding": [
        (
            "(if (> food 0.6) (list (list 'food -0.02) (list 'medicine 0.02)) nil)",
            "Surplus trade: convert excess food to medicine",
        ),
        (
            "(if (> water 0.6) (list (list 'water -0.01) (list 'air 0.01)) nil)",
            "Resource arbitrage: convert surplus water to air processing",
        ),
    ],
}


def draft_covenant(
    colonist_id: str,
    dominant_stat: str,
    year: int,
    registry: CovenantRegistry,
    rng: random.Random,
) -> Covenant | None:
    """Draft a covenant based on colonist personality.

    Returns None if no covenant can be drafted (cooldown, cap, or
    duplicate suppression).
    """
    if not registry.can_enact():
        return None
    if not registry.can_draft(year):
        return None

    templates = COVENANT_TEMPLATES.get(dominant_stat, COVENANT_TEMPLATES["resolve"])
    expression, description = rng.choice(templates)

    # Duplicate suppression
    for existing in registry.covenants:
        if existing.expression == expression:
            return None

    covenant_id = f"cov-y{year}-{len(registry.covenants)}"
    return Covenant(
        id=covenant_id,
        author_id=colonist_id,
        year_enacted=year,
        expression=expression,
        description=description,
        sunset_year=year + rng.randint(15, 40),
    )


def vote_covenant(
    covenant: Covenant,
    active_ids: list[str],
    gov_type: str,
    leader_id: str | None,
    council_ids: list[str],
    social_trust: dict[str, float],
    rng: random.Random,
) -> bool:
    """Colony votes on a covenant. Governance-aware approval."""
    if not active_ids:
        return False

    voters = [cid for cid in active_ids if cid != covenant.author_id]
    if not voters:
        covenant.votes_for = 1
        return True

    votes_for = 1  # author always votes for
    votes_against = 0

    for voter_id in voters:
        trust = social_trust.get(voter_id, 0.5)
        base_chance = 0.45 + trust * 0.3
        if rng.random() < base_chance:
            votes_for += 1
        else:
            votes_against += 1

    covenant.votes_for = votes_for
    covenant.votes_against = votes_against

    if gov_type == "anarchy":
        return votes_for > votes_against
    elif gov_type == "dictator":
        if leader_id and leader_id != covenant.author_id:
            leader_trust = social_trust.get(leader_id, 0.5)
            if leader_trust < 0.4:
                return False
        return votes_for > votes_against
    elif gov_type == "council":
        if council_ids:
            council_for = sum(1 for cid in council_ids
                              if social_trust.get(cid, 0.5) >= 0.4)
            return council_for > len(council_ids) / 2
        return votes_for > votes_against
    elif gov_type == "consensus":
        total = votes_for + votes_against
        return total > 0 and votes_for / total >= 0.75
    elif gov_type in ("ai_governor", "lottery"):
        return votes_for > votes_against
    else:
        return votes_for > votes_against
