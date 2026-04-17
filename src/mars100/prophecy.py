"""Prophecy Engine — colonists make testable predictions about the future.

Predictions are LisPy predicates (e.g. '(> food 0.5)') evaluated against
actual resource/survival state when the target year arrives.  Two types:
  - 'intuitive' — stat-driven (faith, paranoia, resolve)
  - 'empirical' — derived from sub-simulation results

Accuracy feeds back into governance influence: prophets whose predictions
come true gain social weight; bad prophets lose credibility.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone


_RESOURCE_CRISIS = [
    ("(< food 0.2)", "food", "famine"),
    ("(< water 0.2)", "water", "drought"),
    ("(< air 0.15)", "air", "asphyxiation risk"),
    ("(< power 0.2)", "power", "blackout"),
    ("(< medicine 0.15)", "medicine", "plague risk"),
]

_RESOURCE_SURPLUS = [
    ("(> food 0.7)", "food", "abundance"),
    ("(> water 0.7)", "water", "aquifer boom"),
    ("(> power 0.8)", "power", "energy surplus"),
]

_SURVIVAL = [
    ("(> alive_count 5)", None, "colony survives"),
    ("(< alive_count 4)", None, "colony collapse"),
]


@dataclass
class Prophecy:
    """A single testable prediction made by a colonist."""
    author_id: str
    expression: str
    target_year: int
    created_year: int
    prophecy_type: str
    resource_tag: str | None = None
    description: str = ""
    resolved: bool = False
    outcome: bool | None = None
    resolution_year: int | None = None

    def to_dict(self) -> dict:
        return {
            "author_id": self.author_id, "expression": self.expression,
            "target_year": self.target_year, "created_year": self.created_year,
            "type": self.prophecy_type, "resource_tag": self.resource_tag,
            "description": self.description, "resolved": self.resolved,
            "outcome": self.outcome, "resolution_year": self.resolution_year,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Prophecy":
        return cls(
            author_id=d["author_id"], expression=d["expression"],
            target_year=d["target_year"], created_year=d["created_year"],
            prophecy_type=d.get("type", "intuitive"),
            resource_tag=d.get("resource_tag"),
            description=d.get("description", ""),
            resolved=d.get("resolved", False),
            outcome=d.get("outcome"),
            resolution_year=d.get("resolution_year"),
        )


class ProphecyEngine:
    """Manages prophecy creation, resolution, and accuracy tracking."""

    MAX_PER_YEAR = 2

    def __init__(self) -> None:
        self.prophecies: list[Prophecy] = []
        self._year_count = 0

    def begin_year(self) -> None:
        """Reset per-year counters."""
        self._year_count = 0

    def make_prophecy(self, colonist_id: str, stats: dict[str, float],
                      skills: dict[str, float], resources: dict[str, float],
                      current_year: int, from_subsim: bool,
                      rng: random.Random) -> Prophecy | None:
        """Maybe generate a prophecy. Returns Prophecy or None."""
        if self._year_count >= self.MAX_PER_YEAR:
            return None
        prob = 0.12 + stats.get("faith", 0) * 0.08 + stats.get("paranoia", 0) * 0.06
        if from_subsim:
            prob += 0.10
        if rng.random() > prob:
            return None
        low_resources = [n for n, v in resources.items() if v < 0.3]
        if low_resources and rng.random() < 0.6:
            templates = [t for t in _RESOURCE_CRISIS if t[1] in low_resources]
            if not templates:
                templates = list(_RESOURCE_CRISIS)
        elif rng.random() < 0.3:
            templates = list(_RESOURCE_SURPLUS)
        else:
            templates = list(_RESOURCE_CRISIS) + list(_SURVIVAL)
        if not templates:
            return None
        expr, resource_tag, desc = rng.choice(templates)
        target_year = current_year + rng.randint(1, 5)
        ptype = "empirical" if from_subsim else "intuitive"
        p = Prophecy(author_id=colonist_id, expression=expr,
                     target_year=target_year, created_year=current_year,
                     prophecy_type=ptype, resource_tag=resource_tag,
                     description=desc)
        self.prophecies.append(p)
        self._year_count += 1
        return p

    def resolve(self, current_year: int, resources: dict[str, float],
                active_ids: set[str]) -> list[dict]:
        """Resolve prophecies whose target_year == current_year."""
        resolutions: list[dict] = []
        bindings = dict(resources)
        bindings["alive_count"] = len(active_ids)
        for p in self.prophecies:
            if p.resolved or p.target_year != current_year:
                continue
            p.resolved = True
            p.resolution_year = current_year
            if p.author_id not in active_ids:
                p.outcome = False
                resolutions.append(p.to_dict())
                continue
            try:
                p.outcome = bool(_safe_eval_predicate(p.expression, bindings))
            except Exception:
                p.outcome = False
            resolutions.append(p.to_dict())
        return resolutions

    def accuracy(self, colonist_id: str, window: int = 20) -> float:
        """Rolling accuracy (last window resolved). 0.5 if no history."""
        resolved = [p for p in self.prophecies
                    if p.author_id == colonist_id and p.resolved and p.outcome is not None]
        if not resolved:
            return 0.5
        recent = resolved[-window:]
        return sum(1 for p in recent if p.outcome) / len(recent)

    def influence_modifier(self, colonist_id: str) -> float:
        """Score modifier [-0.3, +0.3] based on prophecy accuracy."""
        acc = self.accuracy(colonist_id)
        return max(-0.3, min(0.3, (acc - 0.5) * 0.6))

    def active_warnings(self, current_year: int) -> list[Prophecy]:
        """Return unresolved crisis prophecies whose target is upcoming."""
        return [p for p in self.prophecies
                if not p.resolved and p.target_year >= current_year
                and p.resource_tag is not None and p.expression.startswith("(<")]

    def warning_resources(self, current_year: int) -> set[str]:
        """Resource names with active crisis warnings."""
        return {p.resource_tag for p in self.active_warnings(current_year)
                if p.resource_tag is not None}

    def summary(self) -> dict:
        """Summary statistics for the full simulation."""
        total = len(self.prophecies)
        resolved = [p for p in self.prophecies if p.resolved]
        correct = [p for p in resolved if p.outcome is True]
        intuitive = [p for p in self.prophecies if p.prophecy_type == "intuitive"]
        empirical = [p for p in self.prophecies if p.prophecy_type == "empirical"]
        i_res = [p for p in intuitive if p.resolved]
        e_res = [p for p in empirical if p.resolved]
        i_cor = [p for p in i_res if p.outcome is True]
        e_cor = [p for p in e_res if p.outcome is True]
        return {
            "total": total, "resolved": len(resolved), "correct": len(correct),
            "accuracy": (len(correct) / len(resolved)) if resolved else 0.0,
            "intuitive": {"total": len(intuitive), "resolved": len(i_res),
                          "correct": len(i_cor),
                          "accuracy": (len(i_cor) / len(i_res)) if i_res else 0.0},
            "empirical": {"total": len(empirical), "resolved": len(e_res),
                          "correct": len(e_cor),
                          "accuracy": (len(e_cor) / len(e_res)) if e_res else 0.0},
        }


def _safe_eval_predicate(expr: str, bindings: dict[str, float]) -> bool:
    """Evaluate a simple LisPy predicate: (< x y), (> x y), etc."""
    expr = expr.strip()
    if not expr.startswith("(") or not expr.endswith(")"):
        raise ValueError(f"Invalid expression: {expr}")
    parts = expr[1:-1].strip().split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 parts: {expr}")
    op, left_s, right_s = parts
    left = _resolve_value(left_s, bindings)
    right = _resolve_value(right_s, bindings)
    ops = {"<": left < right, ">": left > right, "<=": left <= right,
           ">=": left >= right, "=": abs(left - right) < 1e-9}
    if op not in ops:
        raise ValueError(f"Unknown operator: {op}")
    return ops[op]


def _resolve_value(token: str, bindings: dict[str, float]) -> float:
    """Resolve token to float (literal or binding name)."""
    try:
        return float(token)
    except ValueError:
        if token in bindings:
            return bindings[token]
        raise ValueError(f"Unknown variable: {token}")
