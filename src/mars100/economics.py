"""Economics organ for Mars-100 colony simulation.

Models labor output, wealth distribution, inequality measurement,
and economic pressure on the colony.  Wealth represents a colonist's
accumulated personal advantage (tools, stored rations, social capital)
— it does NOT duplicate shared colony resources.

Two allocation models (soft-defaulted from governance type):
  - Communal: surplus shared equally regardless of labor
  - Market:   surplus distributed proportional to labor contribution

Key dynamics:
  - Labor score = base + relevant skill for chosen action
  - Wealth grows from surplus distribution, decays from consumption
  - Gini coefficient (normalized for finite populations) measures inequality
  - Economic pressure = f(gini, resource scarcity) → feeds social/governance
  - Death/exile: estate redistributed equally among active colonists
  - Birth/immigration: start with minimal wealth
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

BASE_LABOR = 0.5
WEALTH_DECAY_RATE = 0.05
SURPLUS_RATE = 0.1
HOARDING_BONUS = 0.3
FOUNDER_WEALTH = 0.3
CHILD_WEALTH = 0.0
IMMIGRANT_WEALTH = 0.2

ALLOCATION_MODELS = ("communal", "market")

GOVERNANCE_MODEL_DEFAULTS: dict[str, str] = {
    "anarchy": "communal",
    "council": "communal",
    "consensus": "communal",
    "dictator": "market",
    "lottery": "communal",
    "ai_governor": "market",
}

ACTION_SKILL_MAP: dict[str, str] = {
    "terraform": "terraforming",
    "farm": "hydroponics",
    "mediate": "mediation",
    "code": "coding",
    "pray": "prayer",
    "sabotage": "sabotage",
    "research": "coding",
    "cooperate": "mediation",
    "hoard": "terraforming",
    "explore": "terraforming",
    "rest": "mediation",
}


@dataclass
class EconomicState:
    """Colony economic state."""
    wealth: dict[str, float] = field(default_factory=dict)
    model: str = "communal"
    gini: float = 0.0
    pressure: float = 0.0
    total_surplus_distributed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "wealth": dict(self.wealth),
            "model": self.model,
            "gini": round(self.gini, 4),
            "pressure": round(self.pressure, 4),
            "total_surplus_distributed": round(self.total_surplus_distributed, 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EconomicState:
        """Deserialize from dict."""
        return cls(
            wealth=dict(d.get("wealth", {})),
            model=d.get("model", "communal"),
            gini=d.get("gini", 0.0),
            pressure=d.get("pressure", 0.0),
            total_surplus_distributed=d.get("total_surplus_distributed", 0.0),
        )


@dataclass
class EconomicTickResult:
    """Result of one year's economic tick."""
    labor: dict[str, float]
    surplus: float
    distribution: dict[str, float]
    gini_before: float
    gini_after: float
    pressure: float
    model_used: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "labor": {k: round(v, 4) for k, v in self.labor.items()},
            "surplus": round(self.surplus, 4),
            "distribution": {k: round(v, 4) for k, v in self.distribution.items()},
            "gini_before": round(self.gini_before, 4),
            "gini_after": round(self.gini_after, 4),
            "pressure": round(self.pressure, 4),
            "model_used": self.model_used,
        }


def compute_labor(action: str, skills_dict: dict[str, float],
                  resolve: float) -> float:
    """Compute a colonist's labor output for one year.

    Labor = base + relevant_skill * 0.5 + resolve * 0.2.
    """
    skill_name = ACTION_SKILL_MAP.get(action, "")
    skill_level = skills_dict.get(skill_name, 0.0) if skill_name else 0.0
    return BASE_LABOR + skill_level * 0.5 + resolve * 0.2


def compute_gini(wealth_values: list[float]) -> float:
    """Compute normalized Gini coefficient.

    Returns 0.0 for perfect equality, approaches 1.0 for maximal inequality.
    Uses finite-population normalization: gini * n / (n - 1).
    """
    n = len(wealth_values)
    if n < 2:
        return 0.0
    sorted_w = sorted(wealth_values)
    total = sum(sorted_w)
    if total <= 0:
        return 0.0
    weighted_sum = 0.0
    for i, w in enumerate(sorted_w):
        weighted_sum += (2 * (i + 1) - n - 1) * w
    raw_gini = weighted_sum / (n * total)
    normalized = raw_gini * n / (n - 1)
    return max(0.0, min(1.0, normalized))


def default_model_for_governance(gov_type: str) -> str:
    """Soft default mapping from governance type to economic model."""
    return GOVERNANCE_MODEL_DEFAULTS.get(gov_type, "communal")


def distribute_surplus(model: str, surplus: float,
                       labor: dict[str, float],
                       active_ids: list[str]) -> dict[str, float]:
    """Distribute annual surplus among active colonists.

    Under communal model: equal shares.
    Under market model: proportional to labor contribution.
    """
    if not active_ids or surplus <= 0:
        return {cid: 0.0 for cid in active_ids}

    if model == "market":
        total_labor = sum(labor.get(cid, BASE_LABOR) for cid in active_ids)
        if total_labor <= 0:
            total_labor = len(active_ids) * BASE_LABOR
        return {
            cid: surplus * (labor.get(cid, BASE_LABOR) / total_labor)
            for cid in active_ids
        }
    else:
        share = surplus / len(active_ids)
        return {cid: share for cid in active_ids}


def compute_economic_pressure(gini: float, resource_avg: float) -> float:
    """Compute economic pressure from inequality and resource scarcity.

    High inequality hurts more when resources are scarce.
    Returns 0.0-1.0 pressure value.
    """
    scarcity = max(0.0, 1.0 - resource_avg)
    raw_pressure = gini * 0.6 + gini * scarcity * 0.4
    return max(0.0, min(1.0, raw_pressure))


def initialize_wealth(colonist_ids: list[str],
                      initial_value: float = FOUNDER_WEALTH) -> dict[str, float]:
    """Initialize wealth for a set of colonists."""
    return {cid: initial_value for cid in colonist_ids}


def handle_birth(econ: EconomicState, colonist_id: str,
                 is_immigrant: bool = False) -> None:
    """Initialize wealth for a newborn or immigrant."""
    econ.wealth[colonist_id] = IMMIGRANT_WEALTH if is_immigrant else CHILD_WEALTH


def handle_death(econ: EconomicState, colonist_id: str,
                 active_ids: list[str]) -> None:
    """Redistribute a dead/exiled colonist's estate equally among survivors."""
    estate = econ.wealth.pop(colonist_id, 0.0)
    survivors = [cid for cid in active_ids if cid != colonist_id]
    if survivors and estate > 0:
        share = estate / len(survivors)
        for cid in survivors:
            econ.wealth[cid] = min(1.0, econ.wealth.get(cid, 0.0) + share)


def tick_economics(
    econ: EconomicState,
    actions: dict[str, str],
    colonist_data: list[dict[str, Any]],
    resource_avg: float,
    gov_type: str,
    rng: random.Random,
) -> EconomicTickResult:
    """Advance the colony economy by one year.

    Args:
        econ: Mutable economic state.
        actions: colonist_id → chosen action this year.
        colonist_data: List of colonist dicts with id, stats, skills.
        resource_avg: Average colony resource level (0-1).
        gov_type: Current governance type (for model default).
        rng: Dedicated economics RNG.

    Returns:
        EconomicTickResult with labor, distribution, and inequality data.
    """
    active_ids = [c["id"] for c in colonist_data]

    for cid in active_ids:
        if cid not in econ.wealth:
            econ.wealth[cid] = FOUNDER_WEALTH

    econ.model = default_model_for_governance(gov_type)

    labor: dict[str, float] = {}
    for c in colonist_data:
        cid = c["id"]
        action = actions.get(cid, "rest")
        skills = c.get("skills", {})
        resolve = c.get("stats", {}).get("resolve", 0.5)
        labor[cid] = compute_labor(action, skills, resolve)

    gini_before = compute_gini([econ.wealth.get(cid, 0.0) for cid in active_ids])

    surplus = resource_avg * SURPLUS_RATE * len(active_ids)
    distribution = distribute_surplus(econ.model, surplus, labor, active_ids)

    for cid in active_ids:
        current = econ.wealth.get(cid, 0.0)
        hoarding = 0.0
        for c in colonist_data:
            if c["id"] == cid:
                hoarding = c.get("stats", {}).get("hoarding", 0.0)
                break
        gain = distribution.get(cid, 0.0) * (1.0 + hoarding * HOARDING_BONUS)
        decay = current * (WEALTH_DECAY_RATE + rng.gauss(0, 0.005))
        new_wealth = current + gain - decay
        econ.wealth[cid] = max(0.0, min(1.0, new_wealth))

    econ.total_surplus_distributed += surplus

    gini_after = compute_gini([econ.wealth.get(cid, 0.0) for cid in active_ids])
    econ.gini = gini_after

    econ.pressure = compute_economic_pressure(gini_after, resource_avg)

    return EconomicTickResult(
        labor=labor,
        surplus=surplus,
        distribution=distribution,
        gini_before=gini_before,
        gini_after=gini_after,
        pressure=econ.pressure,
        model_used=econ.model,
    )


# -- engine integration helpers ------------------------------------------------

# Alias for backwards compatibility
EconomicsState = EconomicState


def compute_wealth_effects(wealth: float) -> dict[str, float]:
    """Compute action weight modifiers from colonist wealth.

    Wealthy colonists: more likely to hoard/code, less to cooperate/farm.
    Poor colonists: more likely to farm/cooperate, less to hoard.

    Returns dict mapping action -> weight delta.
    """
    w = max(0.0, min(1.0, wealth))
    return {
        "hoard": (w - 0.5) * 0.8,
        "code": (w - 0.3) * 0.3,
        "explore": (w - 0.4) * 0.2,
        "farm": (0.5 - w) * 0.5,
        "cooperate": (0.5 - w) * 0.4,
        "pray": (0.4 - w) * 0.3,
    }


def inequality_trust_erosion(gini: float) -> float:
    """Compute trust erosion from economic inequality.

    Returns a negative delta applied to all pairwise trust values.
    Zero when gini < 0.3 (mild inequality is tolerated).
    """
    if gini < 0.3:
        return 0.0
    return -(gini - 0.3) * 0.05
