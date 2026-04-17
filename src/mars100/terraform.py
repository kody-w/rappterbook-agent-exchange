"""
Terraform engine for Mars-100.

Models the physical transformation of Mars over 100 years based on
colonist terraforming actions.  One tick = one Martian year.

Physical metrics: pressure, temperature, water, soil fertility,
radiation.  Each evolves via logistic growth with diminishing returns.
Phase-gated milestones mark qualitative shifts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# --- Physical constants (Mars baseline) ---

INITIAL_PRESSURE_ATM = 0.006
INITIAL_TEMPERATURE_C = -60.0
INITIAL_RADIATION_REL = 0.67
INITIAL_WATER_ACCESS = 0.05
INITIAL_SOIL_FERTILITY = 0.02

TARGET_PRESSURE_ATM = 0.12
TARGET_TEMPERATURE_C = -10.0
TARGET_RADIATION_REL = 0.20
TARGET_WATER_ACCESS = 0.40
TARGET_SOIL_FERTILITY = 0.30

# Per-terraformer resource costs per year
TERRAFORM_POWER_COST = 0.02
TERRAFORM_WATER_COST = 0.01


@dataclass
class TerraformState:
    """Physical state of Mars that evolves through terraforming."""

    pressure_atm: float = INITIAL_PRESSURE_ATM
    temperature_c: float = INITIAL_TEMPERATURE_C
    radiation_rel: float = INITIAL_RADIATION_REL
    water_access: float = INITIAL_WATER_ACCESS
    soil_fertility: float = INITIAL_SOIL_FERTILITY
    cumulative_effort: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "pressure_atm": round(self.pressure_atm, 6),
            "temperature_c": round(self.temperature_c, 2),
            "radiation_rel": round(self.radiation_rel, 4),
            "water_access": round(self.water_access, 4),
            "soil_fertility": round(self.soil_fertility, 4),
            "cumulative_effort": round(self.cumulative_effort, 4),
            "terraforming_score": round(self.terraforming_score(), 4),
            "milestone": self.milestone(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TerraformState:
        """Deserialize from dict."""
        return cls(
            pressure_atm=d.get("pressure_atm", INITIAL_PRESSURE_ATM),
            temperature_c=d.get("temperature_c", INITIAL_TEMPERATURE_C),
            radiation_rel=d.get("radiation_rel", INITIAL_RADIATION_REL),
            water_access=d.get("water_access", INITIAL_WATER_ACCESS),
            soil_fertility=d.get("soil_fertility", INITIAL_SOIL_FERTILITY),
            cumulative_effort=d.get("cumulative_effort", 0.0),
        )

    def _normalize(self, current: float, initial: float, target: float) -> float:
        """Normalized 0.0-1.0 progress for one metric."""
        span = target - initial
        if abs(span) < 1e-9:
            return 1.0
        return max(0.0, min(1.0, (current - initial) / span))

    def terraforming_score(self) -> float:
        """Composite 0.0-1.0 progress (mean of five sub-scores)."""
        p = self._normalize(self.pressure_atm, INITIAL_PRESSURE_ATM, TARGET_PRESSURE_ATM)
        t = self._normalize(self.temperature_c, INITIAL_TEMPERATURE_C, TARGET_TEMPERATURE_C)
        r = self._normalize(
            INITIAL_RADIATION_REL - self.radiation_rel,
            0.0,
            INITIAL_RADIATION_REL - TARGET_RADIATION_REL,
        )
        w = self._normalize(self.water_access, INITIAL_WATER_ACCESS, TARGET_WATER_ACCESS)
        s = self._normalize(self.soil_fertility, INITIAL_SOIL_FERTILITY, TARGET_SOIL_FERTILITY)
        return (p + t + r + w + s) / 5.0

    def milestone(self) -> str:
        """Phase-gated milestone with conjunctive thresholds."""
        score = self.terraforming_score()
        if (
            score >= 0.75
            and self.pressure_atm >= 0.08
            and self.temperature_c >= -20.0
            and self.soil_fertility >= 0.15
        ):
            return "ecopoiesis"
        if (
            score >= 0.50
            and self.temperature_c >= -30.0
            and self.water_access >= 0.15
            and self.soil_fertility >= 0.08
        ):
            return "plant_life"
        if (
            score >= 0.25
            and self.water_access >= 0.08
            and self.pressure_atm >= 0.01
        ):
            return "microbes"
        return "barren"


@dataclass
class TerraformDelta:
    """Changes from one year of terraforming."""

    pressure_delta: float = 0.0
    temperature_delta: float = 0.0
    radiation_delta: float = 0.0
    water_delta: float = 0.0
    soil_delta: float = 0.0
    effort_this_year: float = 0.0
    power_cost: float = 0.0
    water_cost: float = 0.0
    milestone_before: str = "barren"
    milestone_after: str = "barren"
    milestone_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "pressure_delta": round(self.pressure_delta, 6),
            "temperature_delta": round(self.temperature_delta, 2),
            "radiation_delta": round(self.radiation_delta, 4),
            "water_delta": round(self.water_delta, 4),
            "soil_delta": round(self.soil_delta, 4),
            "effort_this_year": round(self.effort_this_year, 4),
            "power_cost": round(self.power_cost, 4),
            "water_cost": round(self.water_cost, 4),
            "milestone_before": self.milestone_before,
            "milestone_after": self.milestone_after,
            "milestone_changed": self.milestone_changed,
        }


def _logistic_gain(current: float, target: float, effort: float,
                   rate: float) -> float:
    """Logistic growth toward *target* with diminishing returns near it."""
    if effort <= 0:
        return 0.0
    remaining = max(0.0, target - current)
    return rate * effort * remaining / max(0.01, abs(target))


def tick_terraform(
    state: TerraformState,
    actions: dict[str, str],
    colonist_skills: dict[str, float],
    rng: random.Random,
) -> TerraformDelta:
    """Advance Mars terraforming by one year.

    Args:
        state: Current Mars physical state (mutated in place).
        actions: colonist_id -> chosen action string.
        colonist_skills: colonist_id -> terraforming skill (0-1).
        rng: seeded random source.

    Returns:
        TerraformDelta describing what changed.
    """
    milestone_before = state.milestone()

    terraform_ids = [cid for cid, a in actions.items() if a == "terraform"]
    raw_skill = sum(colonist_skills.get(cid, 0.0) for cid in terraform_ids)
    effort = len(terraform_ids) + raw_skill * 0.5

    # Diminishing returns: each additional unit of effort yields less
    effective_effort = effort / (1.0 + effort * 0.1)
    state.cumulative_effort += effective_effort

    # Pressure: very slow atmospheric thickening
    p_gain = _logistic_gain(
        state.pressure_atm, TARGET_PRESSURE_ATM, effective_effort, 0.0008
    )

    # Temperature: follows pressure (greenhouse effect), gated on min pressure
    t_gain = 0.0
    if state.pressure_atm > INITIAL_PRESSURE_ATM * 1.3:
        t_gain = _logistic_gain(
            state.temperature_c, TARGET_TEMPERATURE_C, effective_effort, 0.4
        )

    # Radiation: passive reduction as atmosphere thickens
    r_change = 0.0
    pressure_above_baseline = state.pressure_atm - INITIAL_PRESSURE_ATM
    if pressure_above_baseline > 0:
        r_change = -pressure_above_baseline * 0.08

    # Water: ice melting as temperature rises + active effort
    w_gain = 0.0
    if state.temperature_c > -50.0:
        w_gain = _logistic_gain(
            state.water_access, TARGET_WATER_ACCESS, effective_effort, 0.003
        )

    # Soil: needs water AND cumulative effort (microbial establishment)
    s_gain = 0.0
    if state.water_access > 0.08 and state.cumulative_effort > 5.0:
        s_gain = _logistic_gain(
            state.soil_fertility, TARGET_SOIL_FERTILITY, effective_effort, 0.002
        )

    # Natural variability
    p_gain += rng.gauss(0, 0.00005)
    t_gain += rng.gauss(0, 0.05)

    # Apply (never below physical minimums)
    state.pressure_atm = max(
        INITIAL_PRESSURE_ATM * 0.95, state.pressure_atm + p_gain
    )
    state.temperature_c = max(
        INITIAL_TEMPERATURE_C - 5.0, state.temperature_c + t_gain
    )
    state.radiation_rel = max(
        TARGET_RADIATION_REL,
        min(INITIAL_RADIATION_REL, state.radiation_rel + r_change),
    )
    state.water_access = max(0.0, min(1.0, state.water_access + w_gain))
    state.soil_fertility = max(0.0, min(1.0, state.soil_fertility + s_gain))

    power_cost = len(terraform_ids) * TERRAFORM_POWER_COST
    water_cost = len(terraform_ids) * TERRAFORM_WATER_COST

    milestone_after = state.milestone()

    return TerraformDelta(
        pressure_delta=p_gain,
        temperature_delta=t_gain,
        radiation_delta=r_change,
        water_delta=w_gain,
        soil_delta=s_gain,
        effort_this_year=effective_effort,
        power_cost=power_cost,
        water_cost=water_cost,
        milestone_before=milestone_before,
        milestone_after=milestone_after,
        milestone_changed=milestone_before != milestone_after,
    )


def event_severity_modifier(state: TerraformState) -> dict[str, float]:
    """Multipliers that modify event severity based on terraforming progress.

    Returned values are multiplied against base severity.
    Values < 1.0 reduce severity; > 1.0 amplify it.
    """
    score = state.terraforming_score()
    return {
        "dust_storm": max(0.3, 1.0 - score * 0.6),
        "solar_flare": max(0.5, 1.0 - score * 0.3),
        "equipment_failure": max(0.7, 1.0 - score * 0.2),
        "epidemic": max(0.6, 1.0 - score * 0.3),
        "resource_strike": min(1.5, 1.0 + score * 0.5),
    }


def resource_production_bonus(state: TerraformState) -> dict[str, float]:
    """Bonus resource production per year from improved conditions."""
    score = state.terraforming_score()
    return {
        "food": score * 0.15 * min(1.0, state.soil_fertility / max(0.01, TARGET_SOIL_FERTILITY)),
        "water": score * 0.10 * min(1.0, state.water_access / max(0.01, TARGET_WATER_ACCESS)),
        "power": 0.0,
        "air": score * 0.05 * min(1.0, state.pressure_atm / max(0.001, TARGET_PRESSURE_ATM)),
        "medicine": score * 0.03,
    }


def subsim_bindings(state: TerraformState) -> dict[str, float]:
    """LisPy bindings for terraforming state in sub-simulations."""
    return {
        "mars-pressure": state.pressure_atm,
        "mars-temperature": state.temperature_c,
        "mars-radiation": state.radiation_rel,
        "mars-water": state.water_access,
        "mars-soil": state.soil_fertility,
        "terraform-score": state.terraforming_score(),
        "terraform-effort": state.cumulative_effort,
    }
