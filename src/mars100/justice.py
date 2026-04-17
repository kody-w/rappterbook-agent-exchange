"""
Justice engine for Mars-100.

When colonists commit harmful actions (sabotage during resource stress, hoarding
during food scarcity), the colony can hold trials.  Accusations are queued in
year N and resolved in year N+1 among survivors.
Verdicts: acquit, rehabilitate (stat adjustment), or exile.

Trial outcomes affect the social graph and become colonist memories.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.mars100.colonist import Colonist, STAT_NAMES
from src.mars100.colony import RESOURCE_NAMES, Resources, SocialGraph
from src.mars100.subsim import SubSimBudget, SubSimResult, spawn_subsim

# Thresholds for detecting harmful actions
CRISIS_THRESHOLD = 0.25  # resource level that counts as "stressed"
HARM_ACTIONS = {"sabotage", "hoard"}
SABOTAGE_HARM_FLOOR = 0.02  # minimum resource loss to qualify
VERDICTS = ("acquit", "rehabilitate", "exile")


@dataclass
class HarmRecord:
    """Evidence of a harmful action tied to real engine state."""
    year: int
    accused_id: str
    action: str
    resource_deltas: dict[str, float]
    crisis_resources: list[str]
    harm_score: float  # 0-1, how bad the harm was

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year, "accused_id": self.accused_id,
            "action": self.action, "resource_deltas": self.resource_deltas,
            "crisis_resources": self.crisis_resources,
            "harm_score": self.harm_score,
        }


@dataclass
class Accusation:
    """A charge queued for trial in the following year."""
    year_filed: int
    accused_id: str
    accuser_id: str
    harm: HarmRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "year_filed": self.year_filed, "accused_id": self.accused_id,
            "accuser_id": self.accuser_id, "harm": self.harm.to_dict(),
        }


@dataclass
class TrialResult:
    """Complete record of a trial."""
    year: int
    accused_id: str
    accusation: Accusation
    juror_ids: list[str]
    votes_guilty: list[str] = field(default_factory=list)
    votes_innocent: list[str] = field(default_factory=list)
    verdict: str = "acquit"
    subsim_defense: dict | None = None
    subsim_prosecution: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "year": self.year, "accused_id": self.accused_id,
            "accusation": self.accusation.to_dict(),
            "juror_ids": self.juror_ids,
            "votes_guilty": self.votes_guilty,
            "votes_innocent": self.votes_innocent,
            "verdict": self.verdict,
        }
        if self.subsim_defense:
            d["subsim_defense"] = self.subsim_defense
        if self.subsim_prosecution:
            d["subsim_prosecution"] = self.subsim_prosecution
        return d


# ---------------------------------------------------------------------------
# Harm detection
# ---------------------------------------------------------------------------

def detect_harm(
    year: int,
    actions: dict[str, str],
    resource_delta: dict[str, float],
    resources: Resources,
    colonists: list[Colonist],
    rng: random.Random,
) -> list[HarmRecord]:
    """Detect harmful actions from this year's action/resource state.

    A harm is registered when:
    - sabotage: colonist sabotaged AND (resources are in crisis, OR any
      resource had a negative delta exceeding SABOTAGE_HARM_FLOOR).
    - hoard: colonist hoarded AND food is below CRISIS_THRESHOLD.
    """
    crisis = resources.critical()  # below 0.15
    stressed = [n for n in RESOURCE_NAMES if getattr(resources, n) < CRISIS_THRESHOLD]
    negative_resources = [r for r in RESOURCE_NAMES
                          if resource_delta.get(r, 0) < -SABOTAGE_HARM_FLOOR]

    harms: list[HarmRecord] = []
    for cid, action in actions.items():
        if action not in HARM_ACTIONS:
            continue
        colonist = next((c for c in colonists if c.id == cid), None)
        if not colonist or not colonist.is_active():
            continue

        if action == "sabotage":
            triggers = crisis or negative_resources
            if not triggers:
                continue
            sabotage_skill = colonist.skills.sabotage
            base_harm = 0.15 + sabotage_skill * 0.15
            num_affected = len(crisis) + len(negative_resources)
            harm_score = min(1.0, base_harm + num_affected * 0.1)
            if crisis:
                harm_score = min(1.0, harm_score * 1.5)
        elif action == "hoard":
            if "food" not in stressed and "food" not in crisis:
                continue
            food_delta = resource_delta.get("food", 0)
            if food_delta >= 0:
                continue
            harm_score = min(1.0, 0.2 + colonist.stats.hoarding * 0.4)
            if "food" in crisis:
                harm_score = min(1.0, harm_score * 1.5)
        else:
            continue

        harms.append(HarmRecord(
            year=year, accused_id=cid, action=action,
            resource_deltas=dict(resource_delta),
            crisis_resources=list(crisis or stressed or negative_resources),
            harm_score=harm_score,
        ))
    return harms


# ---------------------------------------------------------------------------
# Accusations
# ---------------------------------------------------------------------------

def file_accusations(
    harms: list[HarmRecord],
    colonists: list[Colonist],
    social: SocialGraph,
    rng: random.Random,
) -> list[Accusation]:
    """File accusations from detected harms.

    The most-trusted active colonist (excluding the accused) files the
    charge.  Only one accusation per colonist per year.
    """
    active = [c for c in colonists if c.is_active()]
    active_ids = [c.id for c in active]
    if len(active) < 3:
        return []

    seen: set[str] = set()
    accusations: list[Accusation] = []
    for harm in harms:
        if harm.accused_id in seen:
            continue
        seen.add(harm.accused_id)
        others = [c for c in active if c.id != harm.accused_id]
        if not others:
            continue
        accuser = max(
            others,
            key=lambda c: social.colony_cohesion([c.id] + [
                x for x in active_ids if x != harm.accused_id
            ]),
        )
        accusations.append(Accusation(
            year_filed=harm.year, accused_id=harm.accused_id,
            accuser_id=accuser.id, harm=harm,
        ))
    return accusations


# ---------------------------------------------------------------------------
# Sub-sim evidence for trials
# ---------------------------------------------------------------------------

def _run_justice_subsim(
    role: str,
    colonist: Colonist,
    harm: HarmRecord,
    year: int,
    rng: random.Random,
) -> SubSimResult | None:
    """Run a sub-sim for the prosecution or defense."""
    bindings = colonist.lispy_bindings()
    bindings["harm-score"] = harm.harm_score
    bindings["crisis-level"] = len(harm.crisis_resources) / max(1, len(RESOURCE_NAMES))
    bindings["repeat-offense"] = min(1.0, colonist.subsim_count * 0.1)

    if role == "prosecution":
        expr = "(let ((severity (+ harm-score (* crisis-level 0.5)))) (if (> severity 0.3) (+ severity 0.2) (* severity 0.5)))"
    else:
        expr = "(let ((character (+ empathy resolve))) (if (> character 1.0) (- 1.0 harm-score) (* harm-score 0.3)))"

    budget = SubSimBudget(year=year)
    log: list[SubSimResult] = []
    result = spawn_subsim(
        expression=expr, colonist_id=colonist.id,
        year=year, bindings=bindings,
        depth=1, budget=budget, log=log,
    )
    return result


# ---------------------------------------------------------------------------
# Juror voting
# ---------------------------------------------------------------------------

def _juror_vote(
    juror: Colonist,
    accused: Colonist,
    accuser: Colonist,
    harm: HarmRecord,
    social: SocialGraph,
    prosecution_score: float,
    defense_score: float,
    rng: random.Random,
) -> bool:
    """A juror casts a guilty/innocent vote.  Returns True for guilty."""
    trust_accused = social.get(juror.id, accused.id).trust
    trust_accuser = social.get(juror.id, accuser.id).trust

    # Base guilt score from evidence strength
    guilt = harm.harm_score * 0.6

    # Repeat offenders: prior trial memories increase skepticism
    prior_trials = sum(1 for m in accused.memories if "trial" in m.event.lower())
    guilt += min(0.2, prior_trials * 0.05)

    # Sub-sim evidence influence
    if prosecution_score > defense_score:
        guilt += 0.15
    elif defense_score > prosecution_score:
        guilt -= 0.10

    # Social influence: trust deficit toward accused raises guilt
    guilt += (trust_accuser - trust_accused) * 0.2

    # Personality: empathetic jurors lean toward leniency
    guilt -= juror.stats.empathy * 0.1
    # Paranoid jurors lean toward conviction
    guilt += juror.stats.paranoia * 0.1

    # Add noise
    guilt += rng.gauss(0, 0.1)

    return guilt > 0.45


# ---------------------------------------------------------------------------
# Trial resolution
# ---------------------------------------------------------------------------

def run_trial(
    accusation: Accusation,
    year: int,
    colonists: list[Colonist],
    social: SocialGraph,
    rng: random.Random,
) -> TrialResult:
    """Run a trial for a pending accusation.

    Jurors: all active colonists except accused and accuser.
    Requires >= 2 jurors or auto-acquits.
    """
    accused = next((c for c in colonists if c.id == accusation.accused_id), None)
    accuser = next((c for c in colonists if c.id == accusation.accuser_id), None)
    active = [c for c in colonists if c.is_active()]

    result = TrialResult(
        year=year, accused_id=accusation.accused_id,
        accusation=accusation, juror_ids=[],
    )

    if not accused or not accused.is_active():
        result.verdict = "acquit"
        return result
    if not accuser or not accuser.is_active():
        result.verdict = "acquit"
        return result

    jurors = [c for c in active
              if c.id != accusation.accused_id and c.id != accusation.accuser_id]
    result.juror_ids = [j.id for j in jurors]

    if len(jurors) < 2:
        result.verdict = "acquit"
        return result

    # Run sub-sim evidence
    pros_sim = _run_justice_subsim("prosecution", accuser, accusation.harm, year, rng)
    def_sim = _run_justice_subsim("defense", accused, accusation.harm, year, rng)

    pros_score = 0.0
    def_score = 0.0
    if pros_sim and pros_sim.succeeded and isinstance(pros_sim.result, (int, float)):
        pros_score = float(pros_sim.result)
        result.subsim_prosecution = pros_sim.to_dict()
    if def_sim and def_sim.succeeded and isinstance(def_sim.result, (int, float)):
        def_score = float(def_sim.result)
        result.subsim_defense = def_sim.to_dict()

    for juror in jurors:
        if _juror_vote(juror, accused, accuser, accusation.harm,
                       social, pros_score, def_score, rng):
            result.votes_guilty.append(juror.id)
        else:
            result.votes_innocent.append(juror.id)

    guilty_ratio = len(result.votes_guilty) / len(jurors) if jurors else 0
    if guilty_ratio > 0.6:
        if accusation.harm.harm_score > 0.7:
            result.verdict = "exile"
        else:
            result.verdict = "rehabilitate"
    else:
        result.verdict = "acquit"

    return result


# ---------------------------------------------------------------------------
# Verdict application
# ---------------------------------------------------------------------------

def apply_verdict(
    trial: TrialResult,
    colonists: list[Colonist],
    social: SocialGraph,
    year: int,
    rng: random.Random,
) -> None:
    """Apply trial verdict: memories first, then effects."""
    accused = next((c for c in colonists if c.id == trial.accused_id), None)
    if not accused or not accused.is_active():
        return

    # Record memories for ALL participants before any exile
    verdict_desc = f"Trial verdict: {trial.verdict} for {accused.name}"
    for c in colonists:
        if c.id in trial.juror_ids or c.id == trial.accused_id:
            valence = -0.3 if trial.verdict == "exile" else 0.1
            c.add_memory(year, verdict_desc, valence)

    if trial.verdict == "rehabilitate":
        # Reduce problematic stats, boost empathy
        accused.stats.paranoia = max(0, accused.stats.paranoia - 0.15)
        accused.stats.hoarding = max(0, accused.stats.hoarding - 0.1)
        accused.stats.empathy = min(1, accused.stats.empathy + 0.1)
        accused.add_memory(year, "Underwent rehabilitation after trial", -0.2)

    elif trial.verdict == "exile":
        # Drop trust from all active colonists toward the exiled
        active_ids = [c.id for c in colonists if c.is_active()]
        for cid in active_ids:
            if cid != trial.accused_id:
                rel = social.get(cid, trial.accused_id)
                rel.trust = max(0, rel.trust - 0.3)
        accused.add_memory(year, "Exiled by colony trial", -0.8)
        accused.exile(year)

    elif trial.verdict == "acquit":
        # Boost sympathy: increase trust toward acquitted
        for jid in trial.votes_innocent:
            rel = social.get(jid, trial.accused_id)
            rel.trust = min(1, rel.trust + 0.05)
