"""
Governance Laws organ for Mars-100 (v9.0).

Formal laws emerge from governance transitions. Each governance type
generates a distinct set of laws that modify colonist behavior, resource
consumption, exile thresholds, and psychological costs.

Laws are generated when governance changes but enacted the FOLLOWING year
(delayed activation prevents same-tick cascades). Active laws persist
until the next governance transition replaces them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Law dataclass
# ---------------------------------------------------------------------------

@dataclass
class Law:
    """A formal law enacted by the colony's governance."""
    id: str
    name: str
    category: str  # "resource", "behavior", "exile", "labor", "freedom"
    enacted_year: int
    gov_type: str
    modifiers: dict[str, float] = field(default_factory=dict)
    active: bool = True
    repealed_year: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "enacted_year": self.enacted_year,
            "gov_type": self.gov_type,
            "modifiers": dict(self.modifiers),
            "active": self.active,
            "repealed_year": self.repealed_year,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Law:
        return cls(
            id=d["id"],
            name=d["name"],
            category=d.get("category", "behavior"),
            enacted_year=d.get("enacted_year", 0),
            gov_type=d.get("gov_type", "anarchy"),
            modifiers=dict(d.get("modifiers", {})),
            active=d.get("active", True),
            repealed_year=d.get("repealed_year"),
        )


# ---------------------------------------------------------------------------
# Law generation per governance type
# ---------------------------------------------------------------------------

# Each governance type has a palette of possible laws.
_GOV_LAW_PALETTES: dict[str, list[dict]] = {
    "council": [
        {"name": "Fair Rationing Act", "category": "resource",
         "modifiers": {"consumption": 0.9, "hoard": 0.6}},
        {"name": "Council Oversight Decree", "category": "behavior",
         "modifiers": {"sabotage": 0.4, "cooperate": 1.3}},
        {"name": "Community Labor Code", "category": "labor",
         "modifiers": {"research": 1.1, "terraform": 1.1}},
    ],
    "dictator": [
        {"name": "Martial Discipline", "category": "behavior",
         "modifiers": {"sabotage": 0.2, "rest": 0.7, "pray": 0.8}},
        {"name": "Forced Labor Edict", "category": "labor",
         "modifiers": {"terraform": 1.4, "research": 0.8,
                       "stress_cost": 0.06}},
        {"name": "Strict Rationing", "category": "resource",
         "modifiers": {"consumption": 0.8}},
        {"name": "Exile Threshold Lowered", "category": "exile",
         "modifiers": {"exile_trust": 0.2, "exile_sabotage": 0.35}},
    ],
    "lottery": [
        {"name": "Equal Voice Ordinance", "category": "freedom",
         "modifiers": {"cooperate": 1.15, "purpose_boost": 0.02}},
        {"name": "Rotation Readiness", "category": "behavior",
         "modifiers": {"rest": 1.1, "explore": 1.1}},
    ],
    "consensus": [
        {"name": "Unanimous Consent Required", "category": "behavior",
         "modifiers": {"cooperate": 1.4, "sabotage": 0.3,
                       "stress_cost": 0.03}},
        {"name": "Shared Prosperity Pact", "category": "resource",
         "modifiers": {"consumption": 0.85, "hoard": 0.5}},
        {"name": "Transparency Mandate", "category": "freedom",
         "modifiers": {"purpose_boost": 0.03}},
    ],
    "ai_governor": [
        {"name": "Algorithmic Optimization", "category": "resource",
         "modifiers": {"consumption": 0.88, "research": 1.2}},
        {"name": "Machine Directive Alpha", "category": "behavior",
         "modifiers": {"sabotage": 0.3, "terraform": 1.15,
                       "purpose_cost": 0.02}},
        {"name": "Predictive Exile Protocol", "category": "exile",
         "modifiers": {"exile_trust": 0.12, "exile_sabotage": 0.4}},
    ],
    "anarchy": [],  # No formal laws under anarchy
}


def generate_laws(gov_type: str, year: int,
                  resources: dict[str, float],
                  rng: random.Random) -> list[Law]:
    """Generate laws for a new governance type.

    Returns 1-3 laws depending on governance type and resource pressure.
    Laws from anarchy return an empty list.
    """
    palette = _GOV_LAW_PALETTES.get(gov_type, [])
    if not palette:
        return []
    # Under resource pressure, more laws are enacted
    avg_res = sum(resources.get(k, 0.5) for k in
                  ("food", "water", "power", "air", "medicine")) / 5
    max_laws = 2 if avg_res > 0.4 else 3
    count = min(len(palette), rng.randint(1, max_laws))
    chosen = rng.sample(palette, count)
    laws: list[Law] = []
    for i, template in enumerate(chosen):
        law = Law(
            id=f"law-y{year}-{gov_type}-{i}",
            name=template["name"],
            category=template["category"],
            enacted_year=year,
            gov_type=gov_type,
            modifiers=dict(template["modifiers"]),
        )
        laws.append(law)
    return laws


# ---------------------------------------------------------------------------
# Law effect computation
# ---------------------------------------------------------------------------

def compute_law_action_modifiers(laws: list[Law]) -> dict[str, float]:
    """Compute multiplicative action-weight modifiers from active laws.

    Returns a dict mapping action names to multipliers.
    Actions not mentioned default to 1.0 (no effect).
    """
    mods: dict[str, float] = {}
    action_keys = {"sabotage", "cooperate", "hoard", "rest", "pray",
                   "research", "terraform", "explore"}
    for law in laws:
        if not law.active:
            continue
        for key, val in law.modifiers.items():
            if key in action_keys:
                mods[key] = mods.get(key, 1.0) * val
    return mods


def compute_law_consumption_modifier(laws: list[Law]) -> float:
    """Compute resource consumption modifier from active laws.

    Returns a multiplier in [0.5, 1.0] — lower means stricter rationing.
    """
    mod = 1.0
    for law in laws:
        if not law.active:
            continue
        if "consumption" in law.modifiers:
            mod *= law.modifiers["consumption"]
    return max(0.5, min(1.0, mod))


def compute_law_exile_thresholds(laws: list[Law]) -> tuple[float, float]:
    """Return (trust_threshold, sabotage_threshold) for exile decisions.

    Default without laws: trust < 0.15 AND sabotage > 0.5.
    Laws may tighten or loosen these.
    """
    trust_thresh = 0.15
    sab_thresh = 0.5
    for law in laws:
        if not law.active:
            continue
        if "exile_trust" in law.modifiers:
            trust_thresh = law.modifiers["exile_trust"]
        if "exile_sabotage" in law.modifiers:
            sab_thresh = law.modifiers["exile_sabotage"]
    return trust_thresh, sab_thresh


def compute_law_stress_cost(laws: list[Law]) -> float:
    """Compute cumulative stress increase from active laws.

    Some laws (martial discipline, forced labor) impose stress.
    Returns a value in [0.0, 0.15].
    """
    cost = 0.0
    for law in laws:
        if not law.active:
            continue
        cost += law.modifiers.get("stress_cost", 0.0)
    return min(0.15, cost)


def compute_law_purpose_cost(laws: list[Law]) -> float:
    """Compute net purpose adjustment from active laws.

    Positive = purpose drain (oppressive laws).
    Negative = purpose boost (empowering laws).
    Returns value in [-0.1, 0.1].
    """
    cost = 0.0
    for law in laws:
        if not law.active:
            continue
        cost += law.modifiers.get("purpose_cost", 0.0)
        cost -= law.modifiers.get("purpose_boost", 0.0)
    return max(-0.1, min(0.1, cost))


# ---------------------------------------------------------------------------
# Colonist satisfaction
# ---------------------------------------------------------------------------

def compute_satisfaction(
    resolve: float, empathy: float, improvisation: float,
    paranoia: float, faith: float,
    stress: float, morale: float,
    laws: list[Law], resource_trend: float,
) -> float:
    """Compute individual colonist satisfaction with governance.

    Satisfaction is a [0, 1] score combining:
    - Resource trend (improving resources = higher satisfaction)
    - Psychological state (low stress, high morale)
    - Law compatibility with personality
    - Governance type preference (implicit via law effects)

    Returns satisfaction in [0.0, 1.0].
    """
    # Base: psychological wellbeing
    base = 0.3 + 0.35 * morale - 0.2 * stress

    # Resource trend bonus/penalty
    base += max(-0.15, min(0.15, resource_trend * 2.0))

    # Law compatibility: high-paranoia colonists dislike freedom-restricting laws
    freedom_laws = sum(1 for l in laws if l.active and l.category == "freedom")
    behavior_laws = sum(1 for l in laws if l.active and l.category == "behavior")
    exile_laws = sum(1 for l in laws if l.active and l.category == "exile")

    # Paranoid colonists dislike strict behavior laws
    if behavior_laws > 0:
        base -= paranoia * 0.08 * behavior_laws

    # Empathetic colonists like freedom laws
    if freedom_laws > 0:
        base += empathy * 0.06 * freedom_laws

    # High-resolve colonists dislike exile laws (fear of being targeted)
    if exile_laws > 0:
        base -= (1.0 - resolve) * 0.05 * exile_laws

    # Faith provides resilience to dissatisfaction
    base += faith * 0.05

    # Improvisation helps adapt to any system
    base += improvisation * 0.03

    return max(0.0, min(1.0, base))


def compute_colony_satisfaction(satisfactions: list[float]) -> float:
    """Compute colony-wide satisfaction as a weighted average.

    The bottom quartile is weighted 2x to reflect that unhappy colonists
    have outsized impact on governance stability.
    """
    if not satisfactions:
        return 0.5
    sorted_sats = sorted(satisfactions)
    n = len(sorted_sats)
    bottom_q = max(1, n // 4)
    bottom_avg = sum(sorted_sats[:bottom_q]) / bottom_q
    overall_avg = sum(sorted_sats) / n
    # Weighted: 60% overall, 40% bottom quartile
    return 0.6 * overall_avg + 0.4 * bottom_avg


# ---------------------------------------------------------------------------
# Governance crisis check
# ---------------------------------------------------------------------------

def check_governance_crisis(
    satisfaction_history: list[float],
    current_year: int,
    last_gov_change_year: int,
    last_crisis_year: int,
    rng: random.Random,
) -> bool:
    """Check if low satisfaction triggers a governance crisis.

    Conditions for crisis:
    1. At least 3 years of satisfaction history
    2. 3-year moving average satisfaction below 0.35
    3. At least 3 years since last governance change (grace period)
    4. At least 5 years since last crisis (cooldown)
    5. Probabilistic: crisis_prob = (0.35 - avg_sat) * 3, capped at 0.7

    Returns True if a governance crisis is triggered.
    """
    if len(satisfaction_history) < 3:
        return False
    avg_3yr = sum(satisfaction_history[-3:]) / 3
    if avg_3yr >= 0.35:
        return False
    if current_year - last_gov_change_year < 3:
        return False
    if current_year - last_crisis_year < 5:
        return False
    crisis_prob = min(0.7, (0.35 - avg_3yr) * 3.0)
    return rng.random() < crisis_prob
