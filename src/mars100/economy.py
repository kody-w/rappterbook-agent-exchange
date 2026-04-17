"""
Mars-100 economy organ — abstract credit system, taxation, trade, Gini.

Credits are abstract labour tokens, NOT claims on physical resources.
This avoids double-counting with the colony resource model.

Flow per year:
  estate redistribution → income → sabotage → cooperation → taxation → decay → Gini → pressure
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- Constants ---------------------------------------------------------------

ACTION_CREDITS: dict[str, float] = {
    "farm": 3.0, "terraform": 4.0, "code": 3.0, "research": 5.0,
    "mediate": 2.0, "cooperate": 2.0, "explore": 3.0, "pray": 1.0,
    "rest": 0.0, "hoard": 2.0, "sabotage": 0.0,
}

SKILL_BONUS_MAP: dict[str, str] = {
    "farm": "hydroponics", "terraform": "terraforming", "code": "coding",
    "research": "coding", "mediate": "mediation", "explore": "terraforming",
}

TAX_RATES: dict[str, float] = {
    "anarchy": 0.0, "lottery": 0.08, "council": 0.12,
    "consensus": 0.15, "direct_democracy": 0.12,
    "ai_governor": 0.20, "dictator": 0.25,
}

SABOTAGE_STEAL_FRAC = 0.08
COOPERATE_SHARE_FRAC = 0.10
COOPERATE_TRUST_THRESHOLD = 0.4
WEALTH_DECAY_RATE = 0.02
MAX_TRADE_HISTORY = 50


# -- State -------------------------------------------------------------------


@dataclass
class EconomyState:
    """Persistent economy state across years."""
    credits: dict[str, float] = field(default_factory=dict)
    colony_fund: float = 0.0
    total_produced: float = 0.0
    total_taxed: float = 0.0
    gini_history: list[float] = field(default_factory=list)
    trade_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Full serialisation for SimulationResult."""
        return {
            "credits": dict(self.credits),
            "colony_fund": round(self.colony_fund, 2),
            "total_produced": round(self.total_produced, 2),
            "total_taxed": round(self.total_taxed, 2),
            "gini": round(self.gini_history[-1], 4) if self.gini_history else 0.0,
            "gini_history": [round(g, 4) for g in self.gini_history],
            "recent_trades": self.trade_history[-10:],
        }

    def summary(self) -> dict:
        """Short summary for YearResult."""
        vals = [v for v in self.credits.values() if v > 0]
        wealthiest = max(self.credits, key=self.credits.get) if self.credits else None
        poorest = min(self.credits, key=self.credits.get) if self.credits else None
        return {
            "gini": round(self.gini_history[-1], 4) if self.gini_history else 0.0,
            "colony_fund": round(self.colony_fund, 2),
            "active_count": len(self.credits),
            "wealthiest": wealthiest,
            "poorest": poorest,
            "mean_credits": round(sum(vals) / len(vals), 2) if vals else 0.0,
        }


# -- Pure functions ----------------------------------------------------------


def compute_income(actions: dict[str, str],
                   skills: dict[str, dict[str, float]]) -> dict[str, float]:
    """Compute raw income for each colonist based on action + skill bonus."""
    income: dict[str, float] = {}
    for cid, action in actions.items():
        base = ACTION_CREDITS.get(action, 0.0)
        skill_name = SKILL_BONUS_MAP.get(action)
        bonus = 0.0
        if skill_name:
            skill_val = skills.get(cid, {}).get(skill_name, 0.0)
            bonus = base * 0.5 * skill_val
        income[cid] = base + bonus
    return income


def apply_income(econ: EconomyState, income: dict[str, float],
                 active_ids: set[str]) -> float:
    """Add income to balances. Returns total produced."""
    produced = 0.0
    for cid, amount in income.items():
        if cid not in active_ids:
            continue
        econ.credits[cid] = econ.credits.get(cid, 0.0) + amount
        produced += amount
    return produced


def apply_taxation(econ: EconomyState, gov_type: str,
                   dictator_id: str | None) -> float:
    """Tax all positive balances. Returns total taxed."""
    rate = TAX_RATES.get(gov_type, 0.0)
    if rate <= 0:
        return 0.0
    total_tax = 0.0
    for cid in list(econ.credits):
        balance = econ.credits[cid]
        if balance <= 0:
            continue
        tax = balance * rate
        econ.credits[cid] -= tax
        total_tax += tax
    # Dictator gets half of tax revenue
    if dictator_id is not None and gov_type == "dictator":
        dictator_share = total_tax * 0.5
        econ.credits[dictator_id] = econ.credits.get(dictator_id, 0.0) + dictator_share
        econ.colony_fund += total_tax - dictator_share
    else:
        econ.colony_fund += total_tax
    econ.total_taxed += total_tax
    return total_tax


def resolve_sabotage(econ: EconomyState, actions: dict[str, str],
                     active_ids: list[str],
                     rng: random.Random) -> list[dict]:
    """Saboteurs steal credits from a random victim."""
    thefts: list[dict] = []
    saboteurs = [cid for cid, a in actions.items() if a == "sabotage"]
    if not saboteurs or len(active_ids) < 2:
        return thefts
    for sab in saboteurs:
        candidates = [x for x in active_ids if x != sab
                       and econ.credits.get(x, 0.0) > 0]
        if not candidates:
            continue
        victim = rng.choice(candidates)
        stolen = econ.credits[victim] * SABOTAGE_STEAL_FRAC
        if stolen < 0.01:
            continue
        econ.credits[victim] -= stolen
        econ.credits[sab] = econ.credits.get(sab, 0.0) + stolen
        thefts.append({"thief": sab, "victim": victim,
                        "amount": round(stolen, 2)})
    return thefts


def resolve_cooperation(econ: EconomyState, actions: dict[str, str],
                         trust_map: dict[str, dict[str, float]],
                         active_ids: list[str],
                         rng: random.Random) -> list[dict]:
    """Cooperators share credits with their most-trusted peer."""
    transfers: list[dict] = []
    cooperators = [cid for cid, a in actions.items() if a == "cooperate"]
    if not cooperators or len(active_ids) < 2:
        return transfers
    for coop in cooperators:
        balance = econ.credits.get(coop, 0.0)
        if balance <= 0:
            continue
        peers = trust_map.get(coop, {})
        trusted = [(pid, t) for pid, t in peers.items()
                    if pid in set(active_ids) and t >= COOPERATE_TRUST_THRESHOLD]
        if not trusted:
            continue
        trusted.sort(key=lambda x: x[1], reverse=True)
        recipient, _ = trusted[0]
        share = balance * COOPERATE_SHARE_FRAC
        if share < 0.01:
            continue
        econ.credits[coop] -= share
        econ.credits[recipient] = econ.credits.get(recipient, 0.0) + share
        transfers.append({"from": coop, "to": recipient,
                           "amount": round(share, 2)})
    return transfers


def apply_wealth_decay(econ: EconomyState) -> None:
    """Multiplicative decay to prevent unbounded growth."""
    for cid in econ.credits:
        econ.credits[cid] *= (1.0 - WEALTH_DECAY_RATE)


def redistribute_estates(econ: EconomyState,
                          active_ids: set[str]) -> float:
    """Move dead colonists' positive wealth to colony fund."""
    redistributed = 0.0
    dead_ids = [cid for cid in list(econ.credits) if cid not in active_ids]
    for cid in dead_ids:
        balance = econ.credits.pop(cid)
        if balance > 0:
            econ.colony_fund += balance
            redistributed += balance
    return redistributed


def compute_gini(values: list[float]) -> float:
    """Gini coefficient from a list of non-negative values."""
    cleaned = [max(0.0, v) for v in values]
    n = len(cleaned)
    if n < 2:
        return 0.0
    total = sum(cleaned)
    if total <= 0:
        return 0.0
    sorted_vals = sorted(cleaned)
    cumulative = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += (2 * (i + 1) - n - 1) * v
    return cumulative / (n * total)


def compute_economic_pressure(econ: EconomyState) -> dict[str, float]:
    """Return action weight modifiers based on economic state."""
    pressure: dict[str, float] = {}
    gini = econ.gini_history[-1] if econ.gini_history else 0.0

    if gini > 0.5:
        strength = min(2.0, (gini - 0.5) * 4.0)
        pressure["cooperate"] = strength
        pressure["mediate"] = strength * 0.5
    if gini > 0.7:
        pressure["sabotage"] = min(1.5, (gini - 0.7) * 5.0)
    if econ.colony_fund > 50.0:
        pressure["rest"] = min(1.0, econ.colony_fund / 200.0)
    return pressure


# -- Orchestrator ------------------------------------------------------------


def tick_economy(econ: EconomyState,
                 actions: dict[str, str],
                 skills: dict[str, dict[str, float]],
                 trust_map: dict[str, dict[str, float]],
                 active_ids: list[str],
                 gov_type: str,
                 dictator_id: str | None,
                 year: int,
                 rng: random.Random) -> dict:
    """Run one year of economic simulation. Returns year summary dict."""
    active_set = set(active_ids)

    # 1. Estate redistribution (remove dead colonists)
    redistributed = redistribute_estates(econ, active_set)

    # 2. Income
    income = compute_income(actions, skills)
    produced = apply_income(econ, income, active_set)
    econ.total_produced += produced

    # 3. Sabotage
    thefts = resolve_sabotage(econ, actions, active_ids, rng)

    # 4. Cooperation
    transfers = resolve_cooperation(econ, actions, trust_map, active_ids, rng)

    # 5. Taxation
    taxed = apply_taxation(econ, gov_type, dictator_id)

    # 6. Decay
    apply_wealth_decay(econ)

    # 7. Gini
    balances = [econ.credits.get(cid, 0.0) for cid in active_ids]
    gini = compute_gini(balances)
    econ.gini_history.append(gini)

    # 8. Trade history (bounded)
    for t in thefts:
        econ.trade_history.append({"year": year, "type": "theft", **t})
    for t in transfers:
        econ.trade_history.append({"year": year, "type": "share", **t})
    if len(econ.trade_history) > MAX_TRADE_HISTORY:
        econ.trade_history = econ.trade_history[-MAX_TRADE_HISTORY:]

    return {
        "year": year,
        "produced": round(produced, 2),
        "taxed": round(taxed, 2),
        "redistributed": round(redistributed, 2),
        "gini": round(gini, 4),
        "colony_fund": round(econ.colony_fund, 2),
        "thefts": len(thefts),
        "shares": len(transfers),
    }
