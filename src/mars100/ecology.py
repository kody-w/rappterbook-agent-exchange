"""Ecology engine -- slow environmental accumulator for Mars-100.

Tracks terraforming, pollution, atmosphere, biodiversity, and radiation
as 0.0-1.0 bounded values that evolve each year based on colony actions,
infrastructure, and random events. Feeds back into resource modifiers and
death rates.

Tipping points fire exactly once and produce named events.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# --- tipping-point thresholds (fire once each) -----------------------

ATMOSPHERE_BREATHABLE = 0.6
BIODIVERSITY_FLOURISH = 0.5
POLLUTION_TOXIC       = 0.7
RADIATION_LETHAL      = 0.8
TERRAFORM_HABITABLE   = 0.7
TERRAFORM_EDEN        = 0.9

# --- dataclass -------------------------------------------------------

@dataclass
class EcologyState:
    """Complete environmental state of the colony."""
    terraforming: float = 0.0
    pollution: float = 0.0
    atmosphere: float = 0.1   # Mars has a thin atmosphere
    biodiversity: float = 0.0
    radiation: float = 0.5    # Mars baseline

    # tipping-point flags (fire once)
    tp_atmosphere_breathable: bool = False
    tp_biodiversity_flourish: bool = False
    tp_pollution_toxic: bool = False
    tp_radiation_lethal: bool = False
    tp_terraform_habitable: bool = False
    tp_terraform_eden: bool = False

    def to_dict(self) -> dict:
        return {
            "terraforming": round(self.terraforming, 4),
            "pollution": round(self.pollution, 4),
            "atmosphere": round(self.atmosphere, 4),
            "biodiversity": round(self.biodiversity, 4),
            "radiation": round(self.radiation, 4),
            "tipping_points": {
                "atmosphere_breathable": self.tp_atmosphere_breathable,
                "biodiversity_flourish": self.tp_biodiversity_flourish,
                "pollution_toxic": self.tp_pollution_toxic,
                "radiation_lethal": self.tp_radiation_lethal,
                "terraform_habitable": self.tp_terraform_habitable,
                "terraform_eden": self.tp_terraform_eden,
            },
        }


# --- delta computation ------------------------------------------------

def compute_ecology_deltas(
    actions: dict[str, str] | list[dict],
    alive_colonists: list,
    completed_techs: list[str],
    event_context: dict,
) -> dict[str, float]:
    """Compute per-year deltas from colony behaviour.

    actions: dict mapping colonist_id -> action_name (from engine),
             or list of {"action": name} dicts (for standalone use).
    Returns dict of field_name -> delta (may be positive or negative).
    """
    deltas: dict[str, float] = {}

    # Normalize actions to list of action strings
    if isinstance(actions, dict):
        action_names = list(actions.values())
    else:
        action_names = [a.get("action", a) if isinstance(a, dict) else a for a in actions]

    terraform_count = sum(1 for a in action_names if a == "terraform")
    if terraform_count:
        deltas["terraforming"] = terraform_count * 0.02

    hydro_count = sum(1 for a in action_names if a == "hydroponics")
    if hydro_count:
        deltas["biodiversity"] = hydro_count * 0.015

    sabotage_count = sum(1 for a in action_names if a == "sabotage")
    if sabotage_count:
        deltas["pollution"] = deltas.get("pollution", 0.0) + sabotage_count * 0.05

    n_techs = len(completed_techs)
    if n_techs:
        deltas["pollution"] = deltas.get("pollution", 0.0) + n_techs * 0.003

    if "atmospheric_processor" in completed_techs:
        deltas["atmosphere"] = deltas.get("atmosphere", 0.0) + 0.015
        deltas["radiation"] = deltas.get("radiation", 0.0) - 0.01
    if "water_recycler" in completed_techs:
        deltas["biodiversity"] = deltas.get("biodiversity", 0.0) + 0.005

    dust = event_context.get("dust_severity", 0.0)
    if dust > 0:
        deltas["radiation"] = deltas.get("radiation", 0.0) + dust * 0.02
        deltas["atmosphere"] = deltas.get("atmosphere", 0.0) - dust * 0.01

    if terraform_count:
        deltas["atmosphere"] = deltas.get("atmosphere", 0.0) + terraform_count * 0.008

    return deltas


# --- tick -------------------------------------------------------------

def tick_ecology(
    state: EcologyState,
    deltas: dict[str, float],
    rng: random.Random | None = None,
) -> list[dict]:
    """Advance ecology by one year. Mutates *state* in place.

    Returns list of tipping-point events that fired this year.
    """
    events: list[dict] = []

    natural: dict[str, float] = {
        "pollution": -0.005,
        "radiation": -0.002,
    }
    if state.biodiversity > 0.1:
        natural["biodiversity"] = 0.003

    for attr in ("terraforming", "pollution", "atmosphere",
                 "biodiversity", "radiation"):
        d = deltas.get(attr, 0.0) + natural.get(attr, 0.0)
        old = getattr(state, attr)
        new = max(0.0, min(1.0, old + d))
        setattr(state, attr, new)

    tp_checks = [
        ("tp_atmosphere_breathable", "atmosphere", ATMOSPHERE_BREATHABLE,
         "atmosphere_breathable", "The atmosphere is now breathable without suits!"),
        ("tp_biodiversity_flourish", "biodiversity", BIODIVERSITY_FLOURISH,
         "biodiversity_flourish", "Mars ecosystem is self-sustaining."),
        ("tp_pollution_toxic", "pollution", POLLUTION_TOXIC,
         "pollution_toxic", "Pollution has reached toxic levels!"),
        ("tp_radiation_lethal", "radiation", RADIATION_LETHAL,
         "radiation_lethal", "Radiation exposure is now lethal without shielding."),
        ("tp_terraform_habitable", "terraforming", TERRAFORM_HABITABLE,
         "terraform_habitable", "Terraforming milestone: Mars is habitable outdoors."),
        ("tp_terraform_eden", "terraforming", TERRAFORM_EDEN,
         "terraform_eden", "Terraforming complete: Mars is an Eden."),
    ]
    for flag_attr, val_attr, threshold, event_name, description in tp_checks:
        if not getattr(state, flag_attr) and getattr(state, val_attr) >= threshold:
            setattr(state, flag_attr, True)
            events.append({
                "type": "tipping_point",
                "name": event_name,
                "description": description,
                "value": round(getattr(state, val_attr), 4),
            })

    return events


# --- modifiers --------------------------------------------------------

def compute_ecology_modifiers(state: EcologyState) -> dict[str, float]:
    """Compute resource/survival modifiers from current ecology state.

    Returns dict with keys like water_bonus, food_penalty, death_rate_mult,
    event_severity_mult.
    """
    mods: dict[str, float] = {}

    if state.terraforming > 0.3:
        mods["water_bonus"] = (state.terraforming - 0.3) * 0.1
    if state.terraforming > 0.5:
        mods["air_bonus"] = (state.terraforming - 0.5) * 0.08

    if state.biodiversity > 0.2:
        mods["food_bonus"] = (state.biodiversity - 0.2) * 0.12

    if state.pollution > 0.4:
        severity = (state.pollution - 0.4) / 0.6
        mods["food_penalty"] = -severity * 0.06
        mods["water_penalty"] = -severity * 0.03

    death_mult = 1.0
    if state.pollution > 0.5:
        death_mult += (state.pollution - 0.5) * 0.6
    if state.radiation > 0.6:
        death_mult += (state.radiation - 0.6) * 0.8
    if state.terraforming > 0.4:
        death_mult *= max(0.5, 1.0 - (state.terraforming - 0.4) * 0.5)
    mods["death_rate_mult"] = round(death_mult, 4)

    event_mult = 1.0
    event_mult -= state.atmosphere * 0.2
    event_mult += state.pollution * 0.15
    mods["event_severity_mult"] = round(max(0.5, min(1.5, event_mult)), 4)

    return mods


# --- convenience ------------------------------------------------------

def ecology_stress(state: EcologyState) -> float:
    """Single 0-1 stress metric for emergence analysis."""
    return round(max(0.0, min(1.0,
        state.pollution * 0.4
        + state.radiation * 0.3
        - state.terraforming * 0.2
        - state.biodiversity * 0.1
    )), 4)
