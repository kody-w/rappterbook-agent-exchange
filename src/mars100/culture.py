"""
Cultural evolution engine for Mars-100.

Tracks memetic transmission: stories, norms, myths, and prophecies spread
through the social graph, mutate during transmission, and influence colonist
decisions.  Lore variants form lineages — mutations create new objects, never
silently rewrite what existing carriers believe.

Phase boundary: culture ticks BEFORE action selection each year so that
lore accumulated up to year N−1 influences year N decisions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

CATEGORIES = ("myth", "norm", "tradition", "prophecy", "warning")

# Tradition thresholds with hysteresis to prevent flapping
TRADITION_PROMOTE_THRESHOLD = 0.60
TRADITION_RETIRE_THRESHOLD = 0.40

MAX_LORE_PER_YEAR = 3
MAX_TOTAL_LORE = 100
SPREAD_BASE_PROB = 0.25
MUTATION_BASE_PROB = 0.15
FIDELITY_DECAY_PER_YEAR = 0.005


@dataclass
class Lore:
    """A cultural unit that spreads memetically through the colony."""
    id: str
    root_id: str
    parent_id: str | None
    category: str
    content: str
    lispy_effect: str
    origin_year: int
    origin_colonist: str
    carrier_ids: list[str]
    fidelity: float
    virality: float
    mutation_count: int = 0
    extinct_year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id, "root_id": self.root_id, "parent_id": self.parent_id,
            "category": self.category, "content": self.content,
            "lispy_effect": self.lispy_effect,
            "origin_year": self.origin_year, "origin_colonist": self.origin_colonist,
            "carrier_ids": list(self.carrier_ids),
            "fidelity": round(self.fidelity, 4),
            "virality": round(self.virality, 4),
            "mutation_count": self.mutation_count,
        }
        if self.extinct_year is not None:
            d["extinct_year"] = self.extinct_year
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lore:
        return cls(
            id=d["id"], root_id=d["root_id"], parent_id=d.get("parent_id"),
            category=d["category"], content=d["content"],
            lispy_effect=d.get("lispy_effect", "(+ 0 0)"),
            origin_year=d["origin_year"], origin_colonist=d["origin_colonist"],
            carrier_ids=list(d.get("carrier_ids", [])),
            fidelity=d.get("fidelity", 1.0), virality=d.get("virality", 0.5),
            mutation_count=d.get("mutation_count", 0),
            extinct_year=d.get("extinct_year"),
        )


@dataclass
class CulturalState:
    """The colony's memetic landscape."""
    lore: dict[str, Lore] = field(default_factory=dict)
    tradition_ids: list[str] = field(default_factory=list)
    dead_ids: list[str] = field(default_factory=list)
    next_lore_num: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lore": {lid: l.to_dict() for lid, l in self.lore.items()},
            "tradition_ids": list(self.tradition_ids),
            "dead_ids": list(self.dead_ids),
            "next_lore_num": self.next_lore_num,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CulturalState:
        lore = {lid: Lore.from_dict(ld) for lid, ld in d.get("lore", {}).items()}
        return cls(lore=lore, tradition_ids=list(d.get("tradition_ids", [])),
                   dead_ids=list(d.get("dead_ids", [])),
                   next_lore_num=d.get("next_lore_num", 0))

    def active_lore(self) -> list[Lore]:
        """Return living lore sorted by id for determinism."""
        return sorted(
            [l for l in self.lore.values() if l.extinct_year is None],
            key=lambda l: l.id,
        )

    def generate_id(self) -> str:
        lid = f"lore-{self.next_lore_num:04d}"
        self.next_lore_num += 1
        return lid


# ── Lore templates keyed by triggering event ──────────────────────────

_MYTH_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "dust_storm": [
        {"content": "The Red Veil — a storm so fierce it swallowed the sun for 40 sols",
         "lispy_effect": "(* paranoia 0.05)", "category": "myth"},
        {"content": "When the dust speaks, the wise shelter",
         "lispy_effect": "(* resolve 0.03)", "category": "warning"},
    ],
    "resource_strike": [
        {"content": "The Bounty of Meridiani — proof Mars rewards the patient",
         "lispy_effect": "(* faith 0.04)", "category": "myth"},
    ],
    "equipment_failure": [
        {"content": "Check thrice, trust nothing — the Pressure Law",
         "lispy_effect": "(* paranoia 0.03)", "category": "norm"},
    ],
    "earth_contact": [
        {"content": "Earth remembers us — we are not alone",
         "lispy_effect": "(* empathy 0.05)", "category": "tradition"},
    ],
    "alien_signal": [
        {"content": "The Phobos Whisper — something answered, once",
         "lispy_effect": "(* faith 0.06)", "category": "prophecy"},
        {"content": "We are not the first to ask questions on this world",
         "lispy_effect": "(* improvisation 0.03)", "category": "myth"},
    ],
    "solar_flare": [
        {"content": "The Sun's Wrath — survival means going underground",
         "lispy_effect": "(* resolve 0.04)", "category": "warning"},
    ],
    "ice_volcano": [
        {"content": "Mars bleeds water for those who know where to look",
         "lispy_effect": "(* improvisation 0.04)", "category": "myth"},
    ],
    "colonist_conflict": [
        {"content": "The Compact — no colonist raises a hand against another",
         "lispy_effect": "(* empathy 0.06)", "category": "norm"},
        {"content": "Trust is the most critical resource — harder to mine than water",
         "lispy_effect": "(* empathy 0.04)", "category": "norm"},
    ],
    "breakthrough": [
        {"content": "The First Harvest — proof that Mars can feed us",
         "lispy_effect": "(* resolve 0.05)", "category": "tradition"},
    ],
    "epidemic": [
        {"content": "The Red Cough — never let dust enter the water system",
         "lispy_effect": "(* paranoia 0.04)", "category": "warning"},
    ],
    "cave_discovery": [
        {"content": "The Hollow — Mars has rooms we haven't found yet",
         "lispy_effect": "(* improvisation 0.05)", "category": "myth"},
    ],
    "phobos_transit": [
        {"content": "The Double Shadow — when both moons cross, make a wish",
         "lispy_effect": "(* faith 0.03)", "category": "prophecy"},
    ],
}

_GENERIC_LORE: list[dict[str, str]] = [
    {"content": "Share air before food — the Breathing Compact",
     "lispy_effect": "(* empathy 0.03)", "category": "norm"},
    {"content": "Every colonist teaches two skills — the Chain of Hands",
     "lispy_effect": "(* resolve 0.03)", "category": "tradition"},
    {"content": "When the sub-sim agrees, act. When it disagrees, pause.",
     "lispy_effect": "(* improvisation 0.04)", "category": "norm"},
]


def generate_lore_from_event(event_name: str, year: int, colonist_id: str,
                             culture: CulturalState,
                             rng: random.Random) -> Lore | None:
    """Maybe create new lore from a significant event."""
    active_count = len(culture.active_lore())
    if active_count >= MAX_TOTAL_LORE:
        return None

    templates = _MYTH_TEMPLATES.get(event_name, [])
    if not templates:
        return None

    # Only generate lore from ~30% of matching events
    if rng.random() > 0.30:
        return None

    tmpl = rng.choice(templates)
    lid = culture.generate_id()
    lore = Lore(
        id=lid, root_id=lid, parent_id=None,
        category=tmpl["category"], content=tmpl["content"],
        lispy_effect=tmpl["lispy_effect"],
        origin_year=year, origin_colonist=colonist_id,
        carrier_ids=[colonist_id],
        fidelity=max(0.5, min(1.0, 0.9 + rng.gauss(0, 0.05))),
        virality=max(0.1, min(1.0, 0.5 + rng.gauss(0, 0.15))),
    )
    culture.lore[lid] = lore
    return lore


def spread_lore(lore: Lore, social_edges: dict[str, dict],
                active_ids: list[str], rng: random.Random) -> list[str]:
    """Spread lore along high-trust social edges.  Returns newly-infected IDs."""
    new_carriers: list[str] = []
    existing = set(lore.carrier_ids)

    # Process carriers in sorted order for determinism
    for carrier_id in sorted(existing):
        if carrier_id not in social_edges:
            continue
        neighbors = social_edges[carrier_id]
        for target_id in sorted(neighbors.keys()):
            if target_id in existing or target_id not in active_ids:
                continue
            rel = neighbors[target_id]
            trust = rel.get("trust", 0.5) if isinstance(rel, dict) else getattr(rel, "trust", 0.5)
            prob = SPREAD_BASE_PROB * lore.virality * trust
            if rng.random() < prob:
                new_carriers.append(target_id)
                existing.add(target_id)

    lore.carrier_ids = sorted(existing)
    return new_carriers


def mutate_lore(lore: Lore, carrier_id: str, year: int,
                culture: CulturalState, rng: random.Random) -> Lore | None:
    """Maybe create a mutated variant of lore during transmission.

    Mutations create a NEW lore object — the original carriers keep their version.
    The carrier who mutated it becomes the sole carrier of the new variant.
    """
    if len(culture.active_lore()) >= MAX_TOTAL_LORE:
        return None

    mutation_prob = MUTATION_BASE_PROB * (1.0 - lore.fidelity)
    if rng.random() > mutation_prob:
        return None

    # Content drifts
    suffixes = [
        " — or so they say",
        " — though some dispute this",
        " — amplified by time",
        " — reinterpreted for a new generation",
    ]
    new_content = lore.content.split(" — ")[0] + rng.choice(suffixes)

    # Effect drifts slightly
    # Parse multiplier from lispy_effect like "(* stat 0.05)"
    new_effect = lore.lispy_effect
    try:
        parts = lore.lispy_effect.strip("()").split()
        if len(parts) == 3 and parts[0] == "*":
            val = float(parts[2])
            new_val = max(0.01, min(0.15, val + rng.gauss(0, 0.01)))
            new_effect = f"(* {parts[1]} {new_val:.4f})"
    except (ValueError, IndexError):
        pass

    lid = culture.generate_id()
    variant = Lore(
        id=lid, root_id=lore.root_id, parent_id=lore.id,
        category=lore.category, content=new_content,
        lispy_effect=new_effect,
        origin_year=year, origin_colonist=carrier_id,
        carrier_ids=[carrier_id],
        fidelity=max(0.3, lore.fidelity - 0.05),
        virality=max(0.1, min(1.0, lore.virality + rng.gauss(0, 0.05))),
        mutation_count=lore.mutation_count + 1,
    )
    culture.lore[lid] = variant

    # Remove carrier from original — they now hold the variant
    if carrier_id in lore.carrier_ids:
        lore.carrier_ids.remove(carrier_id)

    return variant


def check_traditions(culture: CulturalState, active_count: int) -> list[str]:
    """Promote/retire traditions with hysteresis. Returns newly promoted IDs."""
    newly_promoted: list[str] = []
    if active_count == 0:
        return newly_promoted

    for lore in culture.active_lore():
        adoption_rate = len(lore.carrier_ids) / active_count
        is_tradition = lore.id in culture.tradition_ids

        if not is_tradition and adoption_rate >= TRADITION_PROMOTE_THRESHOLD:
            culture.tradition_ids.append(lore.id)
            newly_promoted.append(lore.id)
        elif is_tradition and adoption_rate < TRADITION_RETIRE_THRESHOLD:
            culture.tradition_ids.remove(lore.id)

    return newly_promoted


def retire_dead_lore(culture: CulturalState, active_ids: list[str],
                     year: int) -> list[str]:
    """Archive lore whose carriers are all dead/exiled.  Returns extinct IDs."""
    active_set = set(active_ids)
    extinct: list[str] = []
    for lore in culture.active_lore():
        living = [c for c in lore.carrier_ids if c in active_set]
        lore.carrier_ids = sorted(living)
        if not living:
            lore.extinct_year = year
            culture.dead_ids.append(lore.id)
            if lore.id in culture.tradition_ids:
                culture.tradition_ids.remove(lore.id)
            extinct.append(lore.id)
    return extinct


def inherit_culture(child_id: str, parent_ids: list[str],
                    culture: CulturalState, rng: random.Random) -> list[str]:
    """Newborn inherits lore from parents. Returns lore IDs inherited."""
    inherited: list[str] = []
    parent_set = set(parent_ids)
    for lore in culture.active_lore():
        # Child inherits lore that at least one parent carries
        parent_carriers = [c for c in lore.carrier_ids if c in parent_set]
        if parent_carriers and rng.random() < 0.7:
            if child_id not in lore.carrier_ids:
                lore.carrier_ids.append(child_id)
                lore.carrier_ids.sort()
                inherited.append(lore.id)
    return inherited


def lore_influence(carrier_ids_lore: list[tuple[str, Lore]]) -> dict[str, float]:
    """Compute stat modifiers from carried lore.

    Returns a dict of stat_name -> additive delta.
    Effects are capped at ±0.15 per stat to prevent runaway accumulation.
    """
    deltas: dict[str, float] = {}
    for _, lore in carrier_ids_lore:
        try:
            parts = lore.lispy_effect.strip("()").split()
            if len(parts) == 3 and parts[0] == "*":
                stat = parts[1]
                modifier = float(parts[2])
                deltas[stat] = deltas.get(stat, 0.0) + modifier
        except (ValueError, IndexError):
            continue

    # Cap each stat delta
    for stat in deltas:
        deltas[stat] = max(-0.15, min(0.15, deltas[stat]))
    return deltas


def tick_culture(
    year: int,
    colonists: list,
    social_edges: dict[str, dict],
    events: list,
    culture: CulturalState,
    rng: random.Random,
) -> dict[str, Any]:
    """Advance cultural evolution by one year.

    Returns a delta dict summarizing what changed.
    """
    active_ids = sorted(c.id for c in colonists if getattr(c, "is_active", lambda: True)())
    delta: dict[str, Any] = {
        "year": year,
        "new_lore": [],
        "spread_events": [],
        "mutations": [],
        "new_traditions": [],
        "extinct_lore": [],
        "influence": {},
    }

    # 1. Generate new lore from this year's events
    generated_this_year = 0
    for ev in events:
        if generated_this_year >= MAX_LORE_PER_YEAR:
            break
        ev_name = ev.name if hasattr(ev, "name") else ev.get("name", "")
        # Pick a random active colonist as the lore originator
        if not active_ids:
            break
        originator = rng.choice(active_ids)
        new_lore = generate_lore_from_event(ev_name, year, originator, culture, rng)
        if new_lore:
            delta["new_lore"].append(new_lore.to_dict())
            generated_this_year += 1

    # 2. Spread existing lore through social graph
    for lore in culture.active_lore():
        new_carriers = spread_lore(lore, social_edges, active_ids, rng)
        if new_carriers:
            delta["spread_events"].append({
                "lore_id": lore.id, "new_carriers": new_carriers,
            })

    # 3. Mutations during spread (only for lore that just spread)
    for spread_ev in delta["spread_events"]:
        lid = spread_ev["lore_id"]
        if lid not in culture.lore:
            continue
        source_lore = culture.lore[lid]
        for new_carrier in spread_ev["new_carriers"]:
            variant = mutate_lore(source_lore, new_carrier, year, culture, rng)
            if variant:
                delta["mutations"].append(variant.to_dict())

    # 4. Fidelity decay
    for lore in culture.active_lore():
        lore.fidelity = max(0.1, lore.fidelity - FIDELITY_DECAY_PER_YEAR)

    # 5. Retire extinct lore
    extinct = retire_dead_lore(culture, active_ids, year)
    delta["extinct_lore"] = extinct

    # 6. Check tradition promotion/retirement
    new_traditions = check_traditions(culture, len(active_ids))
    delta["new_traditions"] = new_traditions

    # 7. Compute cultural influence on each active colonist
    for colonist in colonists:
        if not (getattr(colonist, "is_active", lambda: True)()):
            continue
        carried = [(colonist.id, lore) for lore in culture.active_lore()
                    if colonist.id in lore.carrier_ids]
        if carried:
            influence = lore_influence(carried)
            if influence:
                delta["influence"][colonist.id] = influence

    return delta
