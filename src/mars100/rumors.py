"""
Rumor diffusion organ (engine v13.0).

Beliefs spread through the colony along *vital* comm channels. Flatlined
channels block them. Each year:

  1. A small chance someone originates a rumor.
  2. Each carrier rolls per-neighbour: p_pass = base_pass_rate * channel.vitality.
     Flatlined / inactive channels can't carry signal (information islands).
  3. Rumors mutate with low probability (phrase swap from a small vocab).
  4. Rumors decay: if a rumor gains no new carriers for DECAY_YEARS it dies.

Pure functions + plain dataclasses. No I/O. Deterministic given a seed.

Why this matters: the comm-channels organ (v12) made the network observable.
This organ puts traffic on the wire. Next frame can read divergence between
rumor populations to detect cultural fragmentation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

BASE_ORIGIN_RATE = 0.35
BASE_PASS_RATE = 0.45
MUTATION_RATE = 0.04
DECAY_YEARS = 6
MAX_RUMORS = 64
MAX_HISTORY = 200
MIN_VITALITY_FOR_PASS = 0.05

RUMOR_SEEDS = (
    "Earth supply ship will arrive ahead of schedule",
    "the deep aquifer is drying up faster than reported",
    "{a} saw something move in the storm beyond the perimeter",
    "the reactor's neutron count is creeping",
    "a child was born with a third blood type",
    "{a} hoarded extra rations during the last fast",
    "the south greenhouse is producing something the lab can't classify",
    "Faction leaders met in secret about independence",
    "the archivist found a buried message from the first crew",
    "{a} has been sending unencrypted messages back to Earth",
    "a vein of pure ice was struck under sector 7",
    "an old amendment is being quietly enforced again",
    "the lispy logs show a sub-sim that nobody started",
    "someone's been writing names on the airlock wall",
)

MUTATION_SWAPS = {
    "ahead of": "behind", "creeping": "spiking",
    "drying up": "overflowing", "third": "fourth",
    "south greenhouse": "north greenhouse",
    "secret": "public", "unencrypted": "encrypted",
    "old amendment": "new amendment", "buried message": "buried weapon",
    "vein of pure ice": "vein of pure metal",
    "writing names": "burning names",
    "ration": "fuel", "rations": "fuel",
}


@dataclass
class Rumor:
    id: str
    text: str
    origin_id: str
    origin_year: int
    carriers: set = field(default_factory=set)
    born_year: int = 0
    last_growth_year: int = 0
    mutated_from: str | None = None
    mutation_depth: int = 0
    dead: bool = False
    death_year: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text,
            "origin_id": self.origin_id, "origin_year": self.origin_year,
            "carriers": sorted(self.carriers),
            "carrier_count": len(self.carriers),
            "born_year": self.born_year,
            "last_growth_year": self.last_growth_year,
            "mutated_from": self.mutated_from,
            "mutation_depth": self.mutation_depth,
            "dead": self.dead, "death_year": self.death_year,
        }


@dataclass
class RumorsState:
    rumors: dict = field(default_factory=dict)
    next_id: int = 1
    archive: list = field(default_factory=list)
    transmission_log: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rumors": {rid: r.to_dict() for rid, r in self.rumors.items()},
            "next_id": self.next_id,
            "archive": list(self.archive[-MAX_HISTORY:]),
            "transmission_log": list(self.transmission_log[-50:]),
        }


@dataclass
class RumorsTickResult:
    year: int
    born: list
    mutated: list
    died: list
    transmissions: int
    cross_channel_blocks: int
    active_count: int
    largest_carrier_count: int
    fragmentation: float

    def to_dict(self) -> dict:
        return {
            "year": self.year, "born": list(self.born),
            "mutated": list(self.mutated), "died": list(self.died),
            "transmissions": self.transmissions,
            "cross_channel_blocks": self.cross_channel_blocks,
            "active_count": self.active_count,
            "largest_carrier_count": self.largest_carrier_count,
            "fragmentation": round(self.fragmentation, 4),
        }


def _mutate_text(text: str, rng: random.Random) -> str:
    candidates = [k for k in MUTATION_SWAPS if k in text]
    if candidates:
        target = rng.choice(candidates)
        return text.replace(target, MUTATION_SWAPS[target], 1)
    return text + " (or so they say)"


def _pick_seed(rng: random.Random, active_ids):
    origin = rng.choice(active_ids)
    template = rng.choice(RUMOR_SEEDS)
    other = rng.choice(active_ids)
    text = template.replace("{a}", other)
    return origin, text


def compute_fragmentation(rumors: dict, active_ids: set) -> float:
    """Mean pairwise Jaccard distance over carrier sets.
    0 = consensus, 1 = total fragmentation."""
    if not active_ids:
        return 0.0
    alive = [r for r in rumors.values() if not r.dead and r.carriers]
    if len(alive) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            a = alive[i].carriers & active_ids
            b = alive[j].carriers & active_ids
            union = a | b
            if not union:
                continue
            inter = a & b
            jaccard = len(inter) / len(union)
            total += 1.0 - jaccard
            pairs += 1
    return total / pairs if pairs else 0.0


def tick_rumors(state, year, active_colonist_ids, channel_lookup, rng,
                 origin_rate=BASE_ORIGIN_RATE, pass_rate=BASE_PASS_RATE):
    """Advance the rumors organ by one year."""
    active_set = set(active_colonist_ids)
    born, mutated, died = [], [], []
    transmissions = 0
    blocks = 0

    for rumor in state.rumors.values():
        rumor.carriers = rumor.carriers & active_set

    if active_colonist_ids and rng.random() < origin_rate and len(state.rumors) < MAX_RUMORS:
        origin, text = _pick_seed(rng, active_colonist_ids)
        rid = f"r{state.next_id}"
        state.next_id += 1
        rumor = Rumor(id=rid, text=text, origin_id=origin,
                       origin_year=year, born_year=year,
                       last_growth_year=year)
        rumor.carriers.add(origin)
        state.rumors[rid] = rumor
        born.append(rid)

    for rid, rumor in list(state.rumors.items()):
        if rumor.dead or not rumor.carriers:
            continue
        new_carriers = set()
        for carrier in list(rumor.carriers):
            for other in active_colonist_ids:
                if other == carrier or other in rumor.carriers or other in new_carriers:
                    continue
                key = (carrier, other) if carrier < other else (other, carrier)
                ch = channel_lookup.get(key)
                if ch is None:
                    continue
                vitality, status = ch
                if status in ("flatlined", "inactive"):
                    blocks += 1
                    continue
                if vitality < MIN_VITALITY_FOR_PASS:
                    continue
                if rng.random() < pass_rate * vitality:
                    new_carriers.add(other)
        if new_carriers:
            rumor.carriers |= new_carriers
            rumor.last_growth_year = year
            transmissions += len(new_carriers)

    for rid, rumor in list(state.rumors.items()):
        if rumor.dead or len(rumor.carriers) < 2:
            continue
        if len(state.rumors) >= MAX_RUMORS:
            break
        if rng.random() < MUTATION_RATE:
            new_text = _mutate_text(rumor.text, rng)
            if new_text == rumor.text:
                continue
            child_id = f"r{state.next_id}"
            state.next_id += 1
            seed_carrier = rng.choice(sorted(rumor.carriers))
            child = Rumor(id=child_id, text=new_text,
                           origin_id=seed_carrier, origin_year=year,
                           born_year=year, last_growth_year=year,
                           mutated_from=rid,
                           mutation_depth=rumor.mutation_depth + 1)
            child.carriers.add(seed_carrier)
            state.rumors[child_id] = child
            mutated.append({"parent": rid, "child": child_id,
                             "new_text": new_text})

    for rid, rumor in list(state.rumors.items()):
        if rumor.dead:
            continue
        if not rumor.carriers:
            rumor.dead = True
            rumor.death_year = year
            died.append(rid)
            state.archive.append(rumor.to_dict())
            del state.rumors[rid]
            continue
        if year - rumor.last_growth_year >= DECAY_YEARS:
            rumor.dead = True
            rumor.death_year = year
            died.append(rid)
            state.archive.append(rumor.to_dict())
            del state.rumors[rid]

    active_rumors = [r for r in state.rumors.values() if not r.dead]
    largest = max((len(r.carriers) for r in active_rumors), default=0)
    fragmentation = compute_fragmentation(state.rumors, active_set)

    if born or mutated or died:
        state.transmission_log.append({
            "year": year, "born": list(born),
            "mutated": [m["child"] for m in mutated],
            "died": list(died),
            "transmissions": transmissions, "blocks": blocks,
        })

    return RumorsTickResult(
        year=year, born=born, mutated=mutated, died=died,
        transmissions=transmissions, cross_channel_blocks=blocks,
        active_count=len(active_rumors),
        largest_carrier_count=largest,
        fragmentation=fragmentation,
    )


def build_channel_lookup(comm_state) -> dict:
    """Turn a CommChannelsState into {pair_key: (vitality, status)}."""
    return {key: (ch.vitality, ch.status)
             for key, ch in comm_state.channels.items()}
