"""
Colonist comm-vitals organ (engine v12.1).

The comm-channels organ (v12.0) scores every *pair* (a channel).
This organ scores every *colonist* — the node-centric view.

For each living/active colonist we compute:

  - channel_count        : total channels involving them
  - live_channels        : count of vital/fading/revived channels
  - flatlined_channels   : count of flatlined channels
  - vital_ratio          : live_channels / max(1, channel_count)
  - mean_vitality        : avg vitality across all their channels
  - silence_pressure     : avg silence_streak (years) across their channels
  - isolation_score      : 1 - vital_ratio (higher = lonelier)
  - sole_partners        : peers whose ONLY live channel is to this colonist
                            (this colonist IS their lifeline)
  - is_lifeline          : True iff sole_partners non-empty
  - urgency              : [0,1], priority signal for action chooser
  - classification       : healthy / strained / isolated / ghosted

Pure functions. No I/O. Deterministic. Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# A peer is reachable iff the channel sits in one of these statuses.
LIVE_STATUSES = frozenset({"vital", "fading", "revived"})

# Classification thresholds — tuned to match the comm_channels organ's
# silence buckets so the two views agree on "this colonist is in trouble".
GHOSTED_RATIO = 0.15
ISOLATED_RATIO = 0.4
STRAINED_FLATLINE_COUNT = 3
STRAINED_LIFELINE_RATIO = 0.75

# Urgency weights — sum is intentionally 0.85 so the lifeline bonus (0.15)
# can lift a strained-lifeline colonist into the high-urgency band.
LONELINESS_WEIGHT = 0.55
SILENCE_WEIGHT = 0.30
LIFELINE_BONUS = 0.15
SILENCE_NORMALISER_YEARS = 10.0


@dataclass
class ColonistCommVital:
    """Per-colonist comm health snapshot."""
    colonist_id: str
    name: str = ""
    channel_count: int = 0
    live_channels: int = 0
    flatlined_channels: int = 0
    vital_ratio: float = 0.0
    mean_vitality: float = 0.0
    silence_pressure: float = 0.0
    isolation_score: float = 0.0
    sole_partners: list = field(default_factory=list)
    is_lifeline: bool = False
    urgency: float = 0.0
    classification: str = "healthy"

    def to_dict(self) -> dict:
        return {
            "colonist_id": self.colonist_id,
            "name": self.name,
            "channel_count": self.channel_count,
            "live_channels": self.live_channels,
            "flatlined_channels": self.flatlined_channels,
            "vital_ratio": round(self.vital_ratio, 4),
            "mean_vitality": round(self.mean_vitality, 4),
            "silence_pressure": round(self.silence_pressure, 4),
            "isolation_score": round(self.isolation_score, 4),
            "sole_partners": list(self.sole_partners),
            "is_lifeline": self.is_lifeline,
            "urgency": round(self.urgency, 4),
            "classification": self.classification,
        }


def _neighbour_index(active_ids: list, channels: dict) -> dict:
    """{colonist_id: list[(peer_id, channel)]} for active colonists.

    A channel is included iff BOTH endpoints are active. The result has
    a key for every active colonist (even with no neighbours), so the
    ghosted/empty case is well-defined.
    """
    active = set(active_ids)
    out: dict = {nid: [] for nid in active}
    for key, ch in channels.items():
        a, b = key
        if a not in active or b not in active:
            continue
        out[a].append((b, ch))
        out[b].append((a, ch))
    return out


def _classify(vital_ratio: float, channel_count: int,
              flatlined: int, is_lifeline: bool) -> str:
    """Bucket label. Priority order: ghosted > isolated > strained > healthy."""
    if channel_count == 0 or vital_ratio <= GHOSTED_RATIO:
        return "ghosted"
    if vital_ratio < ISOLATED_RATIO:
        return "isolated"
    if (flatlined >= STRAINED_FLATLINE_COUNT
            or (is_lifeline and vital_ratio < STRAINED_LIFELINE_RATIO)):
        return "strained"
    return "healthy"


def _urgency(vital_ratio: float, silence_pressure: float,
             is_lifeline: bool) -> float:
    """Composite priority score in [0, 1]."""
    loneliness = max(0.0, min(1.0, 1.0 - vital_ratio))
    silence = max(0.0, min(1.0, silence_pressure / SILENCE_NORMALISER_YEARS))
    base = LONELINESS_WEIGHT * loneliness + SILENCE_WEIGHT * silence
    if is_lifeline:
        base += LIFELINE_BONUS
    return max(0.0, min(1.0, base))


def compute_colonist_vitals(
    active_ids: list,
    channels: dict,
    names: dict | None = None,
    live_statuses: frozenset = LIVE_STATUSES,
) -> list:
    """Return a list of ColonistCommVital, one per active colonist.

    Sorted by urgency descending, colonist_id ascending. `names` is
    optional and used purely for display.
    """
    names = names or {}
    idx = _neighbour_index(active_ids, channels)

    # Two-pass: detect sole-partner relationships symmetrically.
    enriched: dict = {}
    for cid, pairs in idx.items():
        enriched[cid] = [(peer, ch, ch.status in live_statuses)
                          for peer, ch in pairs]

    out: list = []
    for cid in sorted(enriched.keys()):
        rows = enriched[cid]
        channel_count = len(rows)
        live_channels = sum(1 for _, _, live in rows if live)
        flatlined_channels = sum(
            1 for _, ch, _ in rows if ch.status == "flatlined"
        )
        if channel_count > 0:
            vital_ratio = live_channels / channel_count
            mean_vitality = sum(ch.vitality for _, ch, _ in rows) / channel_count
            silence_pressure = (
                sum(ch.silence_streak for _, ch, _ in rows) / channel_count
            )
        else:
            vital_ratio = 0.0
            mean_vitality = 0.0
            silence_pressure = 0.0
        isolation_score = max(0.0, min(1.0, 1.0 - vital_ratio))

        # Sole partner: a live peer whose ONLY live channel is to us.
        sole_partners = []
        for peer, _, is_live in rows:
            if not is_live:
                continue
            peer_live_rows = [r for r in enriched.get(peer, []) if r[2]]
            if len(peer_live_rows) == 1 and peer_live_rows[0][0] == cid:
                sole_partners.append(peer)
        sole_partners.sort()
        is_lifeline = bool(sole_partners)

        urgency = _urgency(vital_ratio, silence_pressure, is_lifeline)
        classification = _classify(
            vital_ratio, channel_count, flatlined_channels, is_lifeline
        )

        out.append(ColonistCommVital(
            colonist_id=cid,
            name=names.get(cid, cid),
            channel_count=channel_count,
            live_channels=live_channels,
            flatlined_channels=flatlined_channels,
            vital_ratio=vital_ratio,
            mean_vitality=mean_vitality,
            silence_pressure=silence_pressure,
            isolation_score=isolation_score,
            sole_partners=sole_partners,
            is_lifeline=is_lifeline,
            urgency=urgency,
            classification=classification,
        ))

    out.sort(key=lambda v: (-v.urgency, v.colonist_id))
    return out


def summarise(vitals: list) -> dict:
    """Aggregate stats for the report header."""
    if not vitals:
        return {
            "total_colonists": 0,
            "mean_urgency": 0.0,
            "max_urgency": 0.0,
            "ghosted": 0, "isolated": 0, "strained": 0, "healthy": 0,
            "lifelines": 0,
        }
    buckets: dict = {"ghosted": 0, "isolated": 0, "strained": 0, "healthy": 0}
    for v in vitals:
        buckets[v.classification] = buckets.get(v.classification, 0) + 1
    return {
        "total_colonists": len(vitals),
        "mean_urgency": round(sum(v.urgency for v in vitals) / len(vitals), 4),
        "max_urgency": round(max(v.urgency for v in vitals), 4),
        "ghosted": buckets["ghosted"],
        "isolated": buckets["isolated"],
        "strained": buckets["strained"],
        "healthy": buckets["healthy"],
        "lifelines": sum(1 for v in vitals if v.is_lifeline),
    }


def revival_prompts(vitals: list, year: int, max_prompts: int = 5) -> list:
    """Generate colonist-level revival prompts for the highest-urgency cases.

    These complement the channel-level revival prompts that the
    comm-channels organ already produces: those nag about a specific
    pair, these nag about a specific colonist drowning across multiple
    channels. Prompts are sorted by urgency desc, then colonist_id asc.
    """
    out: list = []
    for v in vitals[:max_prompts]:
        if v.urgency <= 0.0 or v.classification == "healthy":
            continue
        if v.classification == "ghosted":
            action = "schedule a one-on-one with anyone"
            why = "no live channels remain"
        elif v.classification == "isolated":
            action = "rejoin a group activity"
            why = f"only {v.live_channels}/{v.channel_count} channels live"
        else:  # strained
            if v.is_lifeline:
                action = (f"reinforce ties — {len(v.sole_partners)} other "
                          f"colonist(s) depend on you as their only link")
                why = "lifeline for ghosting peers"
            else:
                action = "reach out to a flatlined contact"
                why = f"{v.flatlined_channels} flatlined channel(s)"
        out.append({
            "kind": "colonist-vitals",
            "year": year,
            "colonist_id": v.colonist_id,
            "name": v.name,
            "classification": v.classification,
            "urgency": round(v.urgency, 4),
            "is_lifeline": v.is_lifeline,
            "sole_partners": list(v.sole_partners),
            "suggested_action": action,
            "text": (f"Year {year}: {v.name or v.colonist_id} is "
                     f"{v.classification} ({why}). {action}."),
        })
    return out
