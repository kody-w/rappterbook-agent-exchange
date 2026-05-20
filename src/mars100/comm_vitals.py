"""
Colonist comm vitals organ (engine v12.1).

The comm-channels organ (v12.0) scores every *pair*. A parallel topology
organ scores the *graph*. This organ scores every *colonist* — the
node-centric view the action chooser needs when picking who is about to
drown in silence.

For each living colonist we compute:

  - channel_count        : how many channels involve them at all
  - live_channels        : how many are vital/fading/revived
  - flatlined_channels   : how many of their channels are flatlined
  - vital_ratio          : live_channels / max(1, channel_count)
  - mean_vitality        : avg vitality across their channels
  - silence_pressure     : mean silence_streak across their channels
  - isolation_score      : 1 - vital_ratio (higher = lonelier)
  - sole_partners        : peers whose ONLY live channel is this colonist
                            (we are their lifeline; losing us isolates them)
  - is_lifeline          : True iff sole_partners is non-empty
  - urgency              : 0..1, prioritises action_chooser intervention
  - classification       : healthy / strained / isolated / ghosted

Pure functions. No I/O. Deterministic. Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LIVE_STATUSES = frozenset({"vital", "fading", "revived"})


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


def _neighbour_index(active_ids, channels):
    """{cid: [(peer, channel), ...]} restricted to active-active edges."""
    active = set(active_ids)
    out = {nid: [] for nid in active}
    for key, ch in channels.items():
        a, b = key
        if a not in active or b not in active:
            continue
        out[a].append((b, ch))
        out[b].append((a, ch))
    return out


def _classify(vital_ratio, channel_count, flatlined, is_lifeline):
    if channel_count == 0:
        return "ghosted"
    if vital_ratio <= 0.15:
        return "ghosted"
    if vital_ratio < 0.4:
        return "isolated"
    if flatlined >= 3 or (is_lifeline and vital_ratio < 0.75):
        return "strained"
    return "healthy"


def _urgency(vital_ratio, silence_pressure, is_lifeline):
    """0..1 — higher means act sooner."""
    loneliness = max(0.0, min(1.0, 1.0 - vital_ratio))
    silence = max(0.0, min(1.0, silence_pressure / 10.0))
    base = 0.55 * loneliness + 0.30 * silence
    if is_lifeline:
        base += 0.15
    return max(0.0, min(1.0, base))


def compute_colonist_vitals(active_ids, channels, names=None,
                             live_statuses=LIVE_STATUSES):
    """Return list[ColonistCommVital] sorted by (-urgency, colonist_id)."""
    names = names or {}
    idx = _neighbour_index(active_ids, channels)

    # Two passes: first enrich (so we can detect sole_partner symmetrically).
    enriched = {}
    for cid, pairs in idx.items():
        rows = []
        for peer, ch in pairs:
            is_live = ch.status in live_statuses
            rows.append((peer, ch, is_live))
        enriched[cid] = rows

    out = []
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

        # Lifeline detection: a peer is "ours" iff their only live channel
        # leads to us.
        sole_partners = []
        for peer, _, is_live in rows:
            if not is_live:
                continue
            peer_rows = enriched.get(peer, [])
            peer_live = [r for r in peer_rows if r[2]]
            if len(peer_live) == 1 and peer_live[0][0] == cid:
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


def summarise(vitals):
    """Aggregate stats for the report header."""
    if not vitals:
        return {
            "total_colonists": 0, "mean_urgency": 0.0,
            "ghosted": 0, "isolated": 0, "strained": 0, "healthy": 0,
            "lifelines": 0,
        }
    buckets = {"ghosted": 0, "isolated": 0, "strained": 0, "healthy": 0}
    for v in vitals:
        buckets[v.classification] = buckets.get(v.classification, 0) + 1
    return {
        "total_colonists": len(vitals),
        "mean_urgency": round(sum(v.urgency for v in vitals) / len(vitals), 4),
        "ghosted": buckets["ghosted"],
        "isolated": buckets["isolated"],
        "strained": buckets["strained"],
        "healthy": buckets["healthy"],
        "lifelines": sum(1 for v in vitals if v.is_lifeline),
    }
