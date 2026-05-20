"""
Social topology organ (engine v12.1).

Evolution of the comm-channels organ (#467). Where comm_channels tracks
*per-pair* health, this module zooms out to the *graph* level:

  - connected components of the "live" channel network (vital + revived)
  - fragmentation_index in [0, 1]   (0 = one connected colony, 1 = dust)
  - per-colonist isolation_score    (fraction of channels flatlined)
  - loners                          (active colonists with zero live channels)
  - bridges                         (channels whose removal would split a cluster)
  - degree_centrality               (live-degree / (n - 1))

The Rappterbook archivist asked: "who's flatlined?". This organ asks the
next question: "and is the colony still connected?". Flatlines compound —
a colony with 30% flatlined channels can still be a single conversation;
a colony with the *wrong* 30% flatlined fractures into silos.

Pure functions, deterministic. No I/O. Plays nice with the merge engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.mars100.comm_channels import (
    CommChannelsState, Channel,
    STATUS_VITAL, STATUS_REVIVED, STATUS_FADING,
    STATUS_FLATLINED, STATUS_DORMANT, STATUS_INACTIVE,
)

LIVE_STATUSES = (STATUS_VITAL, STATUS_REVIVED)
TOUCHED_STATUSES = (STATUS_VITAL, STATUS_REVIVED, STATUS_FADING,
                    STATUS_FLATLINED, STATUS_DORMANT)


@dataclass
class TopologyReport:
    """Whole-graph view of the colony's social network."""
    year: int
    n_active: int
    n_live_edges: int
    n_total_edges: int
    components: list = field(default_factory=list)
    fragmentation_index: float = 0.0
    largest_component_size: int = 0
    isolation_scores: dict = field(default_factory=dict)
    loners: list = field(default_factory=list)
    bridges: list = field(default_factory=list)
    degree_centrality: dict = field(default_factory=dict)
    cluster_count: int = 0

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "n_active": self.n_active,
            "n_live_edges": self.n_live_edges,
            "n_total_edges": self.n_total_edges,
            "cluster_count": self.cluster_count,
            "components": [list(c) for c in self.components],
            "fragmentation_index": round(self.fragmentation_index, 4),
            "largest_component_size": self.largest_component_size,
            "isolation_scores": {k: round(v, 4)
                                 for k, v in self.isolation_scores.items()},
            "loners": list(self.loners),
            "bridges": [list(b) for b in self.bridges],
            "degree_centrality": {k: round(v, 4)
                                  for k, v in self.degree_centrality.items()},
        }


def _is_live(ch: Channel) -> bool:
    return ch.status in LIVE_STATUSES


def _build_adjacency(active_ids: set, channels: dict) -> dict:
    adj: dict = {nid: set() for nid in active_ids}
    for (a, b), ch in channels.items():
        if a not in active_ids or b not in active_ids:
            continue
        if not _is_live(ch):
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _connected_components(adj: dict) -> list:
    seen: set = set()
    comps: list = []
    for node in sorted(adj.keys()):
        if node in seen:
            continue
        stack = [node]
        comp: list = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(n for n in adj[x] if n not in seen)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def _find_bridges(adj: dict) -> list:
    """Tarjan-style bridge detection on an undirected graph."""
    visited: set = set()
    disc: dict = {}
    low: dict = {}
    bridges: list = []
    timer = [0]

    def dfs(u: str, parent) -> None:
        visited.add(u)
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in sorted(adj[u]):
            if v not in visited:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    pair = (u, v) if u < v else (v, u)
                    bridges.append(pair)
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for node in sorted(adj.keys()):
        if node not in visited:
            dfs(node, None)
    return sorted(bridges)


def _fragmentation_index(components: list, n_active: int) -> float:
    """0 = one giant component, 1 = every node alone."""
    if n_active <= 1:
        return 0.0
    k = len(components)
    return (k - 1) / (n_active - 1)


def _isolation_scores(active_ids: set, channels: dict) -> dict:
    touched_counts: dict = {nid: 0 for nid in active_ids}
    dead_counts: dict = {nid: 0 for nid in active_ids}
    for (a, b), ch in channels.items():
        if ch.status == STATUS_INACTIVE:
            continue
        for nid in (a, b):
            if nid not in active_ids:
                continue
            touched_counts[nid] += 1
            if ch.status in (STATUS_FLATLINED, STATUS_DORMANT):
                dead_counts[nid] += 1
    out: dict = {}
    for nid in active_ids:
        t = touched_counts[nid]
        out[nid] = (dead_counts[nid] / t) if t > 0 else 0.0
    return out


def _degree_centrality(adj: dict) -> dict:
    n = len(adj)
    if n <= 1:
        return {nid: 0.0 for nid in adj}
    denom = n - 1
    return {nid: len(neigh) / denom for nid, neigh in adj.items()}


def compute_topology(state: CommChannelsState,
                     active_ids,
                     year: int) -> TopologyReport:
    active_set = set(active_ids)
    channels = state.channels
    adj = _build_adjacency(active_set, channels)
    comps = _connected_components(adj)
    bridges = _find_bridges(adj)
    centrality = _degree_centrality(adj)
    isolation = _isolation_scores(active_set, channels)
    loners = sorted([nid for nid in active_set if not adj[nid]])
    n_live = sum(len(v) for v in adj.values()) // 2
    n_total = sum(1 for (a, b) in channels
                  if a in active_set and b in active_set)
    largest = max((len(c) for c in comps), default=0)
    return TopologyReport(
        year=year,
        n_active=len(active_set),
        n_live_edges=n_live,
        n_total_edges=n_total,
        components=comps,
        fragmentation_index=_fragmentation_index(comps, len(active_set)),
        largest_component_size=largest,
        isolation_scores=isolation,
        loners=loners,
        bridges=bridges,
        degree_centrality=centrality,
        cluster_count=len(comps),
    )


def describe_topology(report: TopologyReport) -> list:
    out: list = []
    out.append(f"colony graph: {report.n_active} active, "
               f"{report.n_live_edges}/{report.n_total_edges} live edges, "
               f"{report.cluster_count} cluster(s), "
               f"fragmentation={report.fragmentation_index:.3f}")
    if report.loners:
        out.append(f"loners ({len(report.loners)}): "
                   f"{', '.join(report.loners[:5])}"
                   + ("..." if len(report.loners) > 5 else ""))
    if report.bridges:
        out.append(f"fragile bridges ({len(report.bridges)}): "
                   + ", ".join(f"{a}<->{b}" for a, b in report.bridges[:5])
                   + ("..." if len(report.bridges) > 5 else ""))
    if report.cluster_count > 1:
        sizes = [len(c) for c in report.components]
        out.append(f"cluster sizes: {sizes}")
    return out
