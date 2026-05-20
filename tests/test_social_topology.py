"""Tests for the social topology organ (engine v12.1)."""
from __future__ import annotations

import pytest

from src.mars100.comm_channels import (
    CommChannelsState, Channel,
    STATUS_VITAL, STATUS_REVIVED, STATUS_FADING,
    STATUS_FLATLINED, STATUS_DORMANT, STATUS_INACTIVE,
    pair_key,
)
from src.mars100.social_topology import (
    TopologyReport, compute_topology, describe_topology,
)


def _mk_channel(a, b, status, last_year=5, year=10):
    return Channel(a=a, b=b, born_year=0, last_contact_year=last_year,
                   total_contacts=5, strong_contacts=2,
                   silence_streak=year - last_year,
                   vitality=0.5, status=status)


def _state_from(*edges):
    st = CommChannelsState()
    for a, b, status in edges:
        st.channels[pair_key(a, b)] = _mk_channel(a, b, status)
    return st


def test_single_cluster_when_all_live():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("b", "c", STATUS_REVIVED),
                     ("a", "c", STATUS_VITAL))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert r.cluster_count == 1
    assert r.components == [["a", "b", "c"]]
    assert r.largest_component_size == 3


def test_split_when_no_live_edges():
    st = _state_from(("a", "b", STATUS_FLATLINED),
                     ("b", "c", STATUS_FLATLINED))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert r.cluster_count == 3
    assert r.fragmentation_index == 1.0


def test_two_clusters():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("c", "d", STATUS_VITAL),
                     ("a", "c", STATUS_FLATLINED))
    r = compute_topology(st, {"a", "b", "c", "d"}, year=10)
    assert r.cluster_count == 2
    sizes = sorted(len(c) for c in r.components)
    assert sizes == [2, 2]


def test_components_partition_active_set():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("c", "d", STATUS_VITAL),
                     ("e", "f", STATUS_FLATLINED))
    active = {"a", "b", "c", "d", "e", "f", "g"}
    r = compute_topology(st, active, year=10)
    flat = [n for comp in r.components for n in comp]
    assert sorted(flat) == sorted(active)
    assert len(set(flat)) == len(flat)
    assert sum(len(c) for c in r.components) == r.n_active


def test_fragmentation_bounds():
    for edges in [
        [("a", "b", STATUS_VITAL)],
        [("a", "b", STATUS_FLATLINED)],
        [("a", "b", STATUS_VITAL), ("b", "c", STATUS_VITAL)],
    ]:
        st = _state_from(*edges)
        active = set()
        for e in edges:
            active.update([e[0], e[1]])
        r = compute_topology(st, active, year=10)
        assert 0.0 <= r.fragmentation_index <= 1.0


def test_fragmentation_zero_for_solo():
    st = CommChannelsState()
    r = compute_topology(st, {"alone"}, year=10)
    assert r.fragmentation_index == 0.0
    assert r.cluster_count == 1


def test_fragmentation_one_for_total_atomization():
    st = _state_from(("a", "b", STATUS_FLATLINED),
                     ("b", "c", STATUS_FLATLINED),
                     ("a", "c", STATUS_FLATLINED))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert r.fragmentation_index == pytest.approx(1.0)


def test_isolation_zero_when_all_live():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("a", "c", STATUS_REVIVED))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert r.isolation_scores["a"] == 0.0


def test_isolation_one_when_all_dead():
    st = _state_from(("a", "b", STATUS_FLATLINED),
                     ("a", "c", STATUS_DORMANT))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert r.isolation_scores["a"] == pytest.approx(1.0)


def test_isolation_ignores_inactive():
    st = _state_from(("a", "b", STATUS_INACTIVE),
                     ("a", "c", STATUS_VITAL))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert r.isolation_scores["a"] == 0.0


def test_isolation_in_bounds():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("a", "c", STATUS_FLATLINED),
                     ("b", "c", STATUS_FADING))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    for v in r.isolation_scores.values():
        assert 0.0 <= v <= 1.0


def test_loner_when_no_live_channels():
    st = _state_from(("a", "b", STATUS_FLATLINED))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert "c" in r.loners and "a" in r.loners and "b" in r.loners


def test_not_loner_with_revived():
    st = _state_from(("a", "b", STATUS_REVIVED))
    r = compute_topology(st, {"a", "b"}, year=10)
    assert r.loners == []


def test_bridge_detection_chain():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("b", "c", STATUS_VITAL))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert sorted(r.bridges) == [("a", "b"), ("b", "c")]


def test_no_bridges_in_triangle():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("b", "c", STATUS_VITAL),
                     ("a", "c", STATUS_VITAL))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert r.bridges == []


def test_bridges_only_live():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("b", "c", STATUS_FLATLINED))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert r.bridges == [("a", "b")]


def test_degree_centrality_bounds():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("a", "c", STATUS_VITAL),
                     ("b", "c", STATUS_VITAL))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    for v in r.degree_centrality.values():
        assert 0.0 <= v <= 1.0
    assert all(v == pytest.approx(1.0) for v in r.degree_centrality.values())


def test_degree_centrality_singleton():
    st = CommChannelsState()
    r = compute_topology(st, {"alone"}, year=10)
    assert r.degree_centrality == {"alone": 0.0}


def test_edge_counts():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("a", "c", STATUS_FLATLINED),
                     ("b", "c", STATUS_REVIVED))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    assert r.n_live_edges == 2
    assert r.n_total_edges == 3


def test_inactive_channels_excluded():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("a", "ghost", STATUS_INACTIVE))
    r = compute_topology(st, {"a", "b"}, year=10)
    assert r.n_total_edges == 1
    assert r.n_live_edges == 1


def test_describe_returns_lines():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("c", "d", STATUS_FLATLINED))
    r = compute_topology(st, {"a", "b", "c", "d"}, year=10)
    lines = describe_topology(r)
    assert isinstance(lines, list) and lines
    assert any("cluster" in line for line in lines)


def test_to_dict_json_safe():
    import json
    st = _state_from(("a", "b", STATUS_VITAL))
    r = compute_topology(st, {"a", "b", "c"}, year=10)
    raw = json.dumps(r.to_dict())
    back = json.loads(raw)
    assert back["n_active"] == 3
    assert "components" in back and "bridges" in back


def test_deterministic():
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("b", "c", STATUS_VITAL),
                     ("c", "d", STATUS_FLATLINED))
    r1 = compute_topology(st, {"a", "b", "c", "d"}, year=10)
    r2 = compute_topology(st, {"a", "b", "c", "d"}, year=10)
    assert r1.to_dict() == r2.to_dict()


def test_handshake_invariant():
    """Sum of degrees == 2 * |edges| (graph theory invariant)."""
    st = _state_from(("a", "b", STATUS_VITAL),
                     ("a", "c", STATUS_VITAL),
                     ("b", "c", STATUS_REVIVED))
    active = {"a", "b", "c"}
    r = compute_topology(st, active, year=10)
    n = r.n_active
    sum_deg = sum(v * (n - 1) for v in r.degree_centrality.values())
    assert sum_deg == pytest.approx(2 * r.n_live_edges)


def test_script_emits_topology(tmp_path):
    """End-to-end: channel_health.py must include topology section."""
    import subprocess, sys, json
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "scripts/channel_health.py",
         "--years", "12", "--seed", "3",
         "--state-dir", str(state_dir), "--no-docs", "--quiet"],
        cwd=str(repo), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    snap = json.loads((state_dir / "channel_health.json").read_text())
    assert snap["_meta"]["version"] == "12.1"
    assert "topology" in snap
    topo = snap["topology"]
    assert "fragmentation_index" in topo
    assert 0.0 <= topo["fragmentation_index"] <= 1.0
    flat = [n for c in topo["components"] for n in c]
    assert len(flat) == topo["n_active"]
    assert len(set(flat)) == len(flat)
