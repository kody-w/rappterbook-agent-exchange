"""Smoke tests for docs/mars-100/channels.html — comm channels sociogram viewer."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HTML = REPO / "docs" / "mars-100" / "channels.html"
SCRIPT = REPO / "scripts" / "channel_health.py"
DATA = REPO / "docs" / "mars-100" / "channel_health.json"


def test_html_exists():
    assert HTML.exists(), "channels.html viewer must exist next to channel_health.json"


def test_html_has_required_structure():
    body = HTML.read_text()
    for needle in [
        '<canvas id="cv"',
        'id="tip"',
        'id="flatlined-list"',
        'id="fading-list"',
        'id="prompts"',
        'id="status-bar"',
        'id="kpi-total"',
        'id="kpi-health"',
        'id="kpi-flatlined"',
        "fetch('channel_health.json",
    ]:
        assert needle in body, f"channels.html missing required element: {needle}"


def test_html_renders_all_six_statuses():
    body = HTML.read_text()
    for status in ("vital", "fading", "flatlined", "revived", "dormant", "inactive"):
        assert status in body, f"sociogram legend must reference status: {status}"


def test_index_links_to_channels():
    idx = (REPO / "docs" / "mars-100" / "index.html").read_text()
    assert "channels.html" in idx, "mars-100 index must link to channels.html"


def test_html_no_external_dependencies():
    body = HTML.read_text()
    assert "<script src=" not in body, "channels.html must have no external scripts"
    assert "cdn." not in body.lower(), "channels.html must not pull from CDNs"


def test_script_produces_compatible_payload(tmp_path):
    """Run the script with a short horizon and check the JSON shape the viewer expects."""
    state_dir = tmp_path / "state"
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--years", "10", "--seed", "1",
         "--state-dir", str(state_dir), "--no-docs", "--quiet"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((state_dir / "channel_health.json").read_text())
    for key in ("_meta", "summary", "overall_health_score",
                "flatlined_count", "fading_count",
                "flatlined_channels", "fading_channels",
                "channels", "revival_prompts"):
        assert key in payload, f"viewer expects field: {key}"
    if payload["channels"]:
        ch = payload["channels"][0]
        for key in ("a", "b", "name_a", "name_b", "a_alive", "b_alive",
                    "status", "silence_streak", "last_contact_year",
                    "vitality"):
            assert key in ch, f"channel record missing field viewer needs: {key}"


def test_committed_payload_is_loadable_if_present():
    """If a committed snapshot exists, it must satisfy the viewer's contract."""
    if not DATA.exists():
        return
    payload = json.loads(DATA.read_text())
    assert "channels" in payload and isinstance(payload["channels"], list)
    assert "summary" in payload and isinstance(payload["summary"], dict)
