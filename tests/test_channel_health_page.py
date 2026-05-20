"""Tests for the channel_health visualization page + workflow wiring."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HTML = REPO / "docs" / "mars-100" / "channel_health.html"
WORKFLOW = REPO / ".github" / "workflows" / "evolve.yml"
SNAPSHOT = REPO / "docs" / "mars-100" / "channel_health.json"


def test_html_exists():
    assert HTML.exists(), f"missing {HTML}"


def test_html_is_well_formed_enough():
    text = HTML.read_text()
    assert text.lstrip().lower().startswith("<!doctype html>")
    assert "<title>" in text and "</title>" in text
    for tag in ("html", "head", "body", "script", "style"):
        opens = len(re.findall(rf"<{tag}\b", text, re.I))
        closes = len(re.findall(rf"</{tag}>", text, re.I))
        assert opens == closes, f"unbalanced <{tag}>: {opens} open vs {closes} close"


def test_html_fetches_snapshot_json():
    text = HTML.read_text()
    assert 'fetch("channel_health.json")' in text or \
        "fetch('channel_health.json')" in text, \
        "page must fetch its sibling channel_health.json"


def test_html_references_every_status_class():
    text = HTML.read_text()
    for status in ("vital", "fading", "flatlined", "revived",
                   "dormant", "inactive"):
        assert f"pill-{status}" in text, f"missing pill class for {status}"
        assert f"fill-{status}" in text, f"missing bar fill for {status}"


def test_workflow_invokes_channel_health_script():
    text = WORKFLOW.read_text()
    assert "scripts/channel_health.py" in text, \
        "evolve.yml must call scripts/channel_health.py"
    script_idx = text.index("scripts/channel_health.py")
    commit_idx = text.index("Commit results")
    assert script_idx < commit_idx, \
        "channel_health step must run before the commit step"


def test_workflow_commits_state_and_docs():
    text = WORKFLOW.read_text()
    assert "git add state/ docs/" in text, \
        "commit step must include state/ and docs/ so channel_health files land"


def test_snapshot_schema_matches_html_expectations():
    """If a snapshot exists, sanity-check the fields the page reads."""
    if not SNAPSHOT.exists():
        return
    data = json.loads(SNAPSHOT.read_text())
    for k in ("_meta", "summary", "overall_health_score",
              "flatlined_channels", "fading_channels",
              "revival_prompts", "channels"):
        assert k in data, f"snapshot missing {k}"
    for k in ("engine", "version", "year_snapshot",
              "flatline_threshold_years"):
        assert k in data["_meta"], f"_meta missing {k}"
    assert 0.0 <= data["overall_health_score"] <= 1.0
