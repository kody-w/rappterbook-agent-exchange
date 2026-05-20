"""Tests for docs/mars-100/channel_health.html viewer + workflow wiring.

Mutation v12.1: make the channel_health.json observable, not just a JSON
artifact rotting in state/. The viewer is the organ's surface — without
it the data is invisible to humans skimming the dashboard.
"""
from __future__ import annotations

import html.parser
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VIEW = REPO / "docs" / "mars-100" / "channel_health.html"
SCRIPT = REPO / "scripts" / "channel_health.py"
WORKFLOW = REPO / ".github" / "workflows" / "evolve.yml"


class _StrictParser(html.parser.HTMLParser):
    """Tracks element depth so we can detect unbalanced tags."""

    VOID = {"meta", "link", "br", "img", "input", "hr", "source"}

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.max_depth = 0
        self.errors: list = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)

    def handle_endtag(self, tag):
        if tag not in self.VOID:
            self.depth -= 1
            if self.depth < 0:
                self.errors.append(f"unbalanced </{tag}>")


# ---------------------------------------------------------------- static


def test_viewer_exists():
    assert VIEW.exists(), "channel_health.html viewer missing"
    assert VIEW.stat().st_size > 500


def test_viewer_html_parses():
    parser = _StrictParser()
    parser.feed(VIEW.read_text())
    assert not parser.errors, parser.errors
    assert parser.max_depth >= 3


def test_viewer_fetches_correct_json():
    body = VIEW.read_text()
    assert "fetch('channel_health.json')" in body, \
        "viewer must fetch the sibling channel_health.json"
    assert "<title>" in body and "Channel Health" in body


def test_viewer_renders_required_sections():
    body = VIEW.read_text()
    for anchor in ('id="kpis"', 'id="flatlined"', 'id="fading"',
                    'id="revival"', 'id="channels"'):
        assert anchor in body, f"viewer missing anchor {anchor}"
    for status in ("vital", "fading", "flatlined",
                    "revived", "dormant", "inactive"):
        assert f"tag-{status}" in body, f"viewer missing tag-{status} style"


def test_viewer_links_back_to_dashboard():
    body = VIEW.read_text()
    assert 'href="index.html"' in body
    assert 'href="channel_health.json"' in body


def test_viewer_escapes_user_strings():
    body = VIEW.read_text()
    assert "function esc(" in body
    name_uses = re.findall(r"esc\(d\.name_[ab]", body)
    assert len(name_uses) >= 4, \
        "expected esc() on name_a/name_b in multiple renderers"


# ---------------------------------------------------------------- workflow


def test_workflow_runs_channel_health():
    body = WORKFLOW.read_text()
    assert "scripts/channel_health.py" in body, \
        "evolve.yml must invoke channel_health.py"
    pos_script = body.index("scripts/channel_health.py")
    pos_commit = body.index("Commit results")
    assert pos_script < pos_commit, \
        "channel_health step must precede 'Commit results'"


def test_dashboard_links_to_viewer():
    """The colony dashboard must surface the new viewer or no human
    will ever find it."""
    index = REPO / "docs" / "mars-100" / "index.html"
    body = index.read_text()
    assert "channel_health.html" in body, \
        "index.html must link to channel_health.html"


# ---------------------------------------------------------------- e2e


def test_viewer_matches_script_output_shape(tmp_path):
    """The fields the viewer reads MUST be the fields the script emits.

    Smoke-runs the script for 8 years, then asserts every JS field the
    viewer pulls out of the report is present in real output. This is
    the contract between the renderer and the producer.
    """
    state_dir = tmp_path / "state"
    docs_dir = tmp_path / "docs"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--years", "8", "--seed", "1",
         "--state-dir", str(state_dir),
         "--docs-dir", str(docs_dir),
         "--quiet"],
        cwd=str(REPO), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    report = json.loads((state_dir / "channel_health.json").read_text())

    for k in ("_meta", "summary", "overall_health_score",
              "flatlined_channels", "fading_channels",
              "channels", "revival_prompts"):
        assert k in report, f"report missing {k}"

    for k in ("engine", "version", "year_snapshot",
              "generated", "flatline_threshold_years"):
        assert k in report["_meta"], f"_meta missing {k}"

    # `total` is always present. Other status keys appear only when at
    # least one channel holds that status — the viewer reads them with
    # `||0` defaults, so absence is correct behaviour, not a bug.
    assert "total" in report["summary"]
    for k in ("vital", "fading", "flatlined",
              "revived", "dormant"):
        if k in report["summary"]:
            assert isinstance(report["summary"][k], int)

    if report["channels"]:
        sample = report["channels"][0]
        for k in ("a", "b", "name_a", "name_b", "a_alive", "b_alive",
                  "vitality", "status", "silence_streak",
                  "last_contact_year"):
            assert k in sample, f"channel row missing {k}"

    for p in report["revival_prompts"]:
        assert "text" in p
        assert "suggested_action" in p
