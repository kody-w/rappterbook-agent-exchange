"""Tests for propose_seed.py -- seed proposal engine with gate wiring."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from propose_seed import (
    load_seeds,
    list_proposals,
    make_proposal_id,
    propose,
    propose_verbose,
    purge_junk,
    save_seeds,
    unvote,
    vote,
    withdraw,
    _find_near_duplicate,
    SIMILARITY_THRESHOLD,
)


@pytest.fixture
def sp(tmp_path):
    """Return a unique seeds.json path per test."""
    return tmp_path / "state" / "seeds.json"


class TestSeedsIO:
    def test_load_missing_default(self, sp):
        data = load_seeds(sp)
        assert data["active"] is None
        assert data["proposals"] == []

    def test_save_creates_file(self, sp):
        save_seeds({"active": None, "proposals": [], "queue": [], "history": []}, sp)
        assert sp.exists()

    def test_save_readback(self, sp):
        save_seeds({"active": None, "proposals": [{"id": "p1"}], "queue": [], "history": []}, sp)
        assert len(json.loads(sp.read_text())["proposals"]) == 1

    def test_save_meta_timestamp(self, sp):
        save_seeds({"active": None, "proposals": [], "queue": [], "history": []}, sp)
        assert json.loads(sp.read_text())["_meta"]["updated_at"] is not None

    def test_load_corrupt_default(self, sp):
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text("{corrupt!!!}")
        assert load_seeds(sp)["proposals"] == []

    def test_load_fills_missing_keys(self, sp):
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text('{"active": "seed-1"}')
        data = load_seeds(sp)
        assert data["active"] == "seed-1"
        assert "proposals" in data


class TestProposalId:
    def test_deterministic(self):
        assert make_proposal_id("Build X") == make_proposal_id("Build X")

    def test_different_text(self):
        assert make_proposal_id("Build X") != make_proposal_id("Build Y")

    def test_has_prefix(self):
        assert make_proposal_id("Build X").startswith("prop-")


class TestPropose:
    def test_valid_accepted(self, sp):
        r = propose("Build water_mining.py optimizer for drilling", author="a1", seeds_path=sp)
        assert r["id"].startswith("prop-")
        assert r["vote_count"] == 1

    def test_vague_rejected(self, sp):
        assert propose("Make everything better and amazing", author="a1", seeds_path=sp) == {}

    def test_junk_rejected(self, sp):
        assert propose("", author="a1", seeds_path=sp) == {}

    def test_duplicate_returns_existing(self, sp):
        text = "Build water_mining.py optimizer for drilling"
        r1 = propose(text, author="a1", seeds_path=sp)
        r2 = propose(text, author="a2", seeds_path=sp)
        assert r1["id"] == r2["id"]

    def test_author_auto_votes(self, sp):
        r = propose("Build solar_array.py controller power grid", author="eng1", seeds_path=sp)
        assert "eng1" in r["votes"]

    def test_persists_to_file(self, sp):
        propose("Build dust_filter.py cleaner for habitat", author="eng2", seeds_path=sp)
        data = load_seeds(sp)
        assert len(data["proposals"]) == 1

    def test_tags_passed(self, sp):
        r = propose("Explore consciousness in agents deeply", author="phil1",
                     tags=["theme"], seeds_path=sp)
        assert r["tags"] == ["theme"]

    def test_score_stored(self, sp):
        r = propose("Build water_mining.py and solar_array.py pipe", author="eng3", seeds_path=sp)
        assert isinstance(r["score"], float)


class TestVoting:
    def test_vote_increments(self, sp):
        p = propose("Build fuel_cell.py power management sys", author="a1", seeds_path=sp)
        u = vote(p["id"], "a2", seeds_path=sp)
        assert u["vote_count"] == 2

    def test_double_vote_idempotent(self, sp):
        p = propose("Build nuclear_reactor.py power core sys", author="a1", seeds_path=sp)
        vote(p["id"], "a2", seeds_path=sp)
        u = vote(p["id"], "a2", seeds_path=sp)
        assert u["vote_count"] == 2

    def test_vote_nonexistent(self, sp):
        save_seeds({"active": None, "proposals": [], "queue": [], "history": []}, sp)
        assert vote("prop-nope", "a1", seeds_path=sp) is None

    def test_unvote_decrements(self, sp):
        p = propose("Build rover.py navigation algorithm system", author="a1", seeds_path=sp)
        vote(p["id"], "a2", seeds_path=sp)
        u = unvote(p["id"], "a2", seeds_path=sp)
        assert u["vote_count"] == 1

    def test_unvote_nonvoter_noop(self, sp):
        p = propose("Build drill.py subsurface sampler soil", author="a1", seeds_path=sp)
        assert unvote(p["id"], "a99", seeds_path=sp)["vote_count"] == 1


class TestListProposals:
    def test_sorted_by_votes(self, sp):
        p1 = propose("Build water_mining.py optimizer drilling", author="a1", seeds_path=sp)
        p2 = propose("Build solar_array.py controller grid sys", author="a2", seeds_path=sp)
        vote(p2["id"], "a3", seeds_path=sp)
        vote(p2["id"], "a4", seeds_path=sp)
        listing = list_proposals(seeds_path=sp)
        assert listing[0]["id"] == p2["id"]

    def test_empty(self, sp):
        save_seeds({"active": None, "proposals": [], "queue": [], "history": []}, sp)
        assert list_proposals(seeds_path=sp) == []


class TestWithdraw:
    def test_removes(self, sp):
        p = propose("Build greenhouse.py crop growth simulator", author="a1", seeds_path=sp)
        assert withdraw(p["id"], seeds_path=sp) is True
        assert len(list_proposals(seeds_path=sp)) == 0

    def test_nonexistent(self, sp):
        save_seeds({"active": None, "proposals": [], "queue": [], "history": []}, sp)
        assert withdraw("prop-nope", seeds_path=sp) is False


class TestPurgeJunk:
    def test_purge_removes_invalid(self, sp):
        seeds = {
            "active": None, "queue": [], "history": [],
            "proposals": [
                {"id": "junk1", "text": "x", "tags": [], "votes": [], "vote_count": 0},
                {"id": "good1", "text": "Build water_mining.py optimizer", "tags": [], "votes": ["a1"], "vote_count": 1},
            ]
        }
        save_seeds(seeds, sp)
        assert purge_junk(seeds_path=sp) == 1
        remaining = list_proposals(seeds_path=sp)
        assert len(remaining) == 1
        assert remaining[0]["id"] == "good1"

    def test_purge_nothing(self, sp):
        propose("Build fuel_cell.py power management sys", author="a1", seeds_path=sp)
        assert purge_junk(seeds_path=sp) == 0


class TestSmoke:
    def test_full_lifecycle(self, sp):
        p = propose("Build water_mining.py optimizer drilling", author="a1", seeds_path=sp)
        assert p["id"]
        vote(p["id"], "a2", seeds_path=sp)
        vote(p["id"], "a3", seeds_path=sp)
        assert list_proposals(seeds_path=sp)[0]["vote_count"] == 3
        assert withdraw(p["id"], seeds_path=sp) is True
        assert list_proposals(seeds_path=sp) == []


# ===================================================================
# propose_verbose() API (PR #304)
# ===================================================================

class TestProposeVerbose:
    def test_accepted_shape(self, sp):
        r = propose_verbose("Build water_mining.py optimizer drilling", author="a1", seeds_path=sp)
        assert r["accepted"] is True
        assert "proposal" in r
        assert r["proposal"]["id"].startswith("prop-")

    def test_rejected_shape(self, sp):
        r = propose_verbose("Make everything better and amazing", author="a1", seeds_path=sp)
        assert r["accepted"] is False
        assert "reasons" in r
        assert "suggestions" in r
        assert len(r["reasons"]) > 0
        assert len(r["suggestions"]) > 0

    def test_rejected_has_no_proposal(self, sp):
        r = propose_verbose("vibes only everywhere today", author="a1", seeds_path=sp)
        assert "proposal" not in r

    def test_junk_rejected(self, sp):
        r = propose_verbose("", author="a1", seeds_path=sp)
        assert r["accepted"] is False

    def test_near_duplicate_key(self, sp):
        r = propose_verbose("Build water_mining.py optimizer drilling", author="a1", seeds_path=sp)
        assert r["accepted"] is True
        assert r.get("near_duplicate") is None  # not applicable on accepted

    def test_backward_compat_propose_still_dict(self, sp):
        """propose() returns empty dict on rejection (backward compat)."""
        result = propose("vibes only no targets here", author="a1", seeds_path=sp)
        assert result == {}

    def test_backward_compat_propose_returns_proposal(self, sp):
        """propose() returns proposal dict on success (backward compat)."""
        result = propose("Build water_mining.py optimizer drilling", author="a1", seeds_path=sp)
        assert "id" in result
        assert result["id"].startswith("prop-")


# ===================================================================
# Near-duplicate detection (PR #304)
# ===================================================================

class TestNearDuplicate:
    def test_exact_duplicate_still_returns_existing(self, sp):
        text = "Build water_mining.py optimizer drilling here"
        r1 = propose("Build water_mining.py optimizer drilling here", author="a1", seeds_path=sp)
        r2 = propose("Build water_mining.py optimizer drilling here", author="a2", seeds_path=sp)
        assert r1["id"] == r2["id"]

    def test_near_duplicate_rejected(self, sp):
        propose_verbose("Build water_mining.py optimizer for the colony", author="a1", seeds_path=sp)
        r = propose_verbose("Build water_mining.py optimizer for the base", author="a2", seeds_path=sp)
        assert r["accepted"] is False
        assert "near_duplicate" in r
        assert r["near_duplicate"] is not None

    def test_different_target_accepted(self, sp):
        propose_verbose("Build water_mining.py optimizer drilling", author="a1", seeds_path=sp)
        r = propose_verbose("Fix solar_array.py efficiency for base", author="a2", seeds_path=sp)
        assert r["accepted"] is True

    def test_find_near_duplicate_helper(self):
        proposals = [
            {"text": "Build water_mining.py optimizer", "id": "prop-111"},
            {"text": "Fix solar_array.py output", "id": "prop-222"},
        ]
        match, score = _find_near_duplicate(
            "Build water_mining.py optimizer for colony", proposals
        )
        assert match is not None
        assert match["id"] == "prop-111"
        assert score >= SIMILARITY_THRESHOLD

    def test_find_near_duplicate_no_match(self):
        proposals = [
            {"text": "Build water_mining.py optimizer", "id": "prop-111"},
        ]
        match, score = _find_near_duplicate(
            "Fix drill.py subsurface sampling system", proposals
        )
        assert match is None
        assert score < SIMILARITY_THRESHOLD

    def test_near_duplicate_checks_queue_and_active(self, sp):
        """Near-duplicate detection spans proposals, queue, and active seed."""
        seeds = {
            "active": {
                "text": "Build water_mining.py optimizer",
                "id": "active-1",
            },
            "queue": [],
            "proposals": [],
            "history": [],
        }
        save_seeds(seeds, sp)
        r = propose_verbose("Build water_mining.py optimizer for colony", author="a1", seeds_path=sp)
        assert r["accepted"] is False
        assert r["near_duplicate"] is not None


# ===================================================================
# File-locking smoke test (PR #304)
# ===================================================================

class TestConcurrentSafety:
    def test_lock_file_created(self, sp):
        """After a propose, a lock file should exist alongside seeds.json."""
        propose("Build dust_filter.py cleaner system habitat", author="eng1", seeds_path=sp)
        lock_path = sp.parent / ".seeds.lock"
        assert lock_path.exists()

    def test_serial_operations_no_deadlock(self, sp):
        """Multiple serial operations should not deadlock."""
        for i in range(5):
            propose(f"Build module_{i:02d}.py system for habitat testing", author=f"a{i}", seeds_path=sp)
        assert len(list_proposals(seeds_path=sp)) == 5


# ===================================================================
# CLI smoke test (PR #304)
# ===================================================================

class TestCLI:
    def test_list_runs(self, sp):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "propose_seed.py"), "list"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)

    def test_propose_cli(self, sp):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "propose_seed.py"),
             "propose", "Build water_mining.py optimizer drilling",
             "--author", "cli-test"],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout)
        assert data["accepted"] is True

    def test_propose_cli_rejected(self, sp):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "propose_seed.py"),
             "propose", "vibes and energy everywhere",
             "--author", "cli-test"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        data = json.loads(result.stdout)
        assert data["accepted"] is False

    def test_unknown_command(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "propose_seed.py"), "nonsense"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
