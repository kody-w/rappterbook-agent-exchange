"""Tests for propose_seed.py -- seed proposal system with gate integration.

Covers: propose (gate pass/reject), vote/unvote, withdraw authorization,
list ordering, promote with tie-breaking, purge (junk-only), archive,
atomic writes, state normalization, edge cases.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from propose_seed import (
    archive_active,
    list_proposals,
    promote_winner,
    propose,
    purge_junk,
    unvote,
    vote,
    withdraw,
    _load_seeds,
    _save_seeds,
    _normalize,
    _make_id,
    _seeds_path,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(autouse=True)
def _use_tmp_state(tmp_path, monkeypatch):
    """Redirect STATE_DIR to a temp directory for every test."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("STATE_DIR", str(state_dir))


# ===================================================================
# 1. Gate integration -- propose passes/rejects
# ===================================================================

class TestProposeGate:
    def test_good_proposal_passes(self):
        result = propose("Build thermal_control.py module", "agent-1")
        assert result != {}
        assert result["id"].startswith("prop-")
        assert result["author"] == "agent-1"

    def test_good_proposal_stored(self):
        propose("Build thermal_control.py module", "agent-1")
        proposals = list_proposals()
        assert len(proposals) == 1
        assert proposals[0]["text"] == "Build thermal_control.py module"

    def test_rejected_no_verb(self):
        result = propose("The thermal control module", "agent-1")
        assert result == {}
        assert list_proposals() == []

    def test_rejected_no_target(self):
        result = propose("Build something really cool and interesting", "agent-1")
        assert result == {}

    def test_theme_tag_exempts_target(self):
        result = propose("Explore the philosophy of terraforming Mars", "agent-1",
                         tags=["theme"])
        assert result != {}
        assert result["tags"] == ["theme"]

    def test_junk_rejected(self):
        result = propose("", "agent-1")
        assert result == {}

    def test_short_junk_rejected(self):
        result = propose("hi", "agent-1")
        assert result == {}

    def test_fragment_rejected(self):
        result = propose("the parser grabbed a substring", "agent-1")
        assert result == {}

    def test_duplicate_returns_existing(self):
        p1 = propose("Build thermal_control.py module", "agent-1")
        p2 = propose("Build thermal_control.py module", "agent-2")
        assert p1["id"] == p2["id"]
        # Should still have 1 proposal, not 2
        assert len(list_proposals()) == 1

    def test_context_stored(self):
        result = propose("Build rover.py navigation", "agent-1",
                         context="Needed for exploration")
        assert result["context"] == "Needed for exploration"


# ===================================================================
# 2. Voting
# ===================================================================

class TestVoting:
    def test_vote_increments(self):
        p = propose("Build thermal_control.py module", "agent-1")
        result = vote(p["id"], "agent-2")
        assert result is not None
        assert result["vote_count"] == 2
        assert "agent-2" in result["votes"]

    def test_double_vote_noop(self):
        p = propose("Build thermal_control.py module", "agent-1")
        vote(p["id"], "agent-2")
        result = vote(p["id"], "agent-2")
        assert result["vote_count"] == 2

    def test_vote_nonexistent_returns_none(self):
        assert vote("prop-nonexistent", "agent-1") is None

    def test_author_auto_votes(self):
        p = propose("Build thermal_control.py module", "agent-1")
        assert p["vote_count"] == 1
        assert "agent-1" in p["votes"]

    def test_unvote(self):
        p = propose("Build thermal_control.py module", "agent-1")
        vote(p["id"], "agent-2")
        result = unvote(p["id"], "agent-2")
        assert result["vote_count"] == 1
        assert "agent-2" not in result["votes"]

    def test_unvote_not_voted_noop(self):
        p = propose("Build thermal_control.py module", "agent-1")
        result = unvote(p["id"], "agent-99")
        assert result["vote_count"] == 1

    def test_unvote_nonexistent_returns_none(self):
        assert unvote("prop-nonexistent", "agent-1") is None


# ===================================================================
# 3. Withdraw authorization
# ===================================================================

class TestWithdraw:
    def test_author_can_withdraw(self):
        p = propose("Build thermal_control.py module", "agent-1")
        assert withdraw(p["id"], "agent-1") is True
        assert list_proposals() == []

    def test_non_author_cannot_withdraw(self):
        p = propose("Build thermal_control.py module", "agent-1")
        assert withdraw(p["id"], "agent-2") is False
        assert len(list_proposals()) == 1

    def test_withdraw_nonexistent(self):
        assert withdraw("prop-nonexistent", "agent-1") is False


# ===================================================================
# 4. List ordering
# ===================================================================

class TestListProposals:
    def test_sorted_by_votes_descending(self):
        p1 = propose("Build thermal_control.py module", "agent-1")
        p2 = propose("Fix rover.py navigation bugs", "agent-2")
        vote(p2["id"], "agent-3")
        vote(p2["id"], "agent-4")
        proposals = list_proposals()
        assert proposals[0]["id"] == p2["id"]
        assert proposals[1]["id"] == p1["id"]

    def test_empty_list(self):
        assert list_proposals() == []


# ===================================================================
# 5. Promote with tie-breaking
# ===================================================================

class TestPromote:
    def test_promote_winner(self):
        propose("Build thermal_control.py module", "agent-1")
        p2 = propose("Fix rover.py navigation bugs", "agent-2")
        vote(p2["id"], "agent-3")
        winner = promote_winner()
        assert winner is not None
        assert winner["id"] == p2["id"]
        assert "promoted_at" in winner

    def test_promote_removes_from_proposals(self):
        propose("Build thermal_control.py module", "agent-1")
        promote_winner()
        assert len(list_proposals()) == 0

    def test_promote_blocks_if_active(self):
        propose("Build thermal_control.py module", "agent-1")
        promote_winner()
        propose("Fix rover.py navigation bugs", "agent-2")
        assert promote_winner() is None

    def test_promote_empty_returns_none(self):
        assert promote_winner() is None

    def test_promote_tie_uses_earliest(self):
        """When vote counts tie, earliest proposed_at wins."""
        p1 = propose("Build thermal_control.py module", "agent-1")
        p2 = propose("Fix rover.py navigation bugs", "agent-2")
        # Both have 1 vote — p1 was proposed first
        winner = promote_winner()
        assert winner["id"] == p1["id"]


# ===================================================================
# 6. Purge (junk-only, not all non-passing)
# ===================================================================

class TestPurge:
    def test_purge_removes_junk(self):
        # Manually inject junk into state
        seeds = _load_seeds()
        seeds["proposals"].append({
            "id": "prop-junk1",
            "text": "",
            "author": "agent-bad",
            "tags": [],
            "votes": ["agent-bad"],
            "vote_count": 1,
        })
        _save_seeds(seeds)
        count = purge_junk()
        assert count == 1
        assert len(list_proposals()) == 0

    def test_purge_preserves_valid(self):
        propose("Build thermal_control.py module", "agent-1")
        count = purge_junk()
        assert count == 0
        assert len(list_proposals()) == 1

    def test_purge_preserves_theme_proposals(self):
        """Theme proposals without a target should NOT be purged."""
        propose("Explore the philosophy of terraforming Mars", "agent-1",
                tags=["theme"])
        count = purge_junk()
        assert count == 0
        assert len(list_proposals()) == 1

    def test_purge_removes_fragment_only(self):
        """Only truly junk proposals get purged."""
        propose("Build thermal_control.py module", "agent-1")
        seeds = _load_seeds()
        seeds["proposals"].append({
            "id": "prop-frag",
            "text": "x",
            "author": "agent-bad",
            "tags": [],
            "votes": ["agent-bad"],
            "vote_count": 1,
        })
        _save_seeds(seeds)
        count = purge_junk()
        assert count == 1
        assert len(list_proposals()) == 1


# ===================================================================
# 7. Archive
# ===================================================================

class TestArchive:
    def test_archive_active(self):
        propose("Build thermal_control.py module", "agent-1")
        promote_winner()
        archived = archive_active("completed")
        assert archived is not None
        assert archived["archive_reason"] == "completed"
        seeds = _load_seeds()
        assert seeds["active"] is None
        assert len(seeds["history"]) == 1

    def test_archive_nothing_active(self):
        assert archive_active() is None

    def test_archive_after_archive_allows_new_promote(self):
        propose("Build thermal_control.py module", "agent-1")
        promote_winner()
        archive_active()
        propose("Fix rover.py navigation bugs", "agent-2")
        winner = promote_winner()
        assert winner is not None


# ===================================================================
# 8. State normalization
# ===================================================================

class TestNormalize:
    def test_normalize_missing_fields(self):
        data = {}
        normalized = _normalize(data)
        assert normalized["active"] is None
        assert normalized["proposals"] == []
        assert normalized["history"] == []

    def test_normalize_proposal_defaults(self):
        data = {"proposals": [{"id": "p1", "text": "Build foo.py", "author": "a1"}]}
        normalized = _normalize(data)
        p = normalized["proposals"][0]
        assert "votes" in p
        assert "vote_count" in p
        assert "tags" in p

    def test_corrupt_json_returns_empty(self, tmp_path, monkeypatch):
        state_dir = tmp_path / "corrupt_state"
        state_dir.mkdir()
        monkeypatch.setenv("STATE_DIR", str(state_dir))
        (state_dir / "seeds.json").write_text("{invalid json")
        seeds = _load_seeds()
        assert seeds["proposals"] == []


# ===================================================================
# 9. Atomic writes
# ===================================================================

class TestAtomicWrites:
    def test_write_creates_state_dir(self, tmp_path, monkeypatch):
        state_dir = tmp_path / "new_state"
        monkeypatch.setenv("STATE_DIR", str(state_dir))
        propose("Build thermal_control.py module", "agent-1")
        assert (state_dir / "seeds.json").exists()

    def test_json_valid_after_write(self):
        propose("Build thermal_control.py module", "agent-1")
        path = _seeds_path()
        with open(path) as f:
            data = json.load(f)
        assert "proposals" in data


# ===================================================================
# 10. Make ID deterministic
# ===================================================================

class TestMakeId:
    def test_deterministic(self):
        assert _make_id("Build foo.py") == _make_id("Build foo.py")

    def test_prefix(self):
        assert _make_id("anything").startswith("prop-")

    def test_different_texts_different_ids(self):
        assert _make_id("Build foo.py") != _make_id("Build bar.py")


# ===================================================================
# 11. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_propose_strips_whitespace(self):
        result = propose("  Build thermal_control.py module  ", "agent-1")
        assert result["text"] == "Build thermal_control.py module"

    def test_multiple_proposals(self):
        propose("Build thermal_control.py module", "a1")
        propose("Fix rover.py navigation bugs", "a2")
        propose("Test solar_array.py performance", "a3")
        assert len(list_proposals()) == 3

    def test_vote_persists_across_loads(self):
        p = propose("Build thermal_control.py module", "agent-1")
        vote(p["id"], "agent-2")
        proposals = list_proposals()
        assert proposals[0]["vote_count"] == 2


# ===================================================================
# 12. Integration: full lifecycle
# ===================================================================

class TestLifecycle:
    def test_full_lifecycle(self):
        """Propose → vote → promote → archive → new cycle."""
        p1 = propose("Build thermal_control.py module", "agent-1")
        p2 = propose("Fix rover.py navigation bugs", "agent-2")
        vote(p1["id"], "agent-3")
        vote(p2["id"], "agent-3")
        vote(p2["id"], "agent-4")

        winner = promote_winner()
        assert winner["id"] == p2["id"]

        archived = archive_active("completed")
        assert archived is not None

        # agent-1's proposal is still there
        remaining = list_proposals()
        assert len(remaining) == 1
        assert remaining[0]["id"] == p1["id"]

        # Can promote again
        winner2 = promote_winner()
        assert winner2["id"] == p1["id"]
