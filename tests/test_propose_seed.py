"""Tests for propose_seed.py -- seed proposal engine with gate wiring."""
from __future__ import annotations

import json
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
    purge_junk,
    save_seeds,
    unvote,
    vote,
    withdraw,
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


# ---------------------------------------------------------------------------
# PR #304 -- similarity() and fuzzy dedup
# ---------------------------------------------------------------------------

from propose_seed import similarity, _normalize_tokens, _SIMILARITY_THRESHOLD


class TestNormalizeTokens:
    def test_basic(self):
        tokens = _normalize_tokens("Build seed_gate.py validator")
        assert "build" in tokens
        assert "seed_gate" in tokens
        assert "validator" in tokens

    def test_strips_extension(self):
        tokens = _normalize_tokens("seed_gate.py")
        assert "seed_gate" in tokens

    def test_strips_path(self):
        tokens = _normalize_tokens("src/seed_gate.py")
        assert "seed_gate" in tokens

    def test_removes_stop_words(self):
        tokens = _normalize_tokens("Build the seed_gate.py for the system")
        assert "the" not in tokens
        assert "for" not in tokens

    def test_empty(self):
        assert _normalize_tokens("") == set()

    def test_lowercase(self):
        tokens = _normalize_tokens("BUILD SEED_GATE.PY")
        assert "build" in tokens
        assert "seed_gate" in tokens

    def test_punctuation_stripped(self):
        tokens = _normalize_tokens("build, seed_gate.py!")
        assert "build" in tokens


class TestSimilarity:
    def test_identity(self):
        assert similarity("Build seed_gate.py", "Build seed_gate.py") == 1.0

    def test_disjoint(self):
        assert similarity("Build seed_gate.py validator", "Deploy nuclear_reactor.py system") < 0.3

    def test_partial_overlap(self):
        s = similarity(
            "Build seed_gate.py validator system",
            "Build seed_gate.py testing suite",
        )
        assert 0.2 < s < 0.9

    def test_empty_first(self):
        assert similarity("", "Build seed_gate.py") == 0.0

    def test_empty_second(self):
        assert similarity("Build seed_gate.py", "") == 0.0

    def test_both_empty(self):
        assert similarity("", "") == 0.0

    def test_path_normalization(self):
        """src/seed_gate.py and seed_gate.py should match."""
        s = similarity(
            "Build src/seed_gate.py validator",
            "Build seed_gate.py validator",
        )
        assert s > 0.8

    def test_symmetric(self):
        a = "Build seed_gate.py validator system"
        b = "Build propose_seed.py integration module"
        assert similarity(a, b) == similarity(b, a)

    def test_near_duplicate(self):
        s = similarity(
            "Build water_mining.py optimizer for drilling",
            "Build water_mining.py optimizer for mining",
        )
        assert s >= 0.6

    def test_different_proposals(self):
        s = similarity(
            "Build water_mining.py optimizer for drilling",
            "Deploy nuclear_reactor.py to production servers",
        )
        assert s < 0.3

    def test_range_0_to_1(self):
        for a, b in [
            ("a", "b"), ("Build X.py", "Build X.py"),
            ("hello world", "hello world foo bar baz"),
        ]:
            s = similarity(a, b)
            assert 0.0 <= s <= 1.0


class TestFuzzyDedup:
    def test_near_duplicate_merges(self, sp):
        """Near-identical proposals should merge and auto-vote."""
        r1 = propose("Optimize water_mining.py drilling depth algorithm performance tuning", author="a1", seeds_path=sp)
        r2 = propose("Optimize water_mining.py drilling depth algorithm performance tweaks", author="a2", seeds_path=sp)
        # Should return the same proposal
        assert r1["id"] == r2["id"]
        # a2 should be auto-voted
        assert "a2" in r2["votes"]
        assert r2["vote_count"] == 2

    def test_dissimilar_creates_new(self, sp):
        """Different proposals should NOT merge."""
        r1 = propose("Build water_mining.py optimizer for drilling", author="a1", seeds_path=sp)
        r2 = propose("Deploy nuclear_reactor.py to production servers", author="a2", seeds_path=sp)
        assert r1["id"] != r2["id"]

    def test_fuzzy_dedup_idempotent_vote(self, sp):
        """Same author fuzzy-deduping shouldn't double-vote."""
        r1 = propose("Optimize water_mining.py drilling depth algorithm performance tuning", author="a1", seeds_path=sp)
        r2 = propose("Optimize water_mining.py drilling depth algorithm performance tweaks", author="a1", seeds_path=sp)
        assert r2["vote_count"] == 1  # a1 already voted

    def test_exact_duplicate_still_works(self, sp):
        """Exact duplicate returns existing (hash check before fuzzy)."""
        text = "Build water_mining.py optimizer for drilling"
        r1 = propose(text, author="a1", seeds_path=sp)
        r2 = propose(text, author="a2", seeds_path=sp)
        assert r1["id"] == r2["id"]

    def test_threshold_boundary(self, sp):
        """Proposals right at the boundary should create new entries."""
        # These share 'Build' but have very different targets
        r1 = propose("Build seed_gate.py validator for proposals", author="a1", seeds_path=sp)
        r2 = propose("Build nuclear_reactor.py power core system", author="a2", seeds_path=sp)
        assert r1["id"] != r2["id"]

    def test_fuzzy_dedup_persists(self, sp):
        """Auto-vote from fuzzy dedup should persist to disk."""
        propose("Optimize water_mining.py drilling depth algorithm performance tuning", author="a1", seeds_path=sp)
        propose("Optimize water_mining.py drilling depth algorithm performance tweaks", author="a2", seeds_path=sp)
        data = load_seeds(sp)
        p = data["proposals"][0]
        assert "a2" in p["votes"]


class TestSimilarityThreshold:
    def test_threshold_is_float(self):
        assert isinstance(_SIMILARITY_THRESHOLD, float)

    def test_threshold_in_range(self):
        assert 0.0 < _SIMILARITY_THRESHOLD < 1.0
