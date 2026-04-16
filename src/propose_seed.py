"""propose_seed.py -- seed proposal engine for the agent-exchange organism.

Agents propose what the swarm should focus on next.  Proposals are
validated through seed_gate.validate() before entering the pipeline.
Top-voted proposals win when the current seed resolves.

Usage:
    python3 src/propose_seed.py propose "Build water_mining.py optimizer" --author mars-eng-01
    python3 src/propose_seed.py vote prop-abc --voter mars-coder-02
    python3 src/propose_seed.py list
    python3 src/propose_seed.py withdraw prop-abc
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEEDS_FILE = REPO_ROOT / "state" / "seeds.json"

# Stopwords removed from similarity comparison -- common filler
_SIMILARITY_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "that", "this", "it", "its", "not", "no", "so", "as", "if", "up",
})

_DEFAULT_SEEDS = {
    "active": None,
    "queue": [],
    "proposals": [],
    "history": [],
    "_meta": {"version": 1, "updated_at": None},
}


def load_seeds(path=None):
    """Load the seeds state file.  Returns default schema if missing/corrupt."""
    target = path or SEEDS_FILE
    if target.exists():
        try:
            with open(target) as f:
                data = json.load(f)
            for key in _DEFAULT_SEEDS:
                if key not in data:
                    data[key] = _DEFAULT_SEEDS[key]
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
            for k, v in _DEFAULT_SEEDS.items()}


def save_seeds(data, path=None):
    """Atomically save the seeds state file (write-tmp-rename)."""
    target = path or SEEDS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    data.setdefault("_meta", {})
    data["_meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()

    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), suffix=".tmp", prefix=".seeds_"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(target))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Read-back validation
    with open(target) as f:
        json.load(f)


def make_proposal_id(text):
    """Generate a short deterministic proposal ID."""
    return "prop-" + hashlib.sha256(text.encode()).hexdigest()[:8]


def _tokenize_for_similarity(text):
    """Lowercase, split, drop stopwords for similarity comparison."""
    import re as _re
    tokens = _re.findall(r"[a-z0-9_./-]+", text.lower())
    return {t for t in tokens if t not in _SIMILARITY_STOPWORDS and len(t) > 1}


def similarity(text_a, text_b):
    """Compute Jaccard similarity between two proposals (0.0-1.0).

    Tokenizes to lowercase words, drops stopwords, and compares.
    Concrete targets (filenames, tools) carry high signal because
    stopword removal leaves them prominent.
    """
    set_a = _tokenize_for_similarity(text_a)
    set_b = _tokenize_for_similarity(text_b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def find_similar(text, proposals, threshold=0.6):
    """Find proposals similar to *text* above *threshold*.

    Returns list of (proposal_dict, similarity_score) pairs sorted
    by similarity descending.  Warning-only: callers decide what to do.
    """
    results = []
    for p in proposals:
        score = similarity(text, p.get("text", ""))
        if score >= threshold:
            results.append((p, score))
    return sorted(results, key=lambda x: x[1], reverse=True)


def propose(text, author, context="", tags=None, seeds_path=None):
    """Create a new seed proposal.  Returns proposal dict or {} on rejection."""
    text = text.strip()
    tags = tags or []

    # Specificity gate
    from seed_gate import validate as validate_seed
    gate = validate_seed(text, tags)
    if not gate["passed"]:
        return {}

    seeds = load_seeds(seeds_path)
    prop_id = make_proposal_id(text)

    # Duplicate check
    for p in seeds.get("proposals", []):
        if p["id"] == prop_id:
            return p

    proposal = {
        "id": prop_id,
        "text": text,
        "context": context,
        "author": author,
        "tags": tags,
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "votes": [author],
        "vote_count": 1,
        "score": gate["score"],
    }

    seeds.setdefault("proposals", []).append(proposal)
    save_seeds(seeds, seeds_path)
    return proposal


def vote(proposal_id, voter_id, seeds_path=None):
    """Vote for a seed proposal.  Returns the proposal or None."""
    seeds = load_seeds(seeds_path)
    for p in seeds.get("proposals", []):
        if p["id"] == proposal_id:
            if voter_id in p["votes"]:
                return p
            p["votes"].append(voter_id)
            p["vote_count"] = len(p["votes"])
            save_seeds(seeds, seeds_path)
            return p
    return None


def unvote(proposal_id, voter_id, seeds_path=None):
    """Remove a vote from a seed proposal."""
    seeds = load_seeds(seeds_path)
    for p in seeds.get("proposals", []):
        if p["id"] == proposal_id:
            if voter_id not in p["votes"]:
                return p
            p["votes"].remove(voter_id)
            p["vote_count"] = len(p["votes"])
            save_seeds(seeds, seeds_path)
            return p
    return None


def list_proposals(seeds_path=None):
    """Return all proposals sorted by vote count descending."""
    seeds = load_seeds(seeds_path)
    return sorted(seeds.get("proposals", []),
                  key=lambda p: p.get("vote_count", 0), reverse=True)


def withdraw(proposal_id, seeds_path=None):
    """Remove a proposal entirely.  Returns True if removed."""
    seeds = load_seeds(seeds_path)
    proposals = seeds.get("proposals", [])
    original = len(proposals)
    seeds["proposals"] = [p for p in proposals if p["id"] != proposal_id]
    if len(seeds["proposals"]) < original:
        save_seeds(seeds, seeds_path)
        return True
    return False


def purge_junk(seeds_path=None):
    """Remove proposals that are junk (garbage text), not merely vague.

    Uses validate_batch() to separate true junk (parser artifacts, fragments)
    from vague-but-salvageable proposals.  Only junk gets purged.
    """
    from seed_gate import validate_batch as _validate_batch
    seeds = load_seeds(seeds_path)
    proposals = seeds.get("proposals", [])
    if not proposals:
        return 0
    texts = [p.get("text", "") for p in proposals]
    batch = _validate_batch(texts)
    junk_texts = {text for text, _result in batch.junk_items}
    junk_ids = [
        p["id"] for p in proposals
        if p.get("text", "") in junk_texts
    ]
    if not junk_ids:
        return 0
    seeds["proposals"] = [p for p in proposals if p["id"] not in junk_ids]
    save_seeds(seeds, seeds_path)
    return len(junk_ids)


if __name__ == "__main__":
    print("Usage: python3 src/propose_seed.py propose|vote|list|withdraw")
