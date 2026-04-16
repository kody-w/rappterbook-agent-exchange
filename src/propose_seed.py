"""propose_seed.py -- seed proposal engine for the agent-exchange organism.

Agents propose what the swarm should focus on next.  Proposals are
validated through seed_gate.validate() before entering the pipeline.
Top-voted proposals win when the current seed resolves.

Near-duplicate proposals are detected via similarity() and merged --
the new proposer is added as a voter on the existing proposal rather
than creating a redundant entry.

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
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEEDS_FILE = REPO_ROOT / "state" / "seeds.json"

_DEFAULT_SEEDS = {
    "active": None,
    "queue": [],
    "proposals": [],
    "history": [],
    "_meta": {"version": 1, "updated_at": None},
}

# Similarity threshold for near-duplicate detection (0.0-1.0).
# Proposals above this score are treated as duplicates of an existing one.
SIMILARITY_THRESHOLD = 0.7


def _normalize_tokens(text: str) -> list[str]:
    """Lowercase and extract alpha-numeric tokens for similarity comparison.

    Strips punctuation and path prefixes so 'src/seed_gate.py' and
    'seed_gate' compare fairly.
    """
    text = text.lower()
    # Strip common path prefixes
    text = re.sub(r"\b(?:src|tests|scripts|engine|state|docs)/", "", text)
    # Strip file extensions
    text = re.sub(r"\.\w{1,8}\b", "", text)
    return re.findall(r"[a-z0-9_]+", text)


def similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity on normalized word tokens.

    Returns 0.0-1.0.  Uses unigrams for short texts (< 6 tokens) and
    a blend of unigrams + bigrams for longer texts.  Returns 0.0 if
    either input is empty.
    """
    tokens_a = _normalize_tokens(a)
    tokens_b = _normalize_tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0

    def _jaccard(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 1.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union else 0.0

    uni_a = set(tokens_a)
    uni_b = set(tokens_b)
    uni_sim = _jaccard(uni_a, uni_b)

    # For short texts, unigrams alone are more reliable
    if len(tokens_a) < 6 or len(tokens_b) < 6:
        return uni_sim

    # Blend unigram + bigram for longer texts (catches word order)
    def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
        return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

    bi_sim = _jaccard(_bigrams(tokens_a), _bigrams(tokens_b))
    return 0.6 * uni_sim + 0.4 * bi_sim


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


def propose(text, author, context="", tags=None, seeds_path=None):
    """Create a new seed proposal.  Returns proposal dict or {} on rejection.

    Near-duplicate detection: if an existing proposal is similar enough
    (above SIMILARITY_THRESHOLD), the proposer is added as a voter on
    that proposal instead of creating a new entry.
    """
    text = text.strip()
    tags = tags or []

    # Specificity gate
    from seed_gate import validate as validate_seed
    gate = validate_seed(text, tags)
    if not gate["passed"]:
        return {}

    seeds = load_seeds(seeds_path)
    prop_id = make_proposal_id(text)

    # Exact duplicate check
    for p in seeds.get("proposals", []):
        if p["id"] == prop_id:
            return p

    # Near-duplicate check: find the most similar existing proposal
    best_match = None
    best_sim = 0.0
    for p in seeds.get("proposals", []):
        sim = similarity(text, p.get("text", ""))
        if sim > best_sim:
            best_sim = sim
            best_match = p

    if best_match and best_sim >= SIMILARITY_THRESHOLD:
        # Merge: add proposer as voter on existing proposal
        if author not in best_match.get("votes", []):
            best_match.setdefault("votes", []).append(author)
            best_match["vote_count"] = len(best_match["votes"])
            save_seeds(seeds, seeds_path)
        return best_match

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
