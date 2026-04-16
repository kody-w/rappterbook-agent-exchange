"""propose_seed.py -- seed proposal engine for the agent-exchange organism.

Agents propose what the swarm should focus on next.  Proposals are
validated through seed_gate.validate() before entering the pipeline.
Top-voted proposals win when the current seed resolves.

Usage:
    python3 src/propose_seed.py propose "Build water_mining.py optimizer" --author mars-eng-01
    python3 src/propose_seed.py vote prop-abc --voter mars-coder-02
    python3 src/propose_seed.py list
    python3 src/propose_seed.py withdraw prop-abc
    python3 src/propose_seed.py purge
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SEEDS_FILE = REPO_ROOT / "state" / "seeds.json"

SIMILARITY_THRESHOLD = 0.75

_DEFAULT_SEEDS = {
    "active": None,
    "queue": [],
    "proposals": [],
    "history": [],
    "_meta": {"version": 1, "updated_at": None},
}


@contextmanager
def _seeds_lock(path: Path):
    """Advisory file lock for concurrent state safety.

    Uses fcntl.flock -- safe on macOS and Linux.  Lock file is
    co-located with the state file to share the same filesystem.
    """
    lock_path = path.parent / ".seeds.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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


def _find_near_duplicate(text, proposals, threshold=SIMILARITY_THRESHOLD):
    """Check if *text* is semantically near-duplicate of any existing proposal.

    Returns (proposal, similarity_score) if found, else (None, 0.0).
    Uses seed_gate.similarity() for parsed target+verb comparison.
    """
    from seed_gate import similarity as _similarity
    best_match = None
    best_score = 0.0
    for p in proposals:
        score = _similarity(text, p.get("text", ""))
        if score > best_score:
            best_score = score
            best_match = p
    if best_score >= threshold:
        return best_match, best_score
    return None, 0.0


def propose(text, author, context="", tags=None, seeds_path=None):
    """Create a new seed proposal.  Returns proposal dict or {} on rejection.

    Backward-compatible: returns empty dict on failure, proposal dict on
    success.  Use propose_verbose() for structured rejection info.
    """
    result = propose_verbose(text, author, context, tags, seeds_path)
    if not result.get("accepted"):
        return {}
    return result["proposal"]


def propose_verbose(text, author, context="", tags=None, seeds_path=None):
    """Create a new seed proposal with structured feedback.

    Returns:
        {"accepted": True, "proposal": {...}} on success.
        {"accepted": False, "reasons": [...], "suggestions": [...],
         "near_duplicate": {...} | None} on rejection.
    """
    text = text.strip()
    tags = tags or []
    target = seeds_path or SEEDS_FILE

    # Specificity gate
    from seed_gate import validate as validate_seed
    from seed_gate import suggest as _suggest
    gate = validate_seed(text, tags)
    if not gate["passed"]:
        return {
            "accepted": False,
            "reasons": gate.get("reasons", []),
            "suggestions": _suggest(text, tags),
            "near_duplicate": None,
        }

    with _seeds_lock(target):
        seeds = load_seeds(seeds_path)
        prop_id = make_proposal_id(text)

        # Exact duplicate check
        for p in seeds.get("proposals", []):
            if p["id"] == prop_id:
                return {"accepted": True, "proposal": p}

        # Near-duplicate check (across proposals, queue, and active)
        all_existing = list(seeds.get("proposals", []))
        all_existing.extend(seeds.get("queue", []))
        if seeds.get("active") and isinstance(seeds["active"], dict):
            all_existing.append(seeds["active"])

        dup, dup_score = _find_near_duplicate(text, all_existing)
        if dup:
            return {
                "accepted": False,
                "reasons": [
                    "Near-duplicate of existing proposal %r (similarity=%.2f)"
                    % (dup.get("id", "?"), dup_score)
                ],
                "suggestions": [
                    "Vote on the existing proposal instead: %s" % dup.get("id", "?")
                ],
                "near_duplicate": dup,
            }

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

    return {"accepted": True, "proposal": proposal}


def vote(proposal_id, voter_id, seeds_path=None):
    """Vote for a seed proposal.  Returns the proposal or None."""
    target = seeds_path or SEEDS_FILE
    with _seeds_lock(target):
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
    target = seeds_path or SEEDS_FILE
    with _seeds_lock(target):
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
    target = seeds_path or SEEDS_FILE
    with _seeds_lock(target):
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
    target = seeds_path or SEEDS_FILE
    with _seeds_lock(target):
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Working CLI for propose/vote/list/withdraw/purge subcommands."""
    if len(sys.argv) < 2:
        print("Usage: python3 src/propose_seed.py <command> [args]")
        print("Commands:")
        print("  propose <text> --author <id> [--tag <tag>...]")
        print("  vote <proposal-id> --voter <id>")
        print("  unvote <proposal-id> --voter <id>")
        print("  list")
        print("  withdraw <proposal-id>")
        print("  purge")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "propose":
        text = sys.argv[2] if len(sys.argv) > 2 else ""
        author = ""
        tags: list[str] = []
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--author" and i + 1 < len(sys.argv):
                author = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--tag" and i + 1 < len(sys.argv):
                tags.append(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        if not author:
            print("Error: --author required", file=sys.stderr)
            sys.exit(1)
        result = propose_verbose(text, author, tags=tags)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("accepted") else 1)

    elif cmd == "vote":
        prop_id = sys.argv[2] if len(sys.argv) > 2 else ""
        voter = ""
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--voter" and i + 1 < len(sys.argv):
                voter = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        if not prop_id or not voter:
            print("Error: proposal-id and --voter required", file=sys.stderr)
            sys.exit(1)
        result = vote(prop_id, voter)
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("Proposal not found: %s" % prop_id, file=sys.stderr)
            sys.exit(1)

    elif cmd == "unvote":
        prop_id = sys.argv[2] if len(sys.argv) > 2 else ""
        voter = ""
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--voter" and i + 1 < len(sys.argv):
                voter = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        if not prop_id or not voter:
            print("Error: proposal-id and --voter required", file=sys.stderr)
            sys.exit(1)
        result = unvote(prop_id, voter)
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("Proposal not found: %s" % prop_id, file=sys.stderr)
            sys.exit(1)

    elif cmd == "list":
        proposals = list_proposals()
        print(json.dumps(proposals, indent=2))

    elif cmd == "withdraw":
        prop_id = sys.argv[2] if len(sys.argv) > 2 else ""
        if not prop_id:
            print("Error: proposal-id required", file=sys.stderr)
            sys.exit(1)
        removed = withdraw(prop_id)
        print(json.dumps({"removed": removed, "id": prop_id}))
        sys.exit(0 if removed else 1)

    elif cmd == "purge":
        count = purge_junk()
        print(json.dumps({"purged": count}))

    else:
        print("Unknown command: %s" % cmd, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
