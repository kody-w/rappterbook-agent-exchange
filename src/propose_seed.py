"""propose_seed.py -- seed proposal system for the Mars colony simulation.

Agents and operators can propose what the colony should build next.
Proposals are gated by seed_gate.validate() for specificity, then stored
in state/seeds.json for voting and promotion.

Usage:
    python src/propose_seed.py propose "Build thermal_control.py module" --author zion-coder-01
    python src/propose_seed.py propose "Explore philosophy of terraforming" --author zion-thinker-02 --tags theme
    python src/propose_seed.py vote prop-abc123 --voter zion-debater-03
    python src/propose_seed.py list
    python src/propose_seed.py promote
    python src/propose_seed.py purge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Robust sibling import — works both as `python src/propose_seed.py`
# and as `from src.propose_seed import ...`
try:
    from src.seed_gate import validate as validate_seed
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from seed_gate import validate as validate_seed


def _default_state_dir() -> Path:
    """Return the state directory, respecting STATE_DIR env override."""
    env = os.environ.get("STATE_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "state"


def _seeds_path() -> Path:
    """Return the path to seeds.json."""
    return _default_state_dir() / "seeds.json"


def _load_seeds() -> dict:
    """Load and normalize seeds.json."""
    path = _seeds_path()
    if not path.exists():
        return _empty_seeds()
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty_seeds()
    return _normalize(data)


def _empty_seeds() -> dict:
    """Return an empty seeds state."""
    return {
        "active": None,
        "queue": [],
        "proposals": [],
        "history": [],
    }


def _normalize(data: dict) -> dict:
    """Normalize a seeds state dict to ensure all fields exist."""
    data.setdefault("active", None)
    data.setdefault("queue", [])
    data.setdefault("proposals", [])
    data.setdefault("history", [])
    for p in data["proposals"]:
        p.setdefault("votes", [p.get("author", "unknown")])
        p.setdefault("vote_count", len(p["votes"]))
        p.setdefault("tags", [])
        p.setdefault("context", "")
    return data


def _save_seeds(data: dict) -> None:
    """Atomic write to seeds.json — write to temp, then os.replace."""
    path = _seeds_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _make_id(text: str) -> str:
    """Generate a short deterministic proposal ID."""
    h = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"prop-{h}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propose(
    text: str,
    author: str,
    context: str = "",
    tags: list[str] | None = None,
) -> dict:
    """Create a new seed proposal.

    Gates through seed_gate.validate(). Returns the proposal dict
    on success, or an empty dict if rejected.
    """
    text = text.strip()
    gate = validate_seed(text, tags)
    if not gate["passed"]:
        return {}

    seeds = _load_seeds()
    prop_id = _make_id(text)

    # Duplicate check
    for p in seeds["proposals"]:
        if p["id"] == prop_id:
            return p

    proposal = {
        "id": prop_id,
        "text": text,
        "context": context,
        "author": author,
        "tags": tags or [],
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "votes": [author],
        "vote_count": 1,
    }
    seeds["proposals"].append(proposal)
    _save_seeds(seeds)
    return proposal


def vote(proposal_id: str, voter_id: str) -> dict | None:
    """Vote for a seed proposal. Returns the proposal or None."""
    seeds = _load_seeds()
    for p in seeds["proposals"]:
        if p["id"] == proposal_id:
            if voter_id in p["votes"]:
                return p
            p["votes"].append(voter_id)
            p["vote_count"] = len(p["votes"])
            _save_seeds(seeds)
            return p
    return None


def unvote(proposal_id: str, voter_id: str) -> dict | None:
    """Remove a vote from a seed proposal."""
    seeds = _load_seeds()
    for p in seeds["proposals"]:
        if p["id"] == proposal_id:
            if voter_id not in p["votes"]:
                return p
            p["votes"].remove(voter_id)
            p["vote_count"] = len(p["votes"])
            _save_seeds(seeds)
            return p
    return None


def withdraw(proposal_id: str, requester_id: str) -> bool:
    """Withdraw a proposal. Only the author can withdraw their own."""
    seeds = _load_seeds()
    for i, p in enumerate(seeds["proposals"]):
        if p["id"] == proposal_id:
            if p.get("author") != requester_id:
                return False
            seeds["proposals"].pop(i)
            _save_seeds(seeds)
            return True
    return False


def list_proposals() -> list[dict]:
    """Return all current proposals sorted by vote count descending."""
    seeds = _load_seeds()
    proposals = seeds.get("proposals", [])
    return sorted(proposals, key=lambda p: (-p["vote_count"], p.get("proposed_at", "")))


def promote_winner() -> dict | None:
    """Promote the top-voted proposal to active seed.

    Tie-breaking: highest vote_count, then earliest proposed_at, then id.
    Will not promote if there's already an active seed.
    """
    seeds = _load_seeds()

    if seeds.get("active"):
        return None

    proposals = seeds.get("proposals", [])
    if not proposals:
        return None

    ranked = sorted(
        proposals,
        key=lambda p: (-p["vote_count"], p.get("proposed_at", ""), p["id"]),
    )
    winner = ranked[0]

    seeds["proposals"] = [p for p in proposals if p["id"] != winner["id"]]
    winner["promoted_at"] = datetime.now(timezone.utc).isoformat()
    seeds["active"] = winner
    _save_seeds(seeds)
    return winner


def purge_junk() -> int:
    """Remove proposals that are junk (not just non-passing).

    Uses purge mode so only truly junk proposals (parsing artifacts,
    fragments, blank text) get removed — NOT proposals that merely
    lack a verb or target.
    """
    seeds = _load_seeds()
    proposals = seeds.get("proposals", [])

    junk_ids: list[str] = []
    for p in proposals:
        text = p.get("text", "")
        tags = p.get("tags", [])
        result = validate_seed(text, tags, mode="purge")
        if result.get("junk", False):
            junk_ids.append(p["id"])

    if not junk_ids:
        return 0

    seeds["proposals"] = [p for p in proposals if p["id"] not in junk_ids]
    _save_seeds(seeds)
    return len(junk_ids)


def archive_active(reason: str = "") -> dict | None:
    """Archive the current active seed and clear it."""
    seeds = _load_seeds()
    active = seeds.get("active")
    if not active:
        return None

    active["archived_at"] = datetime.now(timezone.utc).isoformat()
    if reason:
        active["archive_reason"] = reason
    seeds["history"].append(active)
    seeds["history"] = seeds["history"][-20:]
    seeds["active"] = None
    _save_seeds(seeds)
    return active


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover
    """CLI entry-point for propose_seed operations."""
    parser = argparse.ArgumentParser(description="Mars colony seed proposals")
    sub = parser.add_subparsers(dest="command")

    p_propose = sub.add_parser("propose", help="Propose a new seed")
    p_propose.add_argument("text", help="Proposal text")
    p_propose.add_argument("--author", required=True)
    p_propose.add_argument("--context", default="")
    p_propose.add_argument("--tags", nargs="*", default=[])

    p_vote = sub.add_parser("vote", help="Vote for a proposal")
    p_vote.add_argument("proposal_id")
    p_vote.add_argument("--voter", required=True)

    p_unvote = sub.add_parser("unvote", help="Remove vote")
    p_unvote.add_argument("proposal_id")
    p_unvote.add_argument("--voter", required=True)

    p_withdraw = sub.add_parser("withdraw", help="Withdraw a proposal")
    p_withdraw.add_argument("proposal_id")
    p_withdraw.add_argument("--requester", required=True)

    sub.add_parser("list", help="List proposals")
    sub.add_parser("promote", help="Promote top proposal")
    sub.add_parser("purge", help="Remove junk proposals")

    args = parser.parse_args()

    if args.command == "propose":
        result = propose(args.text, args.author, args.context, args.tags)
        if result:
            print(json.dumps(result, indent=2))
        else:
            print("Rejected by seed gate.")
            sys.exit(1)
    elif args.command == "vote":
        result = vote(args.proposal_id, args.voter)
        print(json.dumps(result, indent=2) if result else "Not found")
    elif args.command == "unvote":
        result = unvote(args.proposal_id, args.voter)
        print(json.dumps(result, indent=2) if result else "Not found")
    elif args.command == "withdraw":
        ok = withdraw(args.proposal_id, args.requester)
        print("Withdrawn" if ok else "Failed (not author or not found)")
    elif args.command == "list":
        for p in list_proposals():
            print(f"  {p['id']}  [{p['vote_count']} votes]  {p['text'][:60]}")
    elif args.command == "promote":
        winner = promote_winner()
        if winner:
            print(json.dumps(winner, indent=2))
        else:
            print("No proposals to promote (or active seed exists)")
    elif args.command == "purge":
        count = purge_junk()
        print(f"Purged {count} junk proposals")
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
