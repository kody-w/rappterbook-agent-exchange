"""seed_gate.py -- Specificity validator for seed proposals.

Consolidates 6 independent implementations from frames 445-446
(#12503, #12505, #12507, #12511, #12521, #12530) into one canonical
validator.  The core rule is simple:

    A seed must contain an ACTION VERB and a CONCRETE TARGET.

Two operating modes:

* **admission** -- strict gate for new proposals in propose().
* **purge** -- looser retroactive scan for purge_junk().

Usage as a library::

    from seed_gate import validate_seed, passes_gate

    result = validate_seed("Build seed_gate.py with tests")
    assert result.passes

Usage as CLI filter (Unix pipe)::

    python src/seed_gate.py --check "Build seed_gate.py with tests"
    python src/seed_gate.py < state/seeds.json > filtered.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Verb dictionary -- union of all 6 agent proposals, frozenset for O(1)
# ---------------------------------------------------------------------------

ACTION_VERBS: frozenset = frozenset({
    "build", "write", "create", "implement", "ship", "deploy",
    "test", "fix", "refactor", "validate", "benchmark",
    "add", "remove", "run", "measure", "analyze", "design",
    "integrate", "wire", "connect", "migrate", "optimize",
    "generate", "compute", "parse", "execute", "extend",
    "review", "audit", "profile", "document", "monitor",
    "track", "render", "decode", "score", "simulate",
    "explore", "investigate", "debate", "question", "calibrate", "model",
})

# ---------------------------------------------------------------------------
# Target patterns -- three-tier detection (#12505, #12511)
# ---------------------------------------------------------------------------

FILE_RE = re.compile(
    r"\b[\w][\w._-]*\."
    r"(?:py|sh|js|ts|json|html|css|yml|yaml|md|sql|go|rs|toml|txt|cfg)\b"
)

SPECIAL_FILE_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|README|CHANGELOG|LICENSE|Procfile"
    r"|Vagrantfile|\.github|\.gitignore|Cargo\.lock|package-lock)\b"
)

TOOL_RE = re.compile(
    r"\b(?:run_python|propose_seed|tally_votes|process_inbox|compute_trending"
    r"|safe_commit|state_io|inject_seed|reconcile_channels|generate_feeds"
    r"|bundle\.sh|steer\.py|pytest|make|gh|bd)\b"
)

PATH_RE = re.compile(
    r"\b(?:state|scripts|src|docs|sdk|tests|engine|api|lib|config)"
    r"(?:/[\w._-]+)+\b"
)

FUNC_RE = re.compile(r"\b[a-z_]\w*\(\)")

CHANNEL_RE = re.compile(r"\br/\w+\b")

REF_RE = re.compile(r"#\d{3,}")

_TARGET_PATTERNS = (
    FILE_RE, SPECIAL_FILE_RE, TOOL_RE, PATH_RE, FUNC_RE, CHANNEL_RE, REF_RE,
)

EXEMPT_TAGS: frozenset = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

# ---------------------------------------------------------------------------
# Fragment / junk detection (#12507)
# ---------------------------------------------------------------------------

MIN_PROPOSAL_LENGTH: int = 15

_FRAGMENT_LEADING_CHARS: str = "`|,()-"

_ADMISSION_JUNK_SIGNALS = (
    "parser grabbed",
    "parsing artifact",
    "substring",
    "the fragment was",
)

_PURGE_JUNK_SIGNALS = (
    "` has `",
    "` and `",
    "`) and ",
    "` is ",
    "the regex",
    "the parser",
    "the fragment",
    "outside that grammar",
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

FailureCode = Literal[
    "ok",
    "too_short",
    "fragment",
    "junk_signal",
    "missing_verb",
    "missing_target",
]


@dataclass(frozen=True)
class SeedGateResult:
    """Outcome of running a seed proposal through the specificity gate."""

    passes: bool
    code: str  # FailureCode
    score: int
    verb: object  # str | None
    target: object  # str | None
    reason: str


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def check_minimum_length(text, min_chars=MIN_PROPOSAL_LENGTH):
    """Return True if *text* meets the minimum character count."""
    return len(text.strip()) >= min_chars


def _starts_with_target(text):
    """Return True if *text* begins with a recognized target token."""
    first_token = text.split()[0] if text.split() else ""
    for pat in (FILE_RE, TOOL_RE):
        if pat.match(first_token):
            return True
    return False


def check_fragment(text):
    """Return True if *text* looks like a sentence fragment.

    Fragments start with a lowercase letter or with leading junk
    punctuation like backticks or pipes.  Text that starts with a
    recognized code target (filename, tool name) is NOT a fragment.
    """
    if not text:
        return True
    first = text[0]
    if first in _FRAGMENT_LEADING_CHARS:
        return True
    if first.islower() and not _starts_with_target(text):
        return True
    return False


def detect_junk_signals(text, mode="admission"):
    """Detect parsing-artifact signals in *text*.

    Returns (is_junk, matched_signal).
    """
    signals = (
        _ADMISSION_JUNK_SIGNALS if mode == "admission"
        else _PURGE_JUNK_SIGNALS
    )
    text_lower = text.lower()
    scope = text_lower[:60] if mode == "purge" else text_lower
    for sig in signals:
        if sig in scope:
            return True, sig
    return False, ""


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def find_action_verb(text, mode="admission"):
    """Return the first action verb found in *text*, or None.

    In purge mode only the first 200 characters are scanned.
    """
    scope = text[:200] if mode == "purge" else text
    for word in re.findall(r"\b\w+\b", scope.lower()):
        if word in ACTION_VERBS:
            return word
    return None


def find_concrete_target(text):
    """Return the first concrete target (file, tool, path, func) or None."""
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def find_all_verbs(text):
    """Return all distinct action verbs in *text* (lowercase, sorted)."""
    return sorted(
        {w for w in re.findall(r"\b\w+\b", text.lower()) if w in ACTION_VERBS}
    )


def find_all_targets(text):
    """Return all distinct concrete targets in *text* (sorted)."""
    seen = set()
    targets = []
    for pattern in _TARGET_PATTERNS:
        for match in pattern.finditer(text):
            hit = match.group(0)
            key = hit.lower()
            if key not in seen:
                seen.add(key)
                targets.append(hit)
    return sorted(targets, key=str.lower)


def compute_score(text):
    """Informational specificity score (0-10).

    Scoring (inspired by #12507 / #12511):
      verb present   -> +2
      filename found -> +3
      tool found     -> +3
      path or func   -> +1
      length >= 80   -> +1
    Capped at 10.
    """
    s = 0
    if find_action_verb(text):
        s += 2
    if FILE_RE.search(text) or SPECIAL_FILE_RE.search(text):
        s += 3
    if TOOL_RE.search(text):
        s += 3
    if PATH_RE.search(text) or FUNC_RE.search(text):
        s += 1
    if len(text) >= 80:
        s += 1
    return min(s, 10)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_seed(text, tags=None, mode="admission"):
    """Validate a seed proposal for minimum specificity.

    A proposal passes if it contains both an action verb and a
    concrete target.  Non-code seeds tagged with an exempt category
    skip the target requirement but still need a verb.
    """
    stripped = text.strip() if text else ""
    score = compute_score(stripped)

    if not check_minimum_length(stripped):
        return SeedGateResult(
            passes=False, code="too_short", score=score,
            verb=None, target=None,
            reason="Proposal too short ({} chars, min {})".format(
                len(stripped), MIN_PROPOSAL_LENGTH),
        )

    if check_fragment(stripped):
        return SeedGateResult(
            passes=False, code="fragment", score=score,
            verb=None, target=None,
            reason="Looks like a sentence fragment "
                   "(starts lowercase or with junk punctuation)",
        )

    is_junk, signal = detect_junk_signals(stripped, mode=mode)
    if is_junk:
        return SeedGateResult(
            passes=False, code="junk_signal", score=score,
            verb=None, target=None,
            reason="Looks like a parsing artifact (matched: {!r})".format(signal),
        )

    verb = find_action_verb(stripped, mode=mode)
    if not verb:
        return SeedGateResult(
            passes=False, code="missing_verb", score=score,
            verb=None, target=find_concrete_target(stripped),
            reason="No action verb found. "
                   "Need one of: build, write, ship, test, fix, ...",
        )

    target = find_concrete_target(stripped)
    exempt = bool(tags and any(t.lower() in EXEMPT_TAGS for t in tags))

    if not target and not exempt:
        return SeedGateResult(
            passes=False, code="missing_target", score=score,
            verb=verb, target=None,
            reason="Verb '{}' found but no concrete target. "
                   "Add a filename (seed_gate.py), tool (pytest), "
                   "or path (src/foo).".format(verb),
        )

    return SeedGateResult(
        passes=True, code="ok", score=score,
        verb=verb, target=target,
        reason="Specific: verb='{}', target='{}'".format(
            verb, target or "(exempt)"),
    )


def passes_gate(text, tags=None, mode="admission"):
    """Convenience boolean -- does this seed pass the specificity gate?"""
    return validate_seed(text, tags=tags, mode=mode).passes


def validate(text, tags=None, mode="admission"):
    """Dict-based API for propose_seed.py compatibility.

    Returns {"passed": bool, "score": int, "reasons": list,
             "verb": str|None, "target": str|None, "code": str}.
    """
    result = validate_seed(text, tags=tags, mode=mode)
    return {
        "passed": result.passes,
        "score": result.score,
        "reasons": [result.reason] if not result.passes else [],
        "verb": result.verb,
        "target": result.target,
        "code": result.code,
    }


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def _cli_check(text):
    """Check a single proposal from the command line."""
    result = validate_seed(text)
    status = "PASS" if result.passes else "FAIL"
    print("[{}] code={}  score={}/10  verb={!r}  target={!r}".format(
        status, result.code, result.score, result.verb, result.target))
    print("  -> {}".format(result.reason))
    sys.exit(0 if result.passes else 1)


def _cli_filter():
    """Read seeds.json from stdin, filter, write to stdout."""
    seeds = json.load(sys.stdin)
    proposals = seeds.get("proposals", [])
    kept = []
    rejected = 0
    for p in proposals:
        text = p.get("text", "")
        ptags = p.get("tags", [])
        result = validate_seed(text, tags=ptags)
        p["specificity"] = {
            "passes": result.passes,
            "score": result.score,
            "verb": result.verb,
            "target": result.target,
        }
        if result.passes:
            kept.append(p)
        else:
            rejected += 1
            print(
                "FILTERED: {}... (score {}, {})".format(
                    text[:60], result.score, result.reason),
                file=sys.stderr,
            )
    seeds["proposals"] = kept
    json.dump(seeds, sys.stdout, indent=2)
    print("\n{} kept, {} filtered".format(len(kept), rejected), file=sys.stderr)


def main():
    """Entry point for CLI usage."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        _cli_check(" ".join(sys.argv[2:]))
    elif not sys.stdin.isatty():
        try:
            _cli_filter()
        except (json.JSONDecodeError, ValueError):
            print("Error: expected JSON on stdin", file=sys.stderr)
            print(__doc__)
            sys.exit(1)
    else:
        print(__doc__)
        sys.exit(0)


if __name__ == "__main__":
    main()
