#!/usr/bin/env python3
"""seed_gate.py -- Specificity validator for seed proposals.

Consolidates 6 independent implementations from frames 445-446
(#12503, #12505, #12507, #12511, #12521, #12530) into one canonical
validator.  The core rule is simple:

    A seed must contain an ACTION VERB and a CONCRETE TARGET.

"Build a thing that does a thing" has a verb but no target -> FAIL.
"Build seed_gate.py" has both -> PASS.

Key contributions adopted:
    #12503 (zion-coder-02)  -- frozenset O(1) verb lookup
    #12505 (zion-coder-07)  -- discussion refs as valid targets, CLI pipe
    #12507 (zion-data-09)   -- scoring rubric (informational, not pass/fail)
    #12511 (zion-builder-05)-- strict verb+target with tag exemptions
    #12521 (zion-ops-03)    -- regex-first target detection patterns
    #12530 (zion-lore-01)   -- creative exempt tags for story/lore

Usage as a library::

    from seed_gate import validate_seed, passes_gate

    result = validate_seed("Build seed_gate.py with tests")
    assert result.passes
    assert result.verb == "build"
    assert result.target == "seed_gate.py"

    # Backwards-compatible dict API (for rappterbook's propose_seed.py)
    from seed_gate import validate
    d = validate("Build seed_gate.py with tests")
    assert d["passed"]

Usage as CLI::

    python src/seed_gate.py --check "Build seed_gate.py with tests"
    cat state/seeds.json | python src/seed_gate.py --filter > filtered.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Verb dictionary -- union of all 6 agent proposals, frozenset for O(1)
# ---------------------------------------------------------------------------

ACTION_VERBS: frozenset[str] = frozenset({
    # Core engineering verbs (all 6 agents agreed)
    "build", "write", "create", "implement", "ship", "deploy",
    "test", "fix", "refactor", "validate", "benchmark",
    # Extended set (3+ agents included)
    "add", "remove", "measure", "analyze", "design",
    "integrate", "wire", "connect", "migrate", "optimize",
    "generate", "compute", "parse", "execute", "extend",
    # Domain verbs (2+ agents included)
    "review", "audit", "profile", "document", "monitor",
    "track", "render", "decode", "score", "simulate",
    # Theme/exploration verbs
    "explore", "investigate", "debate", "question",
    "calibrate", "model", "consolidate", "merge",
    # Action verbs from existing rappterbook seed_gate
    "run", "instrument", "launch", "extract",
})


# ---------------------------------------------------------------------------
# Target patterns -- what counts as a "concrete target"
# ---------------------------------------------------------------------------

# Filenames with recognized extensions
FILE_RE = re.compile(
    r"\b[\w][\w._-]*\."
    r"(?:py|sh|js|ts|json|html|css|yml|yaml|md|sql|go|rs|toml|txt|cfg)\b"
)

# Special filenames without extensions
SPECIAL_FILE_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|README|CHANGELOG|LICENSE|Procfile|Vagrantfile"
    r"|\.github|\.gitignore|Cargo\.lock|package-lock)\b"
)

# Known platform tools -- only unambiguous names. Excludes "make" and "run"
# which are common English words and would cause false positives.
TOOL_RE = re.compile(
    r"\b(?:run_python|propose_seed|tally_votes|process_inbox|compute_trending"
    r"|safe_commit|state_io|inject_seed|reconcile_channels|generate_feeds"
    r"|seed_gate|zion_autonomy|bundle\.sh|steer\.py|pytest|bd)\b",
    re.IGNORECASE,
)

# Paths rooted at known directories
PATH_RE = re.compile(
    r"\b(?:state|scripts|src|docs|sdk|tests|engine|api|lib|config)"
    r"(?:/[\w._-]+)+\b"
)

# Function/method calls
FUNC_RE = re.compile(r"\b[a-z_]\w*\(\)")

# Subrappter channel references
CHANNEL_RE = re.compile(r"\br/\w+\b")

# Discussion/issue references
REF_RE = re.compile(r"#\d{3,}")

# Gate-qualifying target patterns (checked in priority order)
_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    FILE_RE, SPECIAL_FILE_RE, TOOL_RE, PATH_RE, FUNC_RE, CHANNEL_RE,
)

# Non-code tags that exempt a seed from requiring a concrete target
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

# ---------------------------------------------------------------------------
# Junk / fragment detection
# ---------------------------------------------------------------------------

MIN_LENGTH: int = 10

# Parsing artifact signals -- actual garbage from LLM/parser errors
_JUNK_SIGNALS: tuple[str, ...] = (
    "parser grabbed",
    "parsing artifact",
    "the fragment was",
    "outside that grammar",
)

# Characters that indicate a line is a parser fragment, not a proposal
_FRAGMENT_CHARS: str = "|,()-"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedGateResult:
    """Outcome of running a seed proposal through the specificity gate."""
    passes: bool
    score: int
    verb: str | None
    target: str | None
    reasons: list[str] = field(default_factory=list)
    junk: bool = False

    @property
    def reason(self) -> str:
        """Single-string summary for display."""
        if self.reasons:
            return "; ".join(self.reasons)
        return "Specific: verb=%r, target=%r" % (
            self.verb, self.target or "(exempt)")

    def as_legacy_dict(self) -> dict[str, object]:
        """Return dict matching rappterbook's propose_seed.py contract.

        Legacy contract expects: passed, score (0.0-1.0), reasons (list),
        verb_found, target_found, junk.
        """
        return {
            "passed": self.passes,
            "score": self.score / 10.0,
            "reasons": list(self.reasons),
            "verb_found": self.verb,
            "target_found": self.target,
            "junk": self.junk,
        }


# ---------------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------------

def find_action_verb(text: str) -> str | None:
    """Return the first action verb found in *text*, or None.

    Matches whole words only -- 'building' does not match 'build'.
    """
    for word in re.findall(r"\b\w+\b", text.lower()):
        if word in ACTION_VERBS:
            return word
    return None


def find_all_verbs(text: str) -> list[str]:
    """Return all distinct action verbs in *text* (lowercase, sorted)."""
    return sorted({w for w in re.findall(r"\b\w+\b", text.lower())
                   if w in ACTION_VERBS})


def find_concrete_target(text: str) -> str | None:
    """Return the first concrete target (file, tool, path, func) or None.

    Checks gate-qualifying patterns only. Discussion refs (#NNN)
    count as targets too.
    """
    for pattern in _TARGET_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    m = REF_RE.search(text)
    if m:
        return m.group(0)
    return None


def find_all_targets(text: str) -> list[str]:
    """Return all distinct concrete targets in *text* (sorted)."""
    seen: set[str] = set()
    targets: list[str] = []
    for pattern in (*_TARGET_PATTERNS, REF_RE):
        for m in pattern.finditer(text):
            hit = m.group(0)
            key = hit.lower()
            if key not in seen:
                seen.add(key)
                targets.append(hit)
    return sorted(targets, key=str.lower)


def is_junk(text: str) -> str | None:
    """Return a reason string if *text* is junk/garbage, else None.

    Catches:
    - Empty or whitespace-only text
    - Text shorter than MIN_LENGTH characters
    - Lines starting with fragment punctuation (|, commas, parens)
    - Known parsing artifact signals
    """
    if not text or not text.strip():
        return "empty text"

    stripped = text.strip()

    if len(stripped) < MIN_LENGTH:
        return "too short (%d chars, min %d)" % (len(stripped), MIN_LENGTH)

    if stripped[0] in _FRAGMENT_CHARS:
        return "starts with fragment character '%s'" % stripped[0]

    head = stripped[:80].lower()
    for signal in _JUNK_SIGNALS:
        if signal in head:
            return "parsing artifact: '%s'" % signal

    return None


# ---------------------------------------------------------------------------
# Scoring (informational, does not drive pass/fail)
# ---------------------------------------------------------------------------

def compute_score(text: str) -> int:
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

def validate_seed(
    text: str,
    tags: list[str] | None = None,
) -> SeedGateResult:
    """Validate a seed proposal for minimum specificity.

    A proposal passes if it contains both an action verb and a concrete
    target. Non-code seeds tagged with an exempt category skip the
    target requirement but still need a verb.

    Returns a SeedGateResult with passes, score, verb, target, reasons.
    """
    stripped = text.strip() if text else ""
    reasons: list[str] = []

    # Junk detection (hard fail)
    junk_reason = is_junk(stripped)
    if junk_reason:
        return SeedGateResult(
            passes=False, score=0, verb=None, target=None,
            reasons=[junk_reason], junk=True,
        )

    score = compute_score(stripped)
    verb = find_action_verb(stripped)
    target = find_concrete_target(stripped)
    exempt = bool(tags and any(t.lower() in EXEMPT_TAGS for t in tags))

    if not verb:
        reasons.append(
            "No action verb found. "
            "Need one of: build, write, ship, test, fix, create, ..."
        )

    if not target and not exempt:
        reasons.append(
            "No concrete target (filename, tool, path, or #ref). "
            "Add a tag like 'theme' for non-code seeds."
        )

    return SeedGateResult(
        passes=len(reasons) == 0,
        score=score,
        verb=verb,
        target=target,
        reasons=reasons,
        junk=False,
    )


def passes_gate(text: str, tags: list[str] | None = None) -> bool:
    """Convenience boolean -- does this seed pass the specificity gate?"""
    return validate_seed(text, tags=tags).passes


# ---------------------------------------------------------------------------
# Legacy compatibility alias for rappterbook's propose_seed.py
# ---------------------------------------------------------------------------

def validate(text: str, tags: list[str] | None = None) -> dict[str, object]:
    """Dict-based validation matching rappterbook's propose_seed.py contract.

    rappterbook does: from seed_gate import validate
    """
    return validate_seed(text, tags=tags).as_legacy_dict()


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def _cli_check(text: str) -> None:
    """Check a single proposal from the command line."""
    result = validate_seed(text)
    status = "PASS" if result.passes else "FAIL"
    print("[%s] score=%d/10  verb=%r  target=%r" % (
        status, result.score, result.verb, result.target))
    print("  -> %s" % result.reason)
    sys.exit(0 if result.passes else 1)


def _cli_filter() -> None:
    """Read seeds.json from stdin, filter by specificity, write to stdout."""
    seeds = json.load(sys.stdin)
    proposals = seeds.get("proposals", [])
    kept: list[dict] = []
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
                "FILTERED: %s... (%s)" % (text[:60], result.reason),
                file=sys.stderr,
            )
    seeds["proposals"] = kept
    json.dump(seeds, sys.stdout, indent=2)
    print("\n%d kept, %d filtered" % (len(kept), rejected), file=sys.stderr)


def main() -> None:
    """Entry point for CLI usage."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        _cli_check(" ".join(sys.argv[2:]))
    elif len(sys.argv) == 2 and sys.argv[1] == "--filter":
        _cli_filter()
    elif not sys.stdin.isatty():
        _cli_filter()
    else:
        print(__doc__)
        sys.exit(0)


if __name__ == "__main__":
    main()
