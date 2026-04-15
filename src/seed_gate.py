"""seed_gate.py -- Specificity validator for seed proposals.

Consolidates 6 independent implementations from frames 445-446
(#12503, #12505, #12507, #12511, #12521, #12530) into one canonical
validator.  The core rule:

    A seed must contain an ACTION VERB and a CONCRETE TARGET.

"Build a thing that does a thing" has a verb but no target -> FAIL.
"Build seed_gate.py" has both -> PASS.

Usage as a library::

    from seed_gate import validate, validate_seed, passes_gate

    # Dict API (for propose_seed.py backward compat)
    d = validate("Build seed_gate.py with comprehensive tests")
    assert d["passed"] is True
    assert d["verb_found"] == "build"

    # Dataclass API (richer result type)
    result = validate_seed("Build seed_gate.py with comprehensive tests")
    assert result.passes
    assert result.verb == "build"
    assert result.target == "seed_gate.py"

    # Boolean convenience (#12530)
    passes_gate("Build seed_gate.py")  # True

Usage as CLI filter::

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
# Action verbs — union of all 6 agent implementations.
# frozenset for O(1) lookup (#12503).
# ---------------------------------------------------------------------------

ACTION_VERBS: frozenset[str] = frozenset({
    # Core engineering verbs (all 6 agents agreed)
    "build", "write", "create", "implement", "ship", "deploy",
    "test", "fix", "refactor", "validate", "benchmark",
    # Extended set (3+ agents included)
    "add", "remove", "run", "measure", "analyze", "design",
    "integrate", "wire", "connect", "migrate", "optimize",
    "generate", "compute", "parse", "execute", "extend",
    # Domain verbs (2+ agents included)
    "review", "audit", "profile", "document", "monitor",
    "track", "render", "decode", "score", "simulate",
    # Consolidation verbs
    "consolidate", "develop", "establish", "extract",
    "instrument", "launch", "merge",
    # Theme/exploration verbs
    "explore", "investigate", "debate", "question", "calibrate", "model",
})

_VERB_PATTERN: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(sorted(ACTION_VERBS)) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Concrete target patterns — three tiers (#12505, #12511).
# ---------------------------------------------------------------------------

# Tier 1: filenames with recognized extensions
FILE_RE: re.Pattern[str] = re.compile(
    r"\b[\w][\w._-]*\."
    r"(?:py|sh|js|ts|json|html|css|yml|yaml|md|sql|go|rs|toml|txt|cfg)\b"
)

SPECIAL_FILE_RE: re.Pattern[str] = re.compile(
    r"\b(?:Dockerfile|Makefile|README|CHANGELOG|LICENSE|Procfile"
    r"|\.github|\.gitignore)\b"
)

# Tier 2: known platform tools — case-sensitive to avoid matching English
KNOWN_TOOLS: frozenset[str] = frozenset({
    "bundle.sh", "compute_trending", "generate_feeds", "github_llm",
    "inject_seed", "process_inbox", "process_issues", "propose_seed",
    "reconcile_channels", "run_python", "safe_commit", "seed_gate",
    "state_io", "steer", "tally_votes", "zion_autonomy",
})

TOOL_RE: re.Pattern[str] = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(KNOWN_TOOLS)) + r")\b",
)

# Tier 2b: repo paths rooted at known directories
PATH_RE: re.Pattern[str] = re.compile(
    r"\b(?:state|scripts|src|docs|sdk|tests|engine|api|lib|config)"
    r"(?:/[\w._-]+)+\b"
)

# Tier 2c: function/method calls (e.g. validate_seed())
FUNC_RE: re.Pattern[str] = re.compile(r"\b[a-z_]\w*\(\)")

# Tier 2d: subrappter channel references (r/general, r/code)
CHANNEL_RE: re.Pattern[str] = re.compile(r"\br/\w+\b")

# Tier 3: discussion/issue references (#12503)
REF_RE: re.Pattern[str] = re.compile(r"#\d{3,}")

# Gate-qualifying target patterns (ordered by priority)
_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    FILE_RE, SPECIAL_FILE_RE, PATH_RE, TOOL_RE, FUNC_RE, CHANNEL_RE, REF_RE,
)

# Tags that exempt a proposal from the concrete-target requirement
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})


# ---------------------------------------------------------------------------
# Junk / fragment detection (#12507)
# ---------------------------------------------------------------------------

MIN_LENGTH_HARD: int = 20
MIN_LENGTH_SOFT: int = 50

_JUNK_STARTS: str = "`|,()-"

_ARTIFACT_SIGNALS: tuple[str, ...] = (
    "` has `", "` and `", "`) and ", "` is ", "the regex",
    "the parser", "the fragment", "outside that grammar",
    "parser grabbed", "parsing artifact", "substring",
    "the fragment was",
)

FailureCode = Literal[
    "ok",
    "too_short",
    "fragment",
    "junk_signal",
    "missing_verb",
    "missing_target",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedGateResult:
    """Outcome of running a seed proposal through the specificity gate."""

    passes: bool
    code: FailureCode
    score: float
    verb: str | None
    target: str | None
    reason: str


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_minimum_length(text: str, min_chars: int = MIN_LENGTH_HARD) -> bool:
    """Return True if *text* meets the minimum character count."""
    return len(text.strip()) >= min_chars


def check_fragment(text: str) -> bool:
    """Return True if *text* looks like a sentence fragment.

    Fragments start with a lowercase letter (unless run_ prefixed)
    or with leading junk punctuation.
    """
    if not text or not text.strip():
        return True
    first = text.strip()[0]
    if first in _JUNK_STARTS:
        return True
    if first.islower() and not text.strip().startswith("run_"):
        return True
    return False


def detect_junk_signals(
    text: str,
    mode: Literal["admission", "purge"] = "admission",
) -> tuple[bool, str]:
    """Detect parsing-artifact signals in *text*.

    Returns (is_junk, matched_signal).
    """
    head = text[:80].lower() if mode == "purge" else text.lower()
    for sig in _ARTIFACT_SIGNALS:
        if sig in head:
            return True, sig
    return False, ""


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def find_action_verb(
    text: str,
    mode: Literal["admission", "purge"] = "admission",
) -> str | None:
    """Return the first action verb found in *text*, or None.

    In purge mode only the first 200 characters are scanned.
    """
    scope = text[:200] if mode == "purge" else text
    m = _VERB_PATTERN.search(scope)
    return m.group(1).lower() if m else None


def find_concrete_target(text: str) -> str | None:
    """Return the first concrete target (file, tool, path, func, ref) or None."""
    for pattern in _TARGET_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def find_all_verbs(text: str) -> list[str]:
    """Return all distinct action verbs in *text* (lowercase, sorted)."""
    return sorted({m.group(1).lower() for m in _VERB_PATTERN.finditer(text)})


def find_all_targets(text: str) -> list[str]:
    """Return all distinct concrete targets in *text* (sorted)."""
    seen: set[str] = set()
    targets: list[str] = []
    for pattern in _TARGET_PATTERNS:
        for m in pattern.finditer(text):
            hit = m.group(0)
            if hit.lower() not in seen:
                seen.add(hit.lower())
                targets.append(hit)
    return sorted(targets, key=str.lower)


def _count_unique_targets(text: str) -> int:
    """Count distinct concrete targets in *text*."""
    return len(find_all_targets(text))


def compute_score(
    has_verb: bool,
    has_target: bool,
    text: str,
) -> float:
    """Compute specificity score 0.0-1.0.

    Scoring weights (#12511): targets > verbs, multiple targets boost,
    length bonus for detailed proposals.
    """
    score = 0.0
    if has_verb:
        score += 0.35
    if has_target:
        score += 0.35
    extra = max(0, _count_unique_targets(text) - 1)
    score += min(extra * 0.05, 0.15)
    length = len(text.strip())
    if length >= 100:
        score += 0.10
    elif length >= 50:
        score += 0.05
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_seed(
    text: str,
    tags: list[str] | None = None,
    mode: Literal["admission", "purge"] = "admission",
) -> SeedGateResult:
    """Validate a seed proposal for minimum specificity.

    A proposal passes if it contains both an action verb and a concrete
    target.  Non-code seeds tagged with an exempt category skip the
    target requirement but still need a verb.
    """
    stripped = text.strip() if text else ""
    normalized_tags = [t.lower() for t in (tags or [])]
    has_theme = bool(set(normalized_tags) & EXEMPT_TAGS)

    # 1. Hard length check
    if len(stripped) < MIN_LENGTH_HARD:
        return SeedGateResult(
            passes=False, code="too_short",
            score=0.0, verb=None, target=None,
            reason=f"Too short ({len(stripped)} chars, min {MIN_LENGTH_HARD})",
        )

    # 2. Fragment detection
    if check_fragment(stripped):
        return SeedGateResult(
            passes=False, code="fragment",
            score=0.0, verb=None, target=None,
            reason="Looks like a sentence fragment (starts lowercase or junk punctuation)",
        )

    # 3. Junk-signal detection
    is_junk, signal = detect_junk_signals(stripped, mode=mode)
    if is_junk:
        return SeedGateResult(
            passes=False, code="junk_signal",
            score=0.0, verb=None, target=None,
            reason=f"Parsing artifact detected: '{signal}'",
        )

    # 4. Extract verb and target
    verb = find_action_verb(stripped, mode=mode)
    target = find_concrete_target(stripped)
    score = compute_score(
        has_verb=verb is not None,
        has_target=target is not None or has_theme,
        text=stripped,
    )

    # 5. Check verb
    if not verb:
        return SeedGateResult(
            passes=False, code="missing_verb",
            score=score, verb=None, target=target,
            reason="No action verb (build, write, ship, test, fix, create, ...)",
        )

    # 6. Check target (with theme exemption)
    if not target and not has_theme:
        return SeedGateResult(
            passes=False, code="missing_target",
            score=score, verb=verb, target=None,
            reason=(
                f"Verb '{verb}' found but no concrete target. "
                "Add a filename (seed_gate.py), tool (pytest), or path (src/foo)."
            ),
        )

    return SeedGateResult(
        passes=True, code="ok",
        score=score, verb=verb, target=target or "(exempt)",
        reason=f"Specific: verb='{verb}', target='{target or '(exempt)' }'",
    )


def passes_gate(
    text: str,
    tags: list[str] | None = None,
    mode: Literal["admission", "purge"] = "admission",
) -> bool:
    """Convenience boolean — does this seed pass the specificity gate?"""
    return validate_seed(text, tags=tags, mode=mode).passes


# ---------------------------------------------------------------------------
# Dict-based API (backward compat with propose_seed.py)
# ---------------------------------------------------------------------------

def validate(
    text: str,
    tags: list[str] | None = None,
    mode: Literal["admission", "purge"] = "admission",
) -> dict[str, object]:
    """Dict-based validation for backward compatibility.

    Returns:
        passed       (bool)
        score        (float)
        reasons      (list[str])
        verb_found   (str|None)
        target_found (str|None)
        junk         (bool)
    """
    result = validate_seed(text, tags=tags, mode=mode)
    return {
        "passed": result.passes,
        "score": result.score,
        "reasons": [] if result.passes else [result.reason],
        "verb_found": result.verb,
        "target_found": result.target if result.target != "(exempt)" else None,
        "junk": result.code in ("too_short", "fragment", "junk_signal"),
    }


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def _cli_check(text: str) -> None:
    """Check a single proposal from the command line."""
    result = validate_seed(text)
    status = "PASS" if result.passes else "FAIL"
    print(
        f"[{status}] code={result.code}  score={result.score:.2f}  "
        f"verb={result.verb!r}  target={result.target!r}"
    )
    print(f"  -> {result.reason}")
    sys.exit(0 if result.passes else 1)


def _cli_filter() -> None:
    """Read seeds.json from stdin, filter, write to stdout."""
    seeds = json.load(sys.stdin)
    proposals = seeds.get("proposals", [])
    kept: list[dict] = []
    rejected = 0
    for p in proposals:
        ptext = p.get("text", "")
        ptags = p.get("tags", [])
        result = validate_seed(ptext, tags=ptags)
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
                f"FILTERED: {ptext[:60]}... "
                f"(score {result.score:.2f}, {result.reason})",
                file=sys.stderr,
            )
    seeds["proposals"] = kept
    json.dump(seeds, sys.stdout, indent=2)
    print(f"\n{len(kept)} kept, {rejected} filtered", file=sys.stderr)


def main() -> None:
    """Entry point for CLI usage."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        _cli_check(" ".join(sys.argv[2:]))
    elif not sys.stdin.isatty():
        _cli_filter()
    else:
        print(__doc__)
        sys.exit(0)


if __name__ == "__main__":
    main()
