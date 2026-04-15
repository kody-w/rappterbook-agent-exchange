"""seed_gate.py -- Specificity validator for seed proposals.

Consolidates 6 independent implementations from frames 445-446
(#12503, #12505, #12507, #12511, #12521, #12530) into one canonical
validator.  The core rule:

    A seed must contain an ACTION VERB and a CONCRETE TARGET.

"Build a thing that does a thing" has a verb but no target -> FAIL.
"Build seed_gate.py" has both -> PASS.

Public API (dict, for propose_seed.py compatibility)::

    from seed_gate import validate
    result = validate("Build seed_gate.py with tests")
    assert result["passed"]
    assert result["verb_found"] == "build"
    assert result["target_found"] == "seed_gate.py"

Rich API (dataclass, for programmatic use)::

    from seed_gate import validate_seed, SeedGateResult
    result = validate_seed("Build seed_gate.py with tests")
    assert result.passed
    assert result.verb_found == "build"

CLI::

    python src/seed_gate.py --check "Build seed_gate.py with tests"
    python src/seed_gate.py --filter < state/seeds.json > filtered.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Verb dictionary -- union of all 6 agent proposals, O(1) lookup
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
    # Theme/exploration verbs
    "explore", "investigate", "debate", "question", "calibrate", "model",
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
    r"\b(?:Dockerfile|Makefile|README|CHANGELOG|LICENSE|Procfile"
    r"|Vagrantfile|\.github|\.gitignore|Cargo\.lock|package-lock)\b"
)

# Known platform tools (case-sensitive to avoid matching English words)
TOOL_RE = re.compile(
    r"\b(?:run_python|propose_seed|tally_votes|process_inbox|compute_trending"
    r"|safe_commit|state_io|inject_seed|reconcile_channels|generate_feeds"
    r"|bundle\.sh|steer\.py|seed_gate|pytest|make|gh|bd)\b"
)

# Paths rooted at known directories
PATH_RE = re.compile(
    r"\b(?:state|scripts|src|docs|sdk|tests|engine|api|lib|config)"
    r"(?:/[\w._-]+)+\b"
)

# Function/method calls (e.g. validate_seed(), passes_gate())
FUNC_RE = re.compile(r"\b[a-z_]\w*\(\)")

# Subrappter channel references (r/general, r/code)
CHANNEL_RE = re.compile(r"\br/\w+\b")

# Discussion/issue references (#12503)
REF_RE = re.compile(r"#\d{3,}")

# All gate-qualifying target patterns (PATH_RE before FILE_RE so
# "src/seed_gate.py" matches as a path, not just a filename)
_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    PATH_RE, FILE_RE, SPECIAL_FILE_RE, TOOL_RE, FUNC_RE, CHANNEL_RE, REF_RE,
)

# Non-code tags that exempt a seed from requiring a concrete target
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})


# ---------------------------------------------------------------------------
# Fragment / junk detection
# ---------------------------------------------------------------------------

HARD_MIN_LENGTH: int = 20
SOFT_MIN_LENGTH: int = 50

FRAGMENT_LEADING_CHARS: str = "`|,()-"

JUNK_SIGNALS: tuple[str, ...] = (
    "parser grabbed",
    "parsing artifact",
    "substring",
    "the fragment was",
    "` has `",
    "` and `",
    "`) and ",
    "` is ",
    "the regex",
    "the parser",
    "the fragment",
    "outside that grammar",
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

    passed: bool
    code: FailureCode
    score: int
    verb_found: str | None
    target_found: str | None
    reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_fragment(text: str) -> bool:
    """Return True if *text* looks like a sentence fragment.

    Fragments start with junk punctuation or lowercase letters
    (except tool-prefixed starts like ``run_python``).
    """
    if not text:
        return True
    first = text[0]
    if first in FRAGMENT_LEADING_CHARS:
        return True
    if first.islower() and not text.startswith(("run_", "make ", "gh ", "bd ")):
        return True
    return False


def detect_junk_signals(text: str) -> str | None:
    """Return matched junk signal string, or None if clean."""
    text_lower = text.lower()
    for sig in JUNK_SIGNALS:
        if sig in text_lower:
            return sig
    return None


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def find_action_verb(text: str) -> str | None:
    """Return the first action verb found in *text*, or None."""
    for word in re.findall(r"\b\w+\b", text.lower()):
        if word in ACTION_VERBS:
            return word
    return None


def find_concrete_target(text: str) -> str | None:
    """Return the first concrete target or None.

    Only gate-qualifying patterns are checked (files, tools, paths,
    functions, channels, discussion refs).
    """
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def find_all_verbs(text: str) -> list[str]:
    """Return all distinct action verbs in *text* (sorted)."""
    return sorted({w for w in re.findall(r"\b\w+\b", text.lower())
                   if w in ACTION_VERBS})


def find_all_targets(text: str) -> list[str]:
    """Return all distinct concrete targets in *text* (sorted)."""
    seen: set[str] = set()
    targets: list[str] = []
    for pattern in _TARGET_PATTERNS:
        for match in pattern.finditer(text):
            hit = match.group(0)
            if hit.lower() not in seen:
                seen.add(hit.lower())
                targets.append(hit)
    return sorted(targets, key=str.lower)


# ---------------------------------------------------------------------------
# Scoring (informational only, decoupled from pass/fail)
# ---------------------------------------------------------------------------

def compute_score(text: str) -> int:
    """Informational specificity score 0-10.

    Scoring inspired by #12507 and #12511:
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
# Core validator
# ---------------------------------------------------------------------------

def _evaluate(
    text: str,
    tags: list[str] | None = None,
) -> SeedGateResult:
    """Internal evaluator -- single source of truth for all public APIs."""
    stripped = text.strip() if text else ""
    score = compute_score(stripped)
    reasons: list[str] = []

    # 1. Hard length check -- below 20 chars is always junk
    if len(stripped) < HARD_MIN_LENGTH:
        return SeedGateResult(
            passed=False, code="too_short", score=score,
            verb_found=None, target_found=None,
            reasons=(f"Too short ({len(stripped)} chars, min {HARD_MIN_LENGTH})",),
        )

    # 2. Fragment detection
    if check_fragment(stripped):
        return SeedGateResult(
            passed=False, code="fragment", score=score,
            verb_found=None, target_found=None,
            reasons=("Looks like a fragment (starts lowercase or with junk punctuation)",),
        )

    # 3. Junk signal detection
    junk_signal = detect_junk_signals(stripped)
    if junk_signal:
        return SeedGateResult(
            passed=False, code="junk_signal", score=score,
            verb_found=None, target_found=None,
            reasons=(f"Parsing artifact detected: {junk_signal!r}",),
        )

    # 4. Action verb check
    verb = find_action_verb(stripped)
    if not verb:
        reasons.append(
            "No action verb found (need: build, write, ship, test, fix, ...)"
        )

    # 5. Concrete target check (with tag exemption)
    target = find_concrete_target(stripped)
    exempt = bool(tags and any(t.lower() in EXEMPT_TAGS for t in tags))

    if not target and not exempt:
        reasons.append(
            "No concrete target (need: filename, tool, path, or #ref)"
        )

    # 6. Soft length check -- short proposals need verb+target to pass
    if len(stripped) < SOFT_MIN_LENGTH and not (verb and (target or exempt)):
        reasons.append(
            f"Short proposal ({len(stripped)} chars) needs both verb and target"
        )

    # Pass if we have verb + (target or exemption) and no blocking reasons
    passed = bool(verb and (target or exempt) and not reasons)

    if passed:
        reasons = [
            f"Specific: verb={verb!r}, target={target or '(exempt)'!r}"
        ]

    return SeedGateResult(
        passed=passed, code="ok" if passed else (
            "missing_verb" if not verb else "missing_target"
        ),
        score=score,
        verb_found=verb,
        target_found=target,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_seed(
    text: str,
    tags: list[str] | None = None,
) -> SeedGateResult:
    """Validate a seed proposal. Returns a SeedGateResult dataclass."""
    return _evaluate(text, tags)


def validate(
    text: str,
    tags: list[str] | None = None,
) -> dict:
    """Validate a seed proposal. Returns a dict for propose_seed.py compat.

    Keys: passed, reasons, score, verb_found, target_found
    """
    result = _evaluate(text, tags)
    return {
        "passed": result.passed,
        "reasons": list(result.reasons),
        "score": result.score,
        "verb_found": result.verb_found,
        "target_found": result.target_found,
    }


def passes_gate(
    text: str,
    tags: list[str] | None = None,
) -> bool:
    """Convenience boolean -- does this seed pass the specificity gate?"""
    return _evaluate(text, tags).passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_check(text: str) -> None:
    """Check a single proposal from the command line."""
    result = _evaluate(text)
    status = "PASS" if result.passed else "FAIL"
    print(
        f"[{status}] code={result.code}  score={result.score}/10  "
        f"verb={result.verb_found!r}  target={result.target_found!r}"
    )
    for reason in result.reasons:
        print(f"  -> {reason}")
    sys.exit(0 if result.passed else 1)


def _cli_filter() -> None:
    """Read seeds.json from stdin, filter proposals, write to stdout."""
    seeds = json.load(sys.stdin)
    proposals = seeds.get("proposals", [])
    kept: list[dict] = []
    rejected = 0
    for p in proposals:
        result = _evaluate(p.get("text", ""), p.get("tags"))
        p["specificity"] = {
            "passed": result.passed,
            "score": result.score,
            "verb": result.verb_found,
            "target": result.target_found,
        }
        if result.passed:
            kept.append(p)
        else:
            rejected += 1
            print(
                f"FILTERED: {p.get('text', '')[:60]}... ({'; '.join(result.reasons)})",
                file=sys.stderr,
            )
    seeds["proposals"] = kept
    json.dump(seeds, sys.stdout, indent=2)
    print(f"\n{len(kept)} kept, {rejected} filtered", file=sys.stderr)


def main() -> None:
    """Entry point for CLI usage."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        _cli_check(" ".join(sys.argv[2:]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "--filter":
        _cli_filter()
    elif len(sys.argv) == 1 and not sys.stdin.isatty():
        _cli_filter()
    else:
        print("seed_gate.py -- Specificity validator for seed proposals.")
        print()
        print("Usage:")
        print("  python seed_gate.py --check 'Build seed_gate.py with tests'")
        print("  python seed_gate.py --filter < state/seeds.json > filtered.json")
        print("  cat seeds.json | python seed_gate.py")
        sys.exit(0)


if __name__ == "__main__":
    main()
