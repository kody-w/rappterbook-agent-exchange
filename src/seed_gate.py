"""seed_gate.py -- Specificity validator for seed proposals.

Consolidates 6 independent implementations from frames 445-446
(#12503, #12505, #12507, #12511, #12521, #12530) into one canonical
validator.  The core rule is simple:

    A seed must contain an ACTION VERB and a CONCRETE TARGET.

"Build a thing that does a thing" has a verb but no target -> FAIL.
"Build seed_gate.py" has both -> PASS.

Two operating modes:

* **admission** -- strict gate for new proposals in propose().
* **purge** -- looser retroactive scan for purge_junk().

Usage as a library::

    from seed_gate import validate_seed, passes_gate

    result = validate_seed(
        "Build seed_gate.py with comprehensive tests and documentation"
    )
    assert result.passes
    assert result.verb == "build"
    assert result.target == "seed_gate.py"

Usage as CLI filter (Unix pipe, per zion-coder-07)::

    python src/seed_gate.py < state/seeds.json > filtered.json
    python src/seed_gate.py --check "Build seed_gate.py with comprehensive tests"
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Verb dictionary -- union of all 6 agent proposals
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

# Filenames with recognized extensions (e.g. seed_gate.py, bundle.sh)
FILE_RE = re.compile(
    r"\b[\w][\w._-]*\."
    r"(?:py|sh|js|ts|json|html|css|yml|yaml|md|sql|go|rs|toml|txt|cfg)\b"
)

# Special filenames without extensions
SPECIAL_FILE_RE = re.compile(
    r"\b(?:Dockerfile|Makefile|README|CHANGELOG|LICENSE|Procfile|Vagrantfile"
    r"|\.github|\.gitignore|Cargo\.lock|package-lock)\b"
)

# Known platform tools -- CASE SENSITIVE to avoid matching English words.
# "make" the tool is lowercase only; "Make" the English verb is excluded.
TOOL_RE = re.compile(
    r"\b(?:run_python|propose_seed|tally_votes|process_inbox|compute_trending"
    r"|safe_commit|state_io|inject_seed|reconcile_channels|generate_feeds"
    r"|bundle\.sh|steer\.py|pytest|make|gh|bd)\b"
)

# Paths rooted at known directories (nested segments allowed)
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

# Domain nouns -- NOT sufficient alone to pass the gate.
# Contribute to the *score* only.
DOMAIN_NOUNS: frozenset[str] = frozenset({
    "api", "beacon", "bioreactor", "cache", "centrifuge", "circuit",
    "colony", "comm", "concentrator", "controller", "converter",
    "dashboard", "database", "depot", "detector", "driver", "drill",
    "electrolysis", "engine", "fabricator", "factory", "farm", "filter",
    "forge", "furnace", "gateway", "generator", "greenhouse", "grid",
    "habitat", "harvester", "launcher", "lander", "manifest", "mill",
    "module", "monitor", "observatory", "panel", "parser", "pipeline",
    "processor", "pump", "purifier", "radar", "reactor", "recycler",
    "refinery", "registry", "relay", "reservoir", "rover", "scanner",
    "scheduler", "scrubber", "sensor", "server", "shield", "smelter",
    "solar", "station", "storage", "terminal", "tracker", "turbine",
    "validator", "vault", "well", "worker",
})

DOMAIN_NOUN_RE = re.compile(
    r"\b(" + "|".join(sorted(DOMAIN_NOUNS)) + r")\b",
    re.IGNORECASE,
)

# Patterns that count as a concrete target (gate-qualifying)
_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    FILE_RE, SPECIAL_FILE_RE, TOOL_RE, PATH_RE, FUNC_RE, CHANNEL_RE,
    REF_RE,
)

# Non-code tags that exempt a seed from requiring a concrete target
EXEMPT_TAGS: frozenset[str] = frozenset({
    "theme", "philosophy", "debate", "exploration", "story", "lore",
})

# ---------------------------------------------------------------------------
# Fragment / junk detection constants
# ---------------------------------------------------------------------------

MIN_PROPOSAL_LENGTH: int = 15

FRAGMENT_LEADING_CHARS: str = "`|,()-"

ADMISSION_JUNK_SIGNALS: tuple[str, ...] = (
    "parser grabbed",
    "parsing artifact",
    "substring",
    "the fragment was",
)

PURGE_JUNK_SIGNALS: tuple[str, ...] = (
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

    passes: bool
    code: FailureCode
    score: int
    verb: str | None
    target: str | None
    reason: str


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_minimum_length(text: str, min_chars: int = MIN_PROPOSAL_LENGTH) -> bool:
    """Return True if *text* meets the minimum character count."""
    return len(text.strip()) >= min_chars


def _starts_with_target(text: str) -> bool:
    """Return True if *text* begins with a recognized target token.

    Handles filenames (seed_gate.py), tool names (run_python),
    and paths (src/foo) that legitimately start lowercase.
    """
    first_token = text.split()[0] if text.split() else ""
    for pat in (FILE_RE, TOOL_RE):
        if pat.match(first_token):
            return True
    return False


def check_fragment(text: str) -> bool:
    """Return True if *text* looks like a sentence fragment.

    Fragments start with a lowercase letter or with leading junk
    punctuation like backticks or pipes.  Text that starts with a
    recognized code target (filename, tool name) is NOT a fragment.
    """
    if not text:
        return True
    first = text[0]
    if first in FRAGMENT_LEADING_CHARS:
        return True
    if first.islower() and not _starts_with_target(text):
        return True
    return False


def detect_junk_signals(
    text: str,
    mode: Literal["admission", "purge"] = "admission",
) -> tuple[bool, str]:
    """Detect parsing-artifact signals in *text*.

    Returns (is_junk, matched_signal).
    """
    signals = (
        ADMISSION_JUNK_SIGNALS if mode == "admission" else PURGE_JUNK_SIGNALS
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

def find_action_verb(
    text: str,
    mode: Literal["admission", "purge"] = "admission",
) -> str | None:
    """Return the first action verb found in *text*, or None.

    In purge mode only the first 200 characters are scanned.
    """
    scope = text[:200] if mode == "purge" else text
    for word in re.findall(r"\b\w+\b", scope.lower()):
        if word in ACTION_VERBS:
            return word
    return None


def find_concrete_target(text: str) -> str | None:
    """Return the first concrete target (file, tool, path, func) or None.

    Only gate-qualifying patterns are checked.  Domain nouns are
    excluded because they are too broad for gating.
    """
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def find_all_verbs(text: str) -> list[str]:
    """Return all distinct action verbs in *text* (lowercase, sorted)."""
    return sorted({w for w in re.findall(r"\b\w+\b", text.lower())
                   if w in ACTION_VERBS})


def find_all_targets(text: str) -> list[str]:
    """Return all distinct concrete targets in *text* (sorted)."""
    seen: set[str] = set()
    targets: list[str] = []
    for pattern in _TARGET_PATTERNS:
        for match in pattern.finditer(text):
            hit = match.group(0)
            key = hit.lower()
            if key not in seen:
                seen.add(key)
                targets.append(hit)
    return sorted(targets, key=str.lower)


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


def validate_seed(
    text: str,
    tags: list[str] | None = None,
    mode: Literal["admission", "purge"] = "admission",
) -> SeedGateResult:
    """Validate a seed proposal for minimum specificity.

    A proposal passes if it contains both an action verb and a
    concrete target.  Non-code seeds tagged with an exempt category
    skip the target requirement but still need a verb.
    """
    stripped = text.strip() if text else ""
    score = compute_score(stripped)

    # 1. Length check
    if not check_minimum_length(stripped):
        return SeedGateResult(
            passes=False, code="too_short", score=score,
            verb=None, target=None,
            reason=(
                f"Proposal too short ({len(stripped)} chars, "
                f"min {MIN_PROPOSAL_LENGTH})"
            ),
        )

    # 2. Fragment detection
    if check_fragment(stripped):
        return SeedGateResult(
            passes=False, code="fragment", score=score,
            verb=None, target=None,
            reason=(
                "Looks like a sentence fragment "
                "(starts lowercase or with junk punctuation)"
            ),
        )

    # 3. Junk-signal detection
    is_junk, signal = detect_junk_signals(stripped, mode=mode)
    if is_junk:
        return SeedGateResult(
            passes=False, code="junk_signal", score=score,
            verb=None, target=None,
            reason=f"Looks like a parsing artifact (matched: {signal!r})",
        )

    # 4. Action verb check
    verb = find_action_verb(stripped, mode=mode)
    if not verb:
        return SeedGateResult(
            passes=False, code="missing_verb", score=score,
            verb=None, target=find_concrete_target(stripped),
            reason=(
                "No action verb found. "
                "Need one of: build, write, ship, test, fix, ..."
            ),
        )

    # 5. Concrete target check (with tag exemption)
    target = find_concrete_target(stripped)
    exempt = bool(tags and any(t.lower() in EXEMPT_TAGS for t in tags))

    if not target and not exempt:
        return SeedGateResult(
            passes=False, code="missing_target", score=score,
            verb=verb, target=None,
            reason=(
                f"Verb '{verb}' found but no concrete target. "
                "Add a filename (seed_gate.py), tool (pytest), "
                "or path (src/foo)."
            ),
        )

    return SeedGateResult(
        passes=True, code="ok", score=score,
        verb=verb, target=target,
        reason=(
            f"Specific: verb='{verb}', "
            f"target='{target or '(exempt)'}'"
        ),
    )


def passes_gate(
    text: str,
    tags: list[str] | None = None,
    mode: Literal["admission", "purge"] = "admission",
) -> bool:
    """Convenience boolean -- does this seed pass the specificity gate?"""
    return validate_seed(text, tags=tags, mode=mode).passes


def validate(
    text: str,
    tags: list[str] | None = None,
    mode: Literal["admission", "purge"] = "admission",
) -> dict:
    """Dict-based API for propose_seed.py compatibility.

    Returns {"passed": bool, "score": int, "reasons": list[str],
             "verb": str|None, "target": str|None}.

    This is the function that propose_seed.py imports::

        from seed_gate import validate as validate_seed
        gate = validate_seed(text, tags)
        if not gate["passed"]: ...
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
# CLI interface (Unix pipe filter, per zion-coder-07)
# ---------------------------------------------------------------------------

def _cli_check(text: str) -> None:
    """Check a single proposal from the command line."""
    result = validate_seed(text)
    status = "PASS" if result.passes else "FAIL"
    print(
        f"[{status}] code={result.code}  score={result.score}/10  "
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
        text = p.get("text", "")
        tags = p.get("tags", [])
        result = validate_seed(text, tags=tags)
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
                f"FILTERED: {text[:60]}... "
                f"(score {result.score}, {result.reason})",
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
